from collections import defaultdict
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
import copy
import json
import math
import numpy as np
import os
import re
import sys
import torch

class ChenCOCOCIDErReward:
    def __init__(self, labels_file_path):
        self.tokenizer = PTBTokenizer()
        self.labels_file_path = labels_file_path

        with open(self.labels_file_path) as f:
            examples = json.load(f)

        self.labels = {}
        for i in examples["train"]:
            for j in i["image_path"]:
                self.labels[j] = [{"caption": re.sub("\s+", " ", i["report"])}]
        with suppress_output(suppress_stdout=True, suppress_stderr=True):
            self.labels = self.tokenizer.tokenize(self.labels)

        self.labels = [v[0] for k, v in self.labels.items()]

        self.cider = CiderD(df="dataset", dataset=self.labels)

    def __call__(self, predictions, labels, **kwargs):
        return self.reward(predictions, labels, **kwargs)

    def reward(self, predictions, labels, **kwargs):

        predictions_dict, labels_dict = {}, {}
        for i, (j, k) in enumerate(zip(predictions, labels)):
            predictions_dict[i] = [{"caption": j}]
            labels_dict[i] = [{"caption": k}]

        with suppress_output(suppress_stdout=True, suppress_stderr=True):
            predictions = self.tokenizer.tokenize(predictions_dict)
            labels = self.tokenizer.tokenize(labels_dict)

        predictions = [{"image_id": k, "caption": v} for k, v in predictions.items()]

        _, scores = self.cider.compute_score(gts=labels, res=predictions)

        return torch.from_numpy(scores)

class CiderD:
    """
    CIDErD metric. This is a modification of https://github.com/vrama91/cider/tree/master/pyciderevalcap/ciderD.
    """
    def __init__(self, n=4, sigma=6.0, df="corpus", dataset=None):
        # set cider to sum over 1 to 4-grams
        self._n = n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma
        # set which where to compute document frequencies from
        self._df = df
        self.cider_scorer = CiderScorer(n=self._n, df_mode=self._df, dataset=dataset)

    def compute_score(self, gts, res):
        """
        Main function to compute CIDEr score
        """

        # clear all the previous hypos and refs
        tmp_cider_scorer = self.cider_scorer.copy_empty()
        tmp_cider_scorer.clear()
        for res_id in res:

            hypo = res_id['caption']
            ref = gts[res_id['image_id']]

            # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)
            tmp_cider_scorer += (hypo[0], ref)

        (score, scores) = tmp_cider_scorer.compute_score()

        return score, scores

    def method(self):
        return "CIDEr-D"

def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4):  ## lhuang: oracle will call with "average"
    """Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    """
    return [precook(ref, n) for ref in refs]


def cook_test(test, n=4):
    """Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.
    :param test: list of string : hypothesis sentence for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (dict)
    """
    return precook(test, n, True)

class CiderScorer(object):
    """
    CIDEr scorer. This is a modification of https://github.com/vrama91/cider/tree/master/pyciderevalcap/ciderD.
    """

    def __init__(self, df_mode="corpus", test=None, refs=None, n=4, sigma=6.0, dataset=None):
        """ singular instance """
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.df_mode = df_mode
        self.ref_len = None
        if self.df_mode == "dataset":
            self.ref_len = np.log(float(len(dataset)))
            self.cook_append(None, dataset)
            self.document_frequency = defaultdict(float)
            self.compute_doc_freq()
            self.crefs = []
        self.cook_append(test, refs)

    def copy(self):
        """ copy the refs."""
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def copy_empty(self):
        new = CiderScorer(df_mode="corpus", n=self.n, sigma=self.sigma)
        new.df_mode = self.df_mode
        new.ref_len = self.ref_len
        new.document_frequency = self.document_frequency
        return new

    def clear(self):
        self.crefs = []
        self.ctest = []

    def cook_append(self, test, refs):
        """called by constructor and __iadd__ to avoid creating new instances."""

        if refs is not None:
            self.crefs.append(cook_refs(refs))
            if test is not None:
                self.ctest.append(cook_test(test))  ## N.B.: -1
            else:
                self.ctest.append(None)  # lens of crefs and ctest have to match

    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        """add an instance (e.g., from another sentence)."""

        if type(other) is tuple:
            ## avoid creating new CiderScorer instances
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)

        return self

    def compute_doc_freq(self):
        """
        Compute term frequency for reference data.
        This will be used to compute idf (inverse document frequency later)
        The term frequency is stored in the object
        :return: None
        """
        for refs in self.crefs:
            for ngram, count in [(ngram, count) for ref in refs for (ngram, count) in ref.items()]:
                self.document_frequency[ngram] += count

    def compute_cider(self):
        def counts2vec(cnts):
            """
            Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.
            :param cnts:
            :return: vec (array of dict), norm (array of float), length (int)
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram, term_freq) in cnts.items():
                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(max(1.0, self.document_frequency[ngram]))
                # ngram index
                n = len(ngram) - 1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq) * (self.ref_len - df)
                # compute norm for the vector.  the norm will be used for computing similarity
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            """
            Compute the cosine similarity of two vectors.
            :param vec_hyp: array of dictionary for vector corresponding to hypothesis
            :param vec_ref: array of dictionary for vector corresponding to reference
            :param norm_hyp: array of float for vector corresponding to hypothesis
            :param norm_ref: array of float for vector corresponding to reference
            :param length_hyp: int containing length of hypothesis
            :param length_ref: int containing length of reference
            :return: array of score for each n-grams cosine similarity
            """
            delta = float(length_hyp - length_ref)
            # measure consine similarity
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                # ngram
                for (ngram, count) in vec_hyp[n].items():
                    # vrama91 : added clipping
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= (norm_hyp[n] * norm_ref[n])

                assert (not math.isnan(val[n]))
                # vrama91: added a length based gaussian penalty
                val[n] *= np.e ** (-(delta ** 2) / (2 * self.sigma ** 2))
            return val

        # compute log reference length
        if self.df_mode == "corpus":
            self.ref_len = np.log(float(len(self.crefs)))

        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            # compute vector for test captions
            vec, norm, length = counts2vec(test)
            # compute vector for ref captions
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            # change by vrama91 - mean of ngram scores, instead of sum
            score_avg = np.mean(score)
            # divide by number of references
            score_avg /= len(refs)
            # multiply score by 10
            score_avg *= 10.0
            # append score of an image to the score list
            scores.append(score_avg)
        return scores

    def compute_score(self, option=None, verbose=0):
        # compute idf
        if self.df_mode == "corpus":
            self.document_frequency = defaultdict(float)
            self.compute_doc_freq()
            # assert to check document frequency
            assert (len(self.ctest) >= max(self.document_frequency.values()))
            # import json for now and write the corresponding files
        # compute cider score
        score = self.compute_cider()
        # debug
        # print score
        return np.mean(np.array(score)), np.array(score)

class suppress_output:
    def __init__(self, suppress_stdout=False, suppress_stderr=False):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self._stdout = None
        self._stderr = None

    def __enter__(self):
        devnull = open(os.devnull, "w")
        if self.suppress_stdout:
            self._stdout = sys.stdout
            sys.stdout = devnull

        if self.suppress_stderr:
            self._stderr = sys.stderr
            sys.stderr = devnull

    def __exit__(self, *args):
        if self.suppress_stdout:
            sys.stdout = self._stdout
        if self.suppress_stderr:
            sys.stderr = self._stderr