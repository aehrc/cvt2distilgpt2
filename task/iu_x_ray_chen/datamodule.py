from collections import Counter
from transmodal.datamodule import DataModule
from task.iu_x_ray_chen.dataset import TaskSubset
import json
import os
import re


class TaskDataModule(DataModule):
    """
    Module for the task's dataset. See the documentation here:
    https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    """

    def __init__(self, self_critical=False, **kwargs):
        """
        Argument/s:
            kwargs - keyword arguments.
        """
        super().__init__(**kwargs)
        self.self_critical = self_critical

        self.labels_file_path = os.path.join(
            self.dataset_dir, "iu_x-ray_chen", "annotation.json"
        )
        self.dataset_dir = os.path.join(self.dataset_dir, "iu_x-ray_chen", "images")
        self.train_dicom_ids = []
        self.val_dicom_ids = []
        self.test_dicom_ids = []
        self.chen_tokenizer = TokenizerChen(
            ann_path=self.labels_file_path,
            threshold=3,
        )
        self.chen_max_seq_length = 60

    def setup(self, stage=None):
        """
        Dataset preparation.

        Argument/s:
            stage - either 'fit' (training & validation sets) or 'test'
                (test set).
        """

        with open(self.labels_file_path) as f:
            examples = json.load(f)

        # Dataset statistics
        images = set()
        for i in examples["train"]:
            images.update(i["image_path"])
        print(
            "Training set #images: {}, #studies: {}".format(
                len(images), len(examples["train"])
            )
        )

        images = set()
        for i in examples["val"]:
            images.update(i["image_path"])
        print(
            "Validation set #images: {}, #studies: {}".format(
                len(images), len(examples["val"])
            )
        )

        images = set()
        for i in examples["test"]:
            images.update(i["image_path"])
        print(
            "Test set #images: {}, #studies: {}".format(
                len(images), len(examples["test"])
            )
        )

        # Assign train & validation sets
        if stage == "fit" or stage is None:
            self.train_set = TaskSubset(
                examples=self.format_examples(examples["train"]),
                tokenizer=self.tokenizer,
                decoder_max_len=self.decoder_max_len,
                colour_space=self.colour_space,
                transforms=self.train_transforms,
                self_critical=self.self_critical,
                add_bos_eos_manually=self.add_bos_eos_manually,
                train=True,
            )

            self.val_set = TaskSubset(
                examples=self.format_examples(examples["val"]),
                tokenizer=self.tokenizer,
                decoder_max_len=self.decoder_max_len,
                colour_space=self.colour_space,
                transforms=self.test_transforms,
                add_bos_eos_manually=self.add_bos_eos_manually,
            )
            print(
                "IU X-Ray Chen's labels no. of training & validation examples: {} & {}.".format(
                    self.train_set.__len__(), self.val_set.__len__()
                )
            )

        # Assign test set
        if stage == "test" or stage is None:
            self.test_set = TaskSubset(
                examples=self.format_examples(examples["test"]),
                tokenizer=self.tokenizer,
                decoder_max_len=self.decoder_max_len,
                colour_space=self.colour_space,
                transforms=self.test_transforms,
                add_bos_eos_manually=self.add_bos_eos_manually,
            )
            print(
                "IU X-Ray Chen's labels no. of test examples: {}.".format(
                    self.test_set.__len__()
                )
            )

    def format_examples(self, examples):
        for i in examples:
            i["image_file_path"] = i.pop("image_path")
            i["label"] = i.pop("report")
            i["image_file_path"] = [os.path.join(self.dataset_dir, j) for j in i["image_file_path"]]
            i["label"] = self.chen_tokenizer(i["label"])[:self.chen_max_seq_length]
            i["label"] = self.chen_tokenizer.decode(i["label"][1:])
        return examples


class TokenizerChen(object):
    def __init__(self, ann_path, threshold):
        self.ann_path = ann_path
        self.threshold = threshold
        self.clean_report = self.clean_report_iu_xray
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.token2idx, self.idx2token = self.create_vocabulary()

    def create_vocabulary(self):
        total_tokens = []

        for example in self.ann['train']:
            tokens = self.clean_report(example['report']).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out
