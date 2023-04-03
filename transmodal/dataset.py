from PIL import Image
from torch.utils.data import Dataset
from wurlitzer import pipes
import random


class Subset(Dataset):
    """
    The base class used to form a subset of the task's dataset. Implemented
    using a torch.utils.data.Dataset for the torch.utils.data.DataLoader for
    the subset of the pytorch_lightning.core.datamodule.LightningDataModule.
    See the tutorial here:
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(
        self,
        examples=None,
        transforms=None,
        colour_space=None,
        tokenizer=None,
        decoder_max_len=None,
        self_critical=False,
        train=False,
        add_bos_eos_manually=False,
        num_samples=None,
        sample_seed=43,
        **kwargs,
    ):
        """
        Argument/s:
            examples - a list of dictionaries, where each dictionary corresponds to
                an example and has keys that are relevant to the dataset.
            transforms - torchvision transforms to be applied to each image.
            colour_space - color space of the images: "L" (greyscale) or "RGB".
            tokenizer - sentence tokenizer.
            decoder_max_len - maximum length for the decoder's input (training).
            self_critical - self-critical sequence training flag.
            train - training flag.
            add_bos_eos_manually - add the beginning of sentence and end of sentence tokens manually.
            num_samples - subset size of the set.
            sample_seed - seed for the sample.
            kwargs - keyword arguments.
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.decoder_max_len = decoder_max_len
        self.transforms = transforms
        self.colour_space = colour_space
        self.self_critical = self_critical
        self.train = train
        self.add_bos_eos_manually = add_bos_eos_manually

        # Issue:
        #
        # Currently, MONAI does not pass the transformed image to NormalizeIntensity with Compose, thus not allowing
        # multi-channel normalisation. A work around is to apply NormalizeIntensity after Compose. The key for this
        # will be 'Normalisation' and will be popped so that it is not in Compose. As the normalisation is typically
        # the same for training and testing, will get only from training. This is implemented in transmodal/datamodule.py
        self.normalisation = kwargs['normalisation'] if 'normalisation' in kwargs else None

        if num_samples:
            random.seed(sample_seed)
            print(f"Number of examples in dataset: {len(self.examples)}.")
            self.examples = random.sample(self.examples, num_samples)
            print(f"Number of examples in subset of dataset: {len(self.examples)}.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        raise NotImplementedError

    def image_loading_and_preprocessing(self, image_file_path):
        """
        Load and pre-process an image.

        Argument/s:
            image_file_path - file path to the image.

        Returns:
            image - tensor of the preprocessed image.
        """
        image = Image.open(image_file_path)
        image = image.convert(self.colour_space)  # "L" (greyscale) or "RGB".
        if self.transforms is not None:
            image = self.transforms(image)
        return image

    def dicom_load_and_preprocess(self, image_file_path):
        """
        Load and pre-process a DICOM image.

        Argument/s:
            image_file_path - file path to the image.

        Returns:
            image - tensor of the preprocessed image.
        """
        with pipes() as (out, err):
            image = self.transforms(image_file_path)
            if self.normalisation:
                image = self.normalisation(image)
        return image

    def tokenize(self, string):

        if self.add_bos_eos_manually:
            string = self.tokenizer.bos_token + string + self.tokenizer.eos_token

        tokenized = self.tokenizer(
            string,
            padding="max_length",
            truncation=True,
            max_length=self.decoder_max_len + 1,  # As we remove a token below.
            return_tensors="pt",
        )

        example_dict = {"decoder_input_ids": tokenized.input_ids[0]}
        if "token_type_ids" in tokenized:
            example_dict["decoder_token_type_ids"] = tokenized.token_type_ids[0][1:]
        example_dict["decoder_attention_mask"] = tokenized.attention_mask[0][1:]
        example_dict["label_ids"] = (
            example_dict["decoder_input_ids"][1:].detach().clone()
        )
        example_dict["decoder_input_ids"] = example_dict["decoder_input_ids"][:-1]
        example_dict["decoder_input_ids"][
            example_dict["decoder_input_ids"] == self.tokenizer.sep_token_id
        ] = self.tokenizer.pad_token_id

        return example_dict
