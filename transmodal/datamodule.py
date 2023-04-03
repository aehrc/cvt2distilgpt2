from monai.transforms import NormalizeIntensity
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader
import importlib


class DataModule(LightningDataModule):
    """
    Base data module for the task. LightningDataModule is the parent of this module. See the documentation here:
    https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    """

    def __init__(
            self,
            mbatch_size,
            dataset_dir,
            train_transforms={"ToTensor": {}},
            test_transforms={"ToTensor": {}},
            train_sitk_transforms=None, # Potentially remove in favour of monai.
            test_sitk_transforms=None, # Potentially remove in favour of monai.
            transform_lib="torchvision",
            tokenizer=None,
            colour_space=None,
            decoder_max_len=None,
            prefetch_factor=5,
            val_split=0.05,
            random_split_seed=43,
            preceding_num_classes=0,
            succeeding_num_classes=0,
            num_workers=0,
            add_bos_eos_manually=False,
            num_samples=None,
            sample_seed=43,
            **kwargs,
    ):
        """
        Argument/s:
            mbatch_size - mini-batch size.
            dataset_dir - path to dataset.
            train_transforms - dict of transform class names with there
                respective keyword arguments for the training images.
            test_transforms - dict of transform class names with there
                respective keyword arguments for the test (and validation)
                images.
            train_sitk_preprocess - dict of SimpleITK pre-processing operations # Potentially remove in favour of monai.
                for the training images.
            test_sitk_preprocess - dict of SimpleITK pre-processing operations # Potentially remove in favour of monai.
                for the test images.
            transform_lib - where the image transformations are imported from (e.g. torchvision or monai).
            tokenizer - sentence tokenizer.
            colour_space - color space of the images: "L" (greyscale) or "RGB".
            decoder_max_len - maximum length for the decoder's input (training).
            prefetch_factor - no. of samples pre-loaded by each worker, i.e.
                prefetch_factor multiplied by num_workers samples are prefetched
                over all workers.
            val_split - fraction of training examples to be used as the
                validation set.
            random_split_seed - seed for random split.
            preceding_num_classes - the number of classes of the preceding datasets.
            succeeding_num_classes - the number of classes of the succeeding datasets.
            num_workers - number of subprocesses to use for DataLoader. 0 means
                that the data will be loaded in the main process.
            add_bos_eos_manually - add the beginning of sentence and end of sentence tokens manually.
            num_samples - subset size of the set.
            sample_seed - seed for the sample.
            kwargs - keyword arguments.
        """
        super().__init__()
        self.mbatch_size = mbatch_size
        self.dataset_dir = dataset_dir
        self.train_sitk_transforms = train_sitk_transforms  # Potentially remove in favour of monai.
        self.test_sitk_transforms = test_sitk_transforms # Potentially remove in favour of monai.
        self.tokenizer = tokenizer
        self.colour_space = colour_space
        self.decoder_max_len = decoder_max_len
        self.prefetch_factor = prefetch_factor
        self.val_split = val_split
        self.random_split_seed = random_split_seed
        self.preceding_num_classes = preceding_num_classes
        self.succeeding_num_classes = succeeding_num_classes
        self.num_workers = num_workers
        self.add_bos_eos_manually = add_bos_eos_manually
        self.num_samples = num_samples
        self.sample_seed = sample_seed

        transforms = getattr(importlib.import_module(transform_lib), "transforms")

        # Issue:
        #
        # Currently, MONAI does not pass the transformed image to NormalizeIntensity with Compose, thus not allowing
        # multi-channel normalisation. A work around is to apply NormalizeIntensity after Compose. The key for this
        # will be 'Normalisation' and will be popped so that it is not in Compose. As the normalisation is typically
        # the same for training and testing, will get only from training.
        if transform_lib == 'monai':
            train_transforms = train_transforms.copy()
            test_transforms = test_transforms.copy()
            if "Normalisation" in train_transforms:
                self.normalisation = NormalizeIntensity(
                    subtrahend=train_transforms['Normalisation']['subtrahend'],
                    divisor=train_transforms['Normalisation']['divisor'],
                    channel_wise=train_transforms['Normalisation']['channel_wise'],
                )
            train_transforms.pop('Normalisation', None)
            test_transforms.pop('Normalisation', None)

        self.train_transforms = (
            transforms.Compose(
                [getattr(transforms, k)(**v) for k, v in train_transforms.items()]
            )
            if train_transforms
            else None
        )

        self.test_transforms = (
            transforms.Compose(
                [getattr(transforms, k)(**v) for k, v in test_transforms.items()]
            )
            if test_transforms
            else None
        )

        if self.tokenizer:
            for i, j in zip(self.tokenizer.all_special_tokens, self.tokenizer.all_special_ids):
                print(f"{i}: {j}")

    def train_dataloader(self, shuffle=True):
        """
        Training set DataLoader.

        Argument/s:
            shuffle - shuffle the order of the examples.

        Returns:
            DataLoader.
        """
        return DataLoader(
            self.train_set,
            batch_size=self.mbatch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn
            if callable(getattr(self, "collate_fn", None))
            else None,
        )

    def val_dataloader(self, shuffle=False):
        """
        Validation set DataLoader.

        Argument/s:
            shuffle - shuffle the order of the examples.

        Returns:
            DataLoader.
        """
        return DataLoader(
            self.val_set,
            batch_size=self.mbatch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn
            if callable(getattr(self, "collate_fn", None))
            else None,
        )

    def test_dataloader(self, shuffle=False):
        """
        Test set DataLoader.

        Argument/s:
            shuffle - shuffle the order of the examples.

        Returns:
            DataLoader.
        """
        return DataLoader(
            self.test_set,
            batch_size=self.mbatch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn
            if callable(getattr(self, "collate_fn", None))
            else None,
        )
