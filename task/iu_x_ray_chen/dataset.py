from transmodal.dataset import Subset
import re
import torch

class TaskSubset(Subset):
    """
    A subset of the task's dataset. Implemented using a torch.utils.data.Dataset
    for the torch.utils.data.DataLoader for the subset of the
    pytorch_lightning.core.datamodule.LightningDataModule. See the tutorial
    here: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __getitem__(self, index):
        example = self.examples[index]
        image_1 = self.image_loading_and_preprocessing(example["image_file_path"][0])
        image_2 = self.image_loading_and_preprocessing(example["image_file_path"][1])
        image = torch.stack((image_1, image_2), 0)
        example_dict = {"id": example["id"], "encoder_images": image, "labels": example["label"]}
        if self.train and not self.self_critical:
            example_dict = {**example_dict, **self.tokenize(example["label"])}
        return example_dict