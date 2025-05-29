import numpy as np
from torch.utils.data import random_split, DataLoader
from lightning import LightningDataModule

class ImageHoldoutDataModule(LightningDataModule):
    def __init__(self, images_list, batch_size=32, split=0.8, num_workers=0, seed=None):
        super().__init__()
        self.images_list = images_list
        self.batch_size = batch_size
        self.split = split
        self.num_workers = num_workers
        self.seed = seed

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        # Define the split sizes
        train_size = int(0.8 * len(self.images_list))
        val_size = len(self.images_list) - train_size

        # Split the dataset
        train_dataset, val_dataset = random_split(self.images_list, [train_size, val_size])

        # Create DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader