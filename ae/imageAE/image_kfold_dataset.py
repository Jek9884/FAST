import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from lightning import LightningDataModule

class ImageKfoldDataModule(LightningDataModule):
    def __init__(self, images_list, batch_size=32, num_folds=5, num_workers=0, seed=None):
        super().__init__()
        self.images_list = images_list
        self.batch_size = batch_size
        self.num_folds = num_folds
        self.num_workers = num_workers
        self.seed = seed

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.dataset = ImageKfoldDataset(self.images_list, self.seed)

        # Generate k-fold loaders
        self.kfold_loaders = self.dataset.get_kfold_loaders(num_folds=self.num_folds, batch_size=self.batch_size, num_workers=self.num_workers)

    def train_dataloader(self):
        return self.kfold_loaders[self.trainer.current_fold][0]

    def val_dataloader(self):
        return self.kfold_loaders[self.trainer.current_fold][1]
    

class ImageKfoldDataset(Dataset):
    def __init__(self, images_list, transform=None, seed=None):
        self.images_list = images_list
        self.transform = transform
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)
        
        self.indices = np.arange(len(self.images_list))
        np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image = self.images_list[idx]

        if self.transform:
            image = self.transform(image)

        # Convert image to float tensor
        image = image.float()

        return image

    def get_kfold_loaders(self, num_folds=5, batch_size=32, num_workers=0):
        kfold_loaders = []
        fold_size = len(self) // num_folds

        for fold in range(num_folds):
            val_indices = self.indices[fold * fold_size : (fold + 1) * fold_size]
            train_indices = np.concatenate([self.indices[:fold * fold_size], self.indices[(fold + 1) * fold_size:]])

            train_dataset = Subset(self, train_indices)
            val_dataset = Subset(self, val_indices)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

            kfold_loaders.append((train_loader, val_loader))

        return kfold_loaders
    
