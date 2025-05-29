from lightning import Trainer

class KFoldTrainer(Trainer):
    def __init__(self, *args, num_folds=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_folds = num_folds

    def fit(self, model, datamodule):
        for fold_index in range(self.num_folds):
            self.current_fold = fold_index  # Set the current fold index
            super().fit(model, datamodule=datamodule)
