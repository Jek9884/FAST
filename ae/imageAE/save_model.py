import ray
from ray import tune
import argparse
from pytorch_lightning import seed_everything
from utilities import normalize_images_tensor, log_scale_images_tensor
from atari_model import ImageAutoencoder
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from kfoldTrainer import KFoldTrainer
from image_kfold_dataset import ImageKfoldDataModule
from image_holdout_dataset import ImageHoldoutDataModule
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from highway_model import HighwayEnvModel
from ray.air.config import RunConfig
import os
import wandb

def trainable(config_dict):
    
    seed = 42
    seed_everything(seed, workers=True)

    images_tensor = ray.get(config_dict['data_pt'])

    if config_dict['normalize']:
        images_tensor = normalize_images_tensor(images_tensor)
    if config_dict['log_scale']:
        images_tensor = log_scale_images_tensor(images_tensor)

    # Initialize Weights & Biases run
    wandb.init(project="Master-Thesis", name = ray.train.get_context().get_trial_name(), config=config_dict)

    # Define the model
    if config_dict['model_type'] == "small":
        model = ImageAutoencoder(n_channels=config_dict['n_channels'], height=config_dict['height'], width=config_dict['width'], latent_dim=config_dict['embedding_dim'], lr=config_dict['lr'], weight_decay=config_dict['weight_decay'], eps=config_dict['eps'])
    else:
        model = HighwayEnvModel(n_channels=config_dict['n_channels'], height=config_dict['height'], width=config_dict['width'], latent_dim=config_dict['embedding_dim'], lr=config_dict['lr'], weight_decay=config_dict['weight_decay'], eps=config_dict['eps'])

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=config_dict["patience"],
        mode="min")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", 
        mode='min', 
        save_top_k=1, 
        save_last=True, 
        every_n_epochs=200)

    wandb_logger = WandbLogger(project="Master-Thesis", config=config_dict, settings=wandb.Settings(start_method="fork"))

    # Define a Lightning Trainer
    if config_dict['cross_validation'] == "Kfold":
        trainer = KFoldTrainer(max_epochs=config_dict['epochs'], num_folds=3, deterministic = True, enable_progress_bar = False, logger=wandb_logger, callbacks=[checkpoint_callback, early_stopping_callback])
        data_module = ImageKfoldDataModule(images_tensor, batch_size=config_dict['batch_size'], num_workers=10)
    elif config_dict['cross_validation'] == "Holdout":
        trainer = Trainer(max_epochs=config_dict['epochs'], deterministic = True, enable_progress_bar = False, logger=wandb_logger, callbacks=[checkpoint_callback, early_stopping_callback])
        data_module = ImageHoldoutDataModule(images_tensor, batch_size=config_dict['batch_size'], num_workers=10)
    trainer.fit(model, datamodule=data_module)

    wandb.finish()

parser = argparse.ArgumentParser(description="Train the image autoencoder")

# Define command-line arguments
parser.add_argument("--device", type=str, help="Device where to execute")
parser.add_argument("--file_path", type=str, help="Path to the dataset")
parser.add_argument("--save_path", type=str, help="Path to save model")
parser.add_argument("--normalize", type=int, help="Normalize data before computation")
parser.add_argument("--log_scale", type=int, help="Log scale data before computation")
parser.add_argument("--game", type=str, help="Game")

parser.add_argument("--model_type", type=int, help="0 for atari model or 1 for highway model")
parser.add_argument("--cross_validation", type=int, help="0 for Holdout CV or 1 for Kfold CV")
parser.add_argument("--criterion", type=str, help="Number of epochs to perform")
parser.add_argument("--epochs", type=int, help="Number of epochs to perform")
parser.add_argument("--emb_dim", type=int, help="Dimension of the latent space")
parser.add_argument("--batch_size", type=int, help="Batch size")
parser.add_argument("--lr", type=float, help="Learning rate")
parser.add_argument("--eps", type=float, help="Adam optimizer precision")
parser.add_argument("--weight_decay", type=float, help="Weight decay")
parser.add_argument("--patience", type=float, help="Patience for early stopping")
parser.add_argument("--divergence_threshold", type=float, help="Divergence threshold for early stopping")

# Parse the command-line arguments
args = parser.parse_args()
dev = args.device
file_path = args.file_path
normalize = args.normalize
log_scale = args.log_scale
game = args.game

criterion = args.criterion
epochs = args.epochs
emb_dim = args.emb_dim
batch_size = args.batch_size
lr = args.lr
eps = args.eps
weight_decay = args.weight_decay
patience = args.patience
divergence_threshold = args.divergence_threshold

if args.model_type == 0:
    model_type = "small"
elif args.model_type == 1:
    model_type = "highway"
else:
    raise ValueError("Model type parameter should be 0 or 1")

if args.cross_validation == 0:
    cross_validation = "Holdout"
elif args.cross_validation == 1:
    cross_validation = "Kfold"
else:
    raise ValueError("Cross validation parameter should be 0 or 1")

seed = 42
seed_everything(seed, workers=True)

if not os.path.exists("final_models"):
    os.makedirs("final_models")
    print(f"Directory 'final_models' created successfully.")


images_tensor = torch.load(file_path)

os.environ["CUDA_VISIBLE_DEVICES"] = dev
device = torch.device(0)

ray.shutdown()
ray.init()

data_pt = ray.put(images_tensor)

search_space = {
    'data_pt' : data_pt,
    'batch_size' : tune.grid_search([batch_size]),
    'lr' : tune.grid_search([lr]),
    'eps' : tune.grid_search([eps]),
    'weight_decay' : tune.grid_search([weight_decay]),
    'epochs' : tune.grid_search([epochs]),
    'embedding_dim' : tune.grid_search([emb_dim]),
    'n_channels' : tune.grid_search([images_tensor.shape[1]]),
    'height' : tune.grid_search([images_tensor.shape[2]]),
    'width' : tune.grid_search([images_tensor.shape[3]]),
    'criterion' : tune.grid_search([criterion]),
    'metric' : tune.grid_search(['mse']),
    'game' : tune.grid_search([game]),
    'patience' : tune.grid_search([patience]),
    'divergence_threshold' : tune.grid_search([divergence_threshold]),
    'n_samples' : tune.grid_search([images_tensor.shape[0]]),
    'model_type' : tune.grid_search([model_type]),
    'normalize' : tune.grid_search([normalize]),
    'log_scale' : tune.grid_search([log_scale]),
    'seed' : tune.grid_search([seed]),
    'cross_validation' : tune.grid_search([cross_validation])
}

tuner = tune.Tuner(tune.with_resources(trainable, 
                                       {"cpu":10,"gpu":1},
                                       ),
                    param_space = search_space,
                    run_config = RunConfig(name=game, verbose=1)
                    )

results = tuner.fit()

ray.shutdown()