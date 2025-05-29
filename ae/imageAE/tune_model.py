import argparse
import torch
import os
from ray import tune
import ray
import wandb
from ray.air.config import RunConfig
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from ray.air.integrations.wandb import setup_wandb
from lightning import seed_everything

from atari_model import ImageAutoencoder
from highway_model import HighwayEnvModel
from image_kfold_dataset import ImageKfoldDataModule
from image_holdout_dataset import ImageHoldoutDataModule
from kfoldTrainer import KFoldTrainer

import sys
sys.path.insert(1, '/storagenfs/a.capurso1/Master-Thesis/ae/lightning_version/data_handler')
from utilities import normalize_images_tensor, log_scale_images_tensor

os.environ["WANDB__SERVICE_WAIT"] = "600"

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
        mode="min"
    )

    model_checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", 
        mode='min', 
        save_top_k=1, 
        save_last=True, 
        every_n_epochs=200)
    
    wandb_logger = WandbLogger(project="Master-Thesis", config=config_dict, settings=wandb.Settings(start_method="fork"))

    # Define a Lightning Trainer
    if config_dict['cross_validation'] == "Kfold":
        trainer = KFoldTrainer(max_epochs=config_dict['epochs'], num_folds=3, deterministic = True, enable_progress_bar = False, logger=wandb_logger, callbacks=[early_stopping_callback, model_checkpoint_callback])
        data_module = ImageKfoldDataModule(images_tensor, batch_size=config_dict['batch_size'], num_workers=10)
    elif config_dict['cross_validation'] == "Holdout":
        trainer = Trainer(max_epochs=config_dict['epochs'], deterministic = True, enable_progress_bar = False, logger=wandb_logger, callbacks=[early_stopping_callback, model_checkpoint_callback])
        data_module = ImageHoldoutDataModule(images_tensor, batch_size=config_dict['batch_size'], num_workers=10)
    trainer.fit(model, datamodule=data_module)

    wandb.finish()

parser = argparse.ArgumentParser(description="Train the image autoencoder")

# Define command-line arguments
parser.add_argument("--device", type=str, help="Device where to execute")
parser.add_argument("--model_type", type=int, help="0 for atari model or 1 for highway model")
parser.add_argument("--file_path", type=str, help="Path to the dataset")
parser.add_argument("--cross_validation", type=int, help="0 for Holdout CV or 1 for Kfold CV")

# Parse the command-line arguments
args = parser.parse_args()

dev = args.device
if args.model_type == 0:
    model_type = "small"
elif args.model_type == 1:
    model_type = "highway"
else:
    raise ValueError("Model type parameter should be 0 or 1")

file_path = args.file_path

if args.cross_validation == 0:
    cross_validation = "Holdout"
elif args.cross_validation == 1:
    cross_validation = "Kfold"
else:
    raise ValueError("Cross validation parameter should be 0 or 1")

seed = 42
seed_everything(seed, workers=True)

images_tensor = torch.load(file_path)
#images_tensor = images_tensor[:10, :, :, :]

os.environ["CUDA_VISIBLE_DEVICES"] = dev
device = torch.device(0)

# Configure Ray Tune
ray.shutdown()
ray.init(num_cpus=5)

data_pt = ray.put(images_tensor)

game = "Intersection" #"Full", IceHockey, Pong, Alien, SpaceInvaders, AirRaid

# Define the search space
search_space = {
    'data_pt' : data_pt,
    'batch_size' : tune.grid_search([64, 128]),
    'lr' : tune.grid_search([1e-3, 1e-5, 1e-7]),
    'eps' : tune.grid_search([1e-8]),
    'weight_decay' : tune.grid_search([1e-3, 1e-5, 1e-7]),
    'epochs' : tune.grid_search([1000]),
    'embedding_dim' : tune.grid_search([128]),
    'n_channels' : tune.grid_search([images_tensor.shape[1]]),
    'height' : tune.grid_search([images_tensor.shape[2]]),
    'width' : tune.grid_search([images_tensor.shape[3]]),
    'criterion' : tune.grid_search(['mse']),
    'metric' : tune.grid_search(['mse']),
    'game' : tune.grid_search([game]),
    'patience' : tune.grid_search([5]),
    'divergence_threshold' : tune.grid_search([1e-5]),
    'n_samples' : tune.grid_search([images_tensor.shape[0]]),
    'model_type' : tune.grid_search([model_type]),
    'normalize' : tune.grid_search([1]),
    'log_scale' : tune.grid_search([0]),
    'seed' : tune.grid_search([seed]),
    'cross_validation' : tune.grid_search([cross_validation])
}

tuner = tune.Tuner(tune.with_resources(trainable, 
                                       {"cpu":5,"gpu":1},
                                       ),
                    param_space = search_space,
                    run_config = RunConfig(name=game, verbose=1)
                )

results = tuner.fit()

ray.shutdown()
