import torch
import os
import ray
from ray import tune
from ray.air.config import RunConfig
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
from lightning import seed_everything
import sys
sys.path.insert(1, "../")
from stable_baselines3.sac.sac import SACMaster
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

def dict_to_string(d):
    return '_'.join(str(value) for value in d.values())

def trainable(config_dict):

    path = "../../../../../../../../data/a.capurso/Master-Thesis/transfer_learning"
    save_path = config_dict['save_path']
    config_dict.pop('save_path')

    run = wandb.init(
        project="SAC_train_new_reward",
        config=config_dict,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,  # optional
    )

    KINEMATICS_OBSERVATION = {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy", "heading", "long_off", "lat_off", "ang_off"],
        "absolute": False,
        "order": "sorted",
    }

    config = {
        "observation": KINEMATICS_OBSERVATION,
        "action": {
            "type": "ContinuousAction",
        },
        "policy_frequency": 5, 
        "vehicles_count": 5,
        # "real_time_rendering":True,
    }

    # Init env
    env = make_vec_env(config_dict['env_name'], n_envs=config_dict["n_envs"], env_kwargs={'config': config}, vec_env_cls=SubprocVecEnv)

    # Without transfer learning
    agent = SACMaster(env=env, policy='MlpPolicy', learning_rate=config_dict['lr'], batch_size=1024, tau=config_dict['tau'], gamma=config_dict['gamma'], gradient_steps=10, 
                        train_freq=15, use_sde=False, device=config_dict["device"], seed=config_dict['seed'], tensorboard_log=f"{path}/tensorboard_log")

    agent.learn(total_timesteps=config_dict["total_timesteps"], progress_bar=True,
                callback=WandbCallback(
                    gradient_save_freq=100,
                    verbose=2
                ))    
    
    agent.save(f"{path}/{save_path}/{dict_to_string(config_dict)}")

    wandb.finish()
    


parser = argparse.ArgumentParser(description="Train a RL agent with SAC algorithm")

# Define command-line arguments
parser.add_argument("--env_name", type=str, help="Environment to use")
parser.add_argument("--device", type=str, help="Device where to execute")
parser.add_argument("--n_envs", type=int, default=1, help="Number of environments to use in parallel")
parser.add_argument("--seed", type=int, default=42, help="Seed to use for replicability")
parser.add_argument("--save_path", type=str, help="Path where to save the models")

# Parse the command-line arguments
args = parser.parse_args()

env_name = args.env_name#"racetrack-v0"
device = args.device
n_envs = args.n_envs
seed = args.seed
save_path = args.save_path

seed_everything(seed, workers=True)

n_gpus = 0.5

if device != "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    device = torch.device(0)

ray.shutdown()
ray.init(num_cpus=n_envs)

search_space = {
    "env_name" : env_name,
    'lr' : tune.grid_search([1e-3]),
    "tau" : tune.grid_search([0.5, 0.7, 0.9]),
    "gamma" : tune.grid_search([0.7, 0.8, 0.99]),
    "total_timesteps" : 1e7,
    "n_envs" : n_envs,
    "seed": seed,
    "device": device,
    "save_path": save_path
}

tuner = tune.Tuner(tune.with_resources(trainable, {"cpu": n_envs, "gpu": n_gpus}), param_space=search_space,
                    run_config= RunConfig(name="SAC_tune", verbose=1))

results = tuner.fit()

ray.shutdown()