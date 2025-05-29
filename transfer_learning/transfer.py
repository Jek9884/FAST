import torch
import os
import sys
sys.path.insert(1, "../")
from stable_baselines3.sac.sac import SACMaster
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from ae.imageAE.highway_model import HighwayEnvModel
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback

def main():

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

    parser = argparse.ArgumentParser(description="Train a RL agent using TL")

    # Define command-line arguments
    parser.add_argument("--env_name", type=str, help="Environment to use")
    parser.add_argument("--policy_dir", type=str, help="Path to the source policies")
    parser.add_argument("--experience_dir", type=str, help="Path to the source dataset")
    parser.add_argument("--description_dir", type=str, help="Path to the source tasks descriptions")
    parser.add_argument("--ae_path", type=str, help="Path to the ae model")
    parser.add_argument("--device", type=str, help="Device where to execute")
    parser.add_argument("--transfer_type", type=int, help="0 for no transfer, 1 for transfer, 2 for transfer with description")
    parser.add_argument("--total_timesteps", type=int, help="Max number of timesteps to perform")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of environments to use in parallel")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for replicability")
    parser.add_argument("--k", type=int, default=None, help="Number of time-steps to perform before to use the metric")
    parser.add_argument("--similarity_thr", default=None, type=float, help="Threshold for the metric")

    # Parse the command-line arguments
    args = parser.parse_args()

    assert args.transfer_type == 0 or args.transfer_type == 1 or args.transfer_type == 2
    assert args.n_envs > 0
    if args.transfer_type == 1 or args.transfer_type == 2:
        assert args.similarity_thr >= -1 and args.similarity_thr <= 1
        assert args.k > 0

    env_name = args.env_name#"racetrack-v0"
    policy_dir = args.policy_dir
    experience_dir = args.experience_dir
    description_dir = args.description_dir
    ae_path = args.ae_path
    device = args.device
    transfer_type = args.transfer_type
    total_timesteps = args.total_timesteps
    n_envs = args.n_envs
    seed = args.seed
    k = args.k
    similarity_thr = args.similarity_thr

    if transfer_type == 2:
        task_description = "The ego-veichle is driving on a racetrack. The agent's objective is to follow the tracks while avoiding collisions with other vehicles."
    else:
        task_description = None

    wandb_config = {
        "transfer_type": transfer_type,
        "total_timesteps": total_timesteps,
        "env_name": env_name,
        "phase": "training",
        "k": k,
        "simil_thr": similarity_thr,
        "task_description" : task_description
    }

    run = wandb.init(
        project="transfer_learning",
        config=wandb_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    # Init env
    env = make_vec_env(env_name, n_envs=n_envs, env_kwargs={'config': config}, vec_env_cls=SubprocVecEnv)

    if device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        device = torch.device(0)

    if transfer_type != 0:
        ae = HighwayEnvModel.load_from_checkpoint(ae_path)

    if transfer_type == 0:
        # Without transfer learning
        agent = SACMaster(env=env, policy='MlpPolicy', learning_rate=1e-4, batch_size=1024, tau=0.9, gamma=0.99, gradient_steps=10, 
                            train_freq=15, use_sde=False, device=device, seed=seed, tensorboard_log="./tensorboard_log")
    elif transfer_type == 1:
        # TL without task description
        agent = SACMaster(env=env, policy='MlpPolicy', learning_rate=1e-4, batch_size=1024, tau=0.9, gamma=0.99, gradient_steps=10, 
                            train_freq=15, use_sde=False, policy_dir=policy_dir, experience_dir=experience_dir, 
                            descriptions_dir=description_dir, ae_model=ae, k=k, similarity_thr=similarity_thr, device=device, seed=seed, 
                            tensorboard_log="./tensorboard_log")
    elif transfer_type == 2:
        # TL with task description
        agent = SACMaster(env=env, policy='MlpPolicy', learning_rate=1e-4, batch_size=1024, tau=0.9, gamma=0.99, gradient_steps=10, 
                            train_freq=15, use_sde=False, policy_dir=policy_dir, experience_dir=experience_dir, 
                            descriptions_dir=description_dir, ae_model=ae, k=k, similarity_thr=similarity_thr, task_description=task_description,
                            tokenizer_str="bert-base-cased", device=device, seed=seed, tensorboard_log="./tensorboard_log")

    agent.learn(total_timesteps=total_timesteps, progress_bar=True,
                callback=WandbCallback(
                    gradient_save_freq=100,
                    model_save_path=f"models/{run.id}",
                    verbose=2,
                ))

    if transfer_type > 0:
        agent.save(f"{env_name}_{transfer_type}_{n_envs}env_{k}_{similarity_thr}")
    else:
        agent.save(f"{env_name}_{transfer_type}_{n_envs}env")

if __name__ == "__main__":

    main()
