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
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import gymnasium as gym

def evaluate(env, agent, n_validation_steps, n_timesteps_per_epoch, epoch):

    dist_list = []

    for video in tqdm(range(n_validation_steps)):
            done = truncated = False
            obs, info = env.reset()
            prev_pos = deepcopy(info["position"])
            dist = 0
            
            while not (done or truncated):
                action, _states = agent.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                
                # Compute distance
                curr_pos = info['position']
                if prev_pos is not None:
                    if info['speed'] >= 0:
                        dist += np.linalg.norm(curr_pos - prev_pos)
                    else:
                        dist -= np.linalg.norm(curr_pos - prev_pos)


            dist_list.append(dist)

    wandb.log({"mean distance": np.mean(np.array(dist_list)), "timestep_val": (n_timesteps_per_epoch*(epoch+1))})


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
        #For racetrack
        #task_description = "The ego-veichle is driving on a racetrack. The agent's objective is to follow the tracks while avoiding collisions with other vehicles."
        #For indiana
        task_description = "The ego-veichle is driving on a circuit. It is an oval circuit. The objective is to complete the track as fast as possible without touching the edges."
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

    wandb_run = wandb.init(
        project="SAC_indiana_transfer",
        config=wandb_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    save_folder = "grid_res_new_rew_complex_track"

    if not os.path.exists(f"./{save_folder}"):
        os.makedirs(save_folder)
    
    # Init env
    env = make_vec_env(env_name, n_envs=n_envs, env_kwargs={'config': config}, vec_env_cls=SubprocVecEnv)

    if device != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        device = torch.device(0)

    if env_name == "indiana-v0":
        flag_indiana = True
    else:
        flag_indiana = False

    if transfer_type != 0:
        ae = HighwayEnvModel.load_from_checkpoint(ae_path)

    if transfer_type == 0:
        # Without transfer learning
        agent = SACMaster(env=env, policy='MlpPolicy', learning_rate=5e-5, batch_size=8192, tau=0.1, gamma=0.65, gradient_steps=10, 
                            train_freq=15, use_sde=False, device=device, seed=seed, tensorboard_log="./tensorboard_log", indiana=flag_indiana)
    elif transfer_type == 1:
        # TL without task description
        agent = SACMaster(env=env, policy='MlpPolicy', learning_rate=5e-5, batch_size=8192, tau=0.1, gamma=0.65, gradient_steps=10, 
                            train_freq=15, use_sde=False, policy_dir=policy_dir, experience_dir=experience_dir, 
                            descriptions_dir=description_dir, ae_model=ae, k=k, similarity_thr=similarity_thr, device=device, seed=seed, 
                            tensorboard_log="./tensorboard_log", indiana=flag_indiana)
    elif transfer_type == 2:
        # TL with task description
        agent = SACMaster(env=env, policy='MlpPolicy', learning_rate=5e-5, batch_size=8192, tau=0.1, gamma=0.65, gradient_steps=10, 
                            train_freq=15, use_sde=False, policy_dir=policy_dir, experience_dir=experience_dir, 
                            descriptions_dir=description_dir, ae_model=ae, k=k, similarity_thr=similarity_thr, task_description=task_description,
                            tokenizer_str="bert-base-cased", device=device, seed=seed, tensorboard_log="./tensorboard_log", indiana=flag_indiana)

    epochs = 10
    n_timesteps_per_epoch = int(total_timesteps/10)
    n_validation_steps = 10

    for epoch in range(epochs):

        agent.learn(total_timesteps=n_timesteps_per_epoch, 
                    progress_bar=True,
                    log_interval=10,
                    reset_num_timesteps=False,
                    callback=WandbCallback(
                        gradient_save_freq=100,
                        model_save_path=f"racetrack_models_new_rew/{wandb_run.id}",
                        verbose=2,
                    ))
        
        # Validation step
        test_env = gym.make(env_name, config = config, render_mode = "rgb_array")
        evaluate(test_env, agent, n_validation_steps, n_timesteps_per_epoch, epoch)
        test_env.close()
    
    wandb_run.finish()
    
    if transfer_type > 0:
        agent.save(f"{save_folder}/{env_name}_{transfer_type}_{n_envs}env_{k}_{similarity_thr}")
    else:
        agent.save(f"{save_folder}/{env_name}_{transfer_type}_{n_envs}env")

    

if __name__ == "__main__":

    main()
