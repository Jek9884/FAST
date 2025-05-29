from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.type_aliases import Schedule
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
from gymnasium import spaces
import torch
import pickle
import argparse
import os
import cv2
import time
import numpy as np

def read_video_frames(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)
    frames = []

    # Check if the video file is opened successfully
    if not video.isOpened():
        print("Error: Unable to open video file")
        return None

    # Read frames until the video is finished
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # Convert frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)

    # Release the video capture object
    video.release()
    return frames[-4:]

def delete_file(file_path):
    try:
        # Check if the file exists
        if os.path.exists(file_path):
            # Delete the file
            os.remove(file_path)
            print(f"{file_path} has been successfully deleted.")
        else:
            print(f"File {file_path} does not exist.")
    except OSError as e:
        print(f"Error: {file_path} - {e}")

parser = argparse.ArgumentParser(description="Create episodes for the specified env")

# Define command-line arguments
parser.add_argument("--device", type=str, default='cpu', help="Device to use")
parser.add_argument("--env_name", type=str, help="Environment to use")
parser.add_argument("--policy_path", type=str, help="Path to trained policy")
parser.add_argument("--n_episodes", type=int, help="Number of episodes to perform")
parser.add_argument("--filename", type=str, help="Filename for saving episodes")

args = parser.parse_args()
device = args.device
env_name = args.env_name
policy_path = args.policy_path
n_episodes = args.n_episodes
filename = args.filename

if device != 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = device

device = torch.device(0)

directory = "./dataset"
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Directory '{directory}' created successfully")

directory = f'{directory}/{env_name}' 
if not os.path.exists(directory):
    os.makedirs(directory)
    print(f"Directory '{directory}' created successfully")

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

env = gym.make(env_name, config=config, render_mode="rgb_array")
env = RecordVideo(env, directory, lambda x: True, disable_logger = True)

scheduler = get_schedule_fn(0.1)

obs_space = env.observation_space

if env_name == "lane-centering-v0":
    obs_space = spaces.Box(-np.inf, np.inf, (1,9), dtype = np.float32)
    

policy = SACPolicy(obs_space, env.action_space, scheduler)
policy.load_state_dict(torch.load(policy_path, map_location=torch.device(device)))

model = SAC('MlpPolicy', env=env, use_sde=False, device=device)
model.policy = policy

episodes_list = {
    'Observation' : [],
    'Action' : [],
    'Reward' : [],
    'Terminated' : [],
    'Truncated' : []
}

start_time = time.time()

i = 0

while i < n_episodes:

    obs, _ = env.reset()

    episode = {
        'Action' : [],
        'Reward' : [],
        'Terminated' : [],
        'Truncated' : []
    }

    while True:
        if env_name == "lane-centering-v0":
            obs = np.expand_dims(obs[0], axis=0)

        action = model.predict(obs)
        episode['Action'].append(action[0])
        obs, reward, terminated, truncated, info = env.step(action[0])
        episode['Reward'].append(reward)
        episode['Terminated'].append(terminated)
        episode['Truncated'].append(truncated)

        if terminated or truncated:
            break
    
    # Get last four frames from the created video
    videoname = f'{directory}/rl-video-episode-{i}'

    if np.array(read_video_frames(f'{videoname}.mp4')).shape[0] < 4:
        i = i + 1
        n_episodes = n_episodes + 1
        # Delete video files
        delete_file(f'{videoname}.mp4')
        delete_file(f'{videoname}.meta.json')
        continue
    
    episodes_list['Observation'].append(read_video_frames(f'{videoname}.mp4'))

    # Delete video files
    delete_file(f'{videoname}.mp4')
    delete_file(f'{videoname}.meta.json')

    # Append data
    episodes_list['Action'].append(episode['Action'])
    episodes_list['Reward'].append(episode['Reward'])
    episodes_list['Terminated'].append(episode['Terminated'])
    episodes_list['Truncated'].append(episode['Truncated'])

    i = i + 1

with open(f'{directory}/{filename}', 'wb') as f:
    pickle.dump(episodes_list, f)