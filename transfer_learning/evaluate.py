from copy import deepcopy
from tqdm import tqdm
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
import pickle
import os
import argparse
import sys
sys.path.insert(1, "..")
from stable_baselines3.sac.sac import SACMaster


def init_variables(env_name):

    if env_name == "indiana-v0":
        next_lane_dict={
            ('a', 'b'): ('b', 'c'),
            ('b', 'c'): ('c', 'd'),
            ('c', 'd'): ('d', 'a'),
            ('d', 'a'): ('a', 'b')
        }

        section_passed_start_from = {
            ('a', 'b'): 0,
            ('b', 'c'): 0,
            ('c', 'd'): 0,
            ('d', 'a'): 0
        }

        section_activated_skill = {
            ('a', 'b'): np.zeros(4),
            ('b', 'c'): np.zeros(4),
            ('c', 'd'): np.zeros(4),
            ('d', 'a'): np.zeros(4)
        }

        section_staying_counter = {
            ('a', 'b'): 0,
            ('b', 'c'): 0,
            ('c', 'd'): 0,
            ('d', 'a'): 0
        }

    elif env_name == "racetrack-v0":
        next_lane_dict={
            ('a', 'b'): ('b', 'c'),
            ('b', 'c'): ('c', 'd'),
            ('c', 'd'): ('d', 'e'),
            ('d', 'e'): ('e', 'f'),
            ('e', 'f'): ('f', 'g'),
            ('f', 'g'): ('g', 'h'),
            ('g', 'h'): ('h', 'i'),
            ('h', 'i'): ('i', 'a'),
            ('i', 'a'): ('a', 'b')
        }

        section_passed_start_from = {
            ('a', 'b'): 0,
            ('b', 'c'): 0,
            ('c', 'd'): 0,
            ('d', 'e'): 0,
            ('e', 'f'): 0,
            ('f', 'g'): 0,
            ('g', 'h'): 0,
            ('h', 'i'): 0,
            ('i', 'a'): 0
        }

        section_activated_skill = {
            ('a', 'b'): np.zeros(4),
            ('b', 'c'): np.zeros(4),
            ('c', 'd'): np.zeros(4),
            ('d', 'e'): np.zeros(4),
            ('e', 'f'): np.zeros(4),
            ('f', 'g'): np.zeros(4),
            ('g', 'h'): np.zeros(4),
            ('h', 'i'): np.zeros(4),
            ('i', 'a'): np.zeros(4),
        }

        section_staying_counter = {
            ('a', 'b'): 0,
            ('b', 'c'): 0,
            ('c', 'd'): 0,
            ('d', 'e'): 0,
            ('e', 'f'): 0,
            ('f', 'g'): 0,
            ('g', 'h'): 0,
            ('h', 'i'): 0,
            ('i', 'a'): 0,
        }

    elif env_name == "racetrack-complex-v0":
        next_lane_dict={
            ('a', 'b'): ('b', 'c'),
            ('b', 'c'): ('c', 'd'),
            ('c', 'd'): ('d', 'e'),
            ('d', 'e'): ('e', 'f'),
            ('e', 'f'): ('f', 'g'),
            ('f', 'g'): ('g', 'h'),
            ('g', 'h'): ('h', 'i'),
            ('h', 'i'): ('i', 'j'),
            ('i', 'j'): ('j', 'k'),
            ('j', 'k'): ('k', 'l'),
            ('k', 'l'): ('a', 'b')
        }

        section_passed_start_from = {
            ('a', 'b'): 0,
            ('b', 'c'): 0,
            ('c', 'd'): 0,
            ('d', 'e'): 0,
            ('e', 'f'): 0,
            ('f', 'g'): 0,
            ('g', 'h'): 0,
            ('h', 'i'): 0,
            ('i', 'j'): 0,
            ('j', 'k'): 0,
            ('k', 'l'): 0
        }

        section_activated_skill = {
            ('a', 'b'): np.zeros(4),
            ('b', 'c'): np.zeros(4),
            ('c', 'd'): np.zeros(4),
            ('d', 'e'): np.zeros(4),
            ('e', 'f'): np.zeros(4),
            ('f', 'g'): np.zeros(4),
            ('g', 'h'): np.zeros(4),
            ('h', 'i'): np.zeros(4),
            ('i', 'j'): np.zeros(4),
            ('j', 'k'): np.zeros(4),
            ('k', 'l'): np.zeros(4),
        }

        section_staying_counter = {
            ('a', 'b'): 0,
            ('b', 'c'): 0,
            ('c', 'd'): 0,
            ('d', 'e'): 0,
            ('e', 'f'): 0,
            ('f', 'g'): 0,
            ('g', 'h'): 0,
            ('h', 'i'): 0,
            ('i', 'j'): 0,
            ('j', 'k'): 0,
            ('k', 'l'): 0,
        }

    else:
        raise ValueError(f"The environment {env_name} doesn't exists")

    return next_lane_dict, section_passed_start_from, section_activated_skill, section_staying_counter

def eval_model(env_name, agent, n_episodes, video_save_dir = None):

    next_lane_dict, section_passed_start_from, section_activated_skill, section_staying_counter = init_variables(env_name)

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

    env = gym.make(env_name, config = config, render_mode="rgb_array")

    if video_save_dir is not None:
        #env = RecordVideo(env, video_save_dir, episode_trigger=lambda episode_id: episode_id % 10 == 0)
        env = RecordVideo(env, video_save_dir, episode_trigger=lambda episode_id: True)

    all_passed_sections = [] # how many sections are passed
    all_mean_dist_per_action = [] # the distance performed per action (avg)
    per_sector_actions = [] # how many actions were performed in a single section
    section_passed = [0] * 10 # distribution of passed sections
    dist_list = [] # save the distance performed in each episode
    reward_list = [] # save the total reward for each episode
    lap_closed_list = [] # keep track of the number of completed laps
    speed_list = []

    for episode in tqdm(range(n_episodes)):
        done = truncated = False
        obs, info = env.reset(seed=episode)
        prev_pos = deepcopy(info["position"])
        road_net = info["road"]
        distance = 0
        episode_length = 0
        n_actions = 0
        passed_sections = 0
        ep_reward = 0
        speed = 0
        steps_count = 0
        lane_edge_dict = {}

        for key, value in road_net.lanes_dict().items():
            lane_edge_dict[value] = (key[0], key[1])

        current_lane = lane_edge_dict[info['current_lane']]
        next_lane = next_lane_dict[current_lane]

        spawning_lane = current_lane

        while not (done or truncated):
            action, _ = agent.predict(obs)
            section_staying_counter[current_lane] += 1
            obs, reward, done, truncated, info = env.step(action)

            ep_reward += reward

            speed += info['speed']
            steps_count += 1

            # Compute distance
            curr_pos = info['position']
            if prev_pos is not None:
                if info['speed'] >= 0:
                    distance += np.linalg.norm(curr_pos - prev_pos)
                else:
                    distance -= np.linalg.norm(curr_pos - prev_pos)

            episode_length += 1
            n_actions += 1
            current_lane = lane_edge_dict[info['current_lane']]
            if current_lane == next_lane:
                next_lane = next_lane_dict[current_lane]
                passed_sections += 1
                per_sector_actions.append(n_actions)
                n_actions = 0
            
            prev_pos = deepcopy(curr_pos)

        n_sections = len(next_lane_dict.keys())
        lap_closed_list.append(passed_sections/n_sections)
        dist_list.append(distance)
        reward_list.append(ep_reward)
        speed_list.append(speed/steps_count)

        all_passed_sections.append(passed_sections)  
        all_mean_dist_per_action.append(distance/episode_length)          
        section_passed_start_from[spawning_lane] += passed_sections

        for k, v in section_activated_skill.items():
            v /= section_staying_counter[k]

        if passed_sections < 10:
            section_passed[passed_sections] += 1
        else:
            section_passed[-1] += 1

    env.close()

    res_dict = {
        "all_mean_dist_per_action" : all_mean_dist_per_action,
        "section_staying_counter" : section_staying_counter,
        "section_activated_skill" : section_activated_skill,
        "section_passed_start_from" : section_passed_start_from,
        "per_sector_actions" : per_sector_actions, 
        "all_passed_sections" : all_passed_sections,
        "section_passed" : section_passed,
        "dist_list" : dist_list,
        "reward_list" : reward_list,
        "lap_closed_list" : lap_closed_list,
        "speed_list" : speed_list
    }
    
    return res_dict

parser = argparse.ArgumentParser(description="Evaluate RL models")

# Define command-line arguments
parser.add_argument("--models_path", type=str, help="Path to the trained models")
parser.add_argument("--save_path", type=str, help="Path where to save the metrics")
parser.add_argument("--n_episodes", type=int, help="Number of episodes to perform")

# Parse the command-line arguments
args = parser.parse_args()

models_path = args.models_path
save_path = args.save_path
n_episodes = args.n_episodes

dir = os.listdir(models_path)
res_list = []

for file in dir:

    print(f"Evaluating model: {file}")

    video_save_path = save_path+"/"+file

    if not os.path.exists(video_save_path):
        os.mkdir(video_save_path)

    tokens = file.split("_")
    env_name = tokens[0]
    agent = SACMaster.load(f"{models_path}/{file}")

    res_dict = eval_model(env_name, agent, n_episodes, video_save_dir=video_save_path)

    res_dict['model'] = file

    res_list.append(res_dict)

    # Plotting all passed sections histogram
    plt.hist(res_dict["all_passed_sections"], bins=max(res_dict["all_passed_sections"]))

    # Adding labels and title
    plt.xlabel('Number of sections passed in an episode')
    plt.ylabel('Frequency')
    plt.title('Distribution of the number of sections passed in an episode')
    
    # Display the plot
    plt.savefig(f'{video_save_path}/all_passed_sections.png')

    plt.clf()
    
    # Plotting a basic histogram
    plt.hist(res_dict["section_passed"], bins=max(res_dict["section_passed"]))

    # Adding labels and title
    plt.xlabel('Number of sections passed in an episode')
    plt.ylabel('Frequency')
    plt.title('Distribution of the number of sections passed in an episode')
    
    # Display the plot
    plt.savefig(f'{video_save_path}/section_passed.png')

    plt.clf()

    # Plotting a basic histogram
    plt.hist(res_dict["dist_list"], bins=10)

    # Adding labels and title
    plt.xlabel('Distance done by the ego veichle')
    plt.ylabel('Frequency')
    plt.title('Distribution of distance done by the ego veichle in an episode')
    
    # Display the plot
    plt.savefig(f'{video_save_path}/distance_list.png')

    plt.clf()
    
if os.path.exists(f'{save_path}/res.pkl'):
    # Load existing data
    with open(f'{save_path}/res.pkl', 'rb') as file:
        existing_data = pickle.load(file)
else:
    # If the file does not exist, initialize an empty list
    existing_data = []

# Append the new data
existing_data.extend(res_list)

# Save the updated data back to the pickle file
with open(f'{save_path}/res.pkl', 'wb') as file:
    pickle.dump(existing_data, file)