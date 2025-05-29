import pickle
import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import multiprocessing
import sys
sys.path.insert(1, '../')
from utils import apply_upsampling

def find_pkls_in_subfolders(directory):
    pkl_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pkl'):
                pkl_files.append(os.path.join(root, file))
    return pkl_files

def process_episode(images):

    upsampled_images = []

    for i in range(images.shape[0]):
        upsampled_images.append(apply_upsampling(images[i]))

    return np.array(upsampled_images)

def extract_images_from_pikle(file):

    with open(file, 'rb') as f:
        data = pickle.load(f)
        obs = data['Observation']
        obs = np.array(obs)
        if n_sample < 1:
            n_obs = int(obs.shape[0] * n_sample)
            print(f'Env name: {env_name} - Num sample {n_obs}')
            obs = obs[:n_obs, :, :, :]
        result = obs

        # Make upsampling only for observations that are smaller than 600x600
        if obs[0][0].shape[0] != 600 or obs[0][0].shape[1] != 600:

            num_processes = multiprocessing.cpu_count()

            # Create a pool of workers
            pool = multiprocessing.Pool(processes=num_processes)

            # Map the function to the collection of items using multiple processes
            result = pool.map(process_episode, obs)

            # Close the pool to release resources
            pool.close()
            pool.join()

            result = np.array(result)

        return result


parser = argparse.ArgumentParser(description="Extract images from the full dataset to train the Autoencoder", allow_abbrev=False)

# Define command-line arguments
parser.add_argument("--dataset_path", type=str, help="Path of the dataset")
parser.add_argument("--save_path", type=str, help="Path where to save the extracted data")
parser.add_argument("--n_sample", type=float, default=1, help="Percentage of sample to consider for each task")
parser.add_argument("--env_name", type=str, help="Name of the environment to consider, if 'all' consider all the environments")

args = parser.parse_args()
dataset_path = args.dataset_path
save_path = args.save_path
n_sample = args.n_sample
name = args.env_name

if n_sample <= 0 or n_sample > 1:
    raise ValueError("Percentage out of range (0,1]")

if not os.path.exists(dataset_path):
    raise ValueError(f"File does not exist")

tensor_list = None

pkl_files = find_pkls_in_subfolders(dataset_path)

for file in pkl_files:

    split_string = file.split('/')
    env_name = split_string[2]

    # Exclude u-turn for observations dimension issues
    if env_name != "u-turn-v0":

        result = extract_images_from_pikle(file)
                
        if tensor_list is None:
            tensor_list = result
        else:
            tensor_list = np.concatenate((tensor_list, result))

tensor = torch.tensor(tensor_list)

# Save tensor to a file
torch.save(tensor, f'{save_path}/{name}_final_res_{n_sample}.pt')
    





                    

            

            
            

            
