import subprocess
import argparse
import pickle
import pandas as pd
import os

def generate_dataset(device, env_name, policy_path, n_episodes):
    # Construct the command with arguments
    command = [
        'python', '../highway-env/create_dataset.py',
        '--device', device,
        '--env_name', env_name,
        '--policy_path', policy_path,
        '--n_episodes', str(n_episodes),
        '--filename', "tmp.pkl"
    ]

    # Print the command for debugging
    print(f'Running command: {" ".join(command)}')

    # Run the command
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        # Check for errors
        if result.returncode != 0:
            print("Error executing script:")
            print(result.stderr)
        else:
            print("Script executed successfully")
    except Exception as e:
        print(f"Exception occurred: {e}")


parser = argparse.ArgumentParser(description="Create episodes for the specified env")
parser.add_argument("--device", type=str, default='cpu', help="Device to use")
parser.add_argument("--env_name", type=str, help="Environment to use")
parser.add_argument("--policy_path", type=str, help="Path to trained policy")
parser.add_argument("--n_episodes", type=int, help="Number of episodes to perform")
parser.add_argument("--k", type=int, help="Number of the best episodes to keep")

args = parser.parse_args()
device = args.device
env_name = args.env_name
policy_path = args.policy_path
n_episodes = args.n_episodes
k = args.k

directory = "./transfer_dataset"
if not os.path.exists(directory):
    os.mkdir(directory)

# Generate dataset of episodes
generate_dataset(device, env_name, policy_path, n_episodes)

with open(f"../highway-env/dataset/{env_name}/tmp.pkl", "rb") as f:
    data = pickle.load(f)

df = pd.DataFrame(data)
# Drop all the failed episodes
df = df[df['Terminated'].apply(lambda x: any(x))]
# Compute sum of rewards
df['sum_float_list'] = df['Reward'].apply(lambda x: sum(x))
# Sort by sum of rewards 
df = df.sort_values(by='sum_float_list', ascending=False)

df = df.drop(columns=['sum_float_list'])

# Get the first 100 best episodes
df = df.head(k)

data = df.to_dict()

tokens = env_name.split('-')
if tokens[1] != "v0" and tokens[1] != "v1":
    name = f'{tokens[0]}-{tokens[1]}'
else:
    name = tokens[0]

with open(f"{directory}/{name}_best_{k}.pkl", "wb") as f:
    pickle.dump(data, f)