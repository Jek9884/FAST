import gzip
import numpy as np
import torch
import os
from atari_model import ImageAutoencoder
from image_kfold_dataset import ImageKfoldDataset
from torch.utils.data import DataLoader

def get_episodes_idx(dir_path, seed, ckpt):

    episodes_idx = []

    try:
        filename_term = f"{dir_path}/{seed}/replay_logs/$store$_terminal_ckpt.{ckpt}.gz"
        pickled_data = gzip.GzipFile(filename=filename_term)
        np_term = np.load(pickled_data)

        for idx, elem in enumerate(np_term):
            if elem == 1:
                episodes_idx.append(idx)

        return episodes_idx
    
    except FileNotFoundError:
        print(f"File {filename_term} not found.")
        return episodes_idx
    except PermissionError:
        print(f"Permission error: Unable to read {filename_term}.")
        return episodes_idx
    except Exception as e:
        print(f"An error occurred: {e}")
        return episodes_idx
    
def normalize_images_tensor(input_tensor):
    return input_tensor / 255.0

def scale_images_tensor(input_tensor, min_scale=0.0, max_scale=1.0):
    # Reshape the tensor to [n_samples * n_channels, height, width]
    reshaped_tensor = input_tensor.view(-1, input_tensor.size(2), input_tensor.size(3))
    
    # Scale the tensor to the desired range
    scaled_tensor = min_scale + (max_scale - min_scale) * reshaped_tensor
    
    # Reshape the scaled tensor back to the original shape
    scaled_tensor = scaled_tensor.view(input_tensor.size())
    
    return scaled_tensor

def log_scale_images_tensor(input_tensor, epsilon=1e-5):
    # Reshape the tensor to [n_samples * n_channels, height, width]
    reshaped_tensor = input_tensor.view(-1, input_tensor.size(2), input_tensor.size(3))

    # Add a small epsilon to avoid taking the log of zero
    reshaped_tensor = reshaped_tensor + epsilon
    
    # Apply log scaling to the tensor
    log_scaled_tensor = torch.log(reshaped_tensor)
    
    # Reshape the log-scaled tensor back to the original shape
    log_scaled_tensor = log_scaled_tensor.view(input_tensor.size())
    
    return log_scaled_tensor

def extract_embeddings(images_tensor, dev, model_path, normalize, log_scale, num_workers=20):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = dev
    device = torch.device(0)

    if normalize:
        images_tensor = normalize_images_tensor(images_tensor)
    if log_scale:
        images_tensor = log_scale_images_tensor(images_tensor)

    model = ImageAutoencoder.load_from_checkpoint(model_path)
    model.to(device)

    images_dataset = ImageKfoldDataset(images_tensor)
    images_dataloader = DataLoader(images_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    embeddings_list = []
    model.eval()

    with torch.no_grad():

        for data in images_dataloader:
            images = data
            images = images.to(device)

            #Forward pass
            embeddings = model(images, return_encodings=True)

            embeddings_list.append(embeddings.cpu())

    # Cast list in tensor and save it
    embeddings_torch = torch.cat(embeddings_list, dim=0)
    
    return embeddings_torch