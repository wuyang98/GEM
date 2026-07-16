import os
import numpy as np
import torch
from tqdm import tqdm
from vae import VAE, load_points_as_images

def load_vae_model(model_path, device="cuda"):
    # Ensure model architecture matches training
    model = VAE(input_dim=1, output_dim=1)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

def encode_image(model, range_map, device="cuda"):
    with torch.no_grad():
        latent, _, _ = model.encoder(range_map)
    return latent

def load_kitti_points_as_images(bin_path):
    """
    Convert a KITTI bin file to a range map.
    """
    range_map = load_points_as_images(bin_path)
    return range_map

# Model path
model_path = "vae/checkpoints/vae_epoch_80.pth"
vae_model = load_vae_model(model_path, device="cuda")

# KITTI dataset path
kitti_root = "path/dataset/kitti_odometry/dataset"
output_root = "path/dataset_processed/kitti_odometry_vae_train"

# all
# sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

# # train
sequences = ["00", "01", "02", "03", "04", "05", "06", "07"]

# # test
# sequences = ["08", "09", "10"]

# Iterate over all sequences
for sequence in tqdm(sequences, desc="Processing Sequences"):
    print(f"\nProcessing sequence: {sequence}")
    
    # Create output directory
    seq_output_dir = os.path.join(output_root, "sequences", sequence)
    os.makedirs(seq_output_dir, exist_ok=True)
    
    # Point cloud file directory
    velo_dir = os.path.join(kitti_root, "sequences", sequence, "velodyne")
    
    if not os.path.exists(velo_dir):
        print(f"Velodyne directory not found: {velo_dir}")
        continue
    
    # Get all point cloud files
    velo_files = sorted([f for f in os.listdir(velo_dir) if f.endswith('.bin')])
    
    print(f"Processing {len(velo_files)} frames in sequence {sequence}")
    
    # Iterate over all point cloud files in this sequence
    for velo_file in tqdm(velo_files, desc=f"Sequence {sequence}", leave=False):
        # Build input file path
        bin_path = os.path.join(velo_dir, velo_file)
        
        # Build output file path
        output_filename = velo_file.replace('.bin', '.pt')
        save_path = os.path.join(seq_output_dir, output_filename)
        
        # Check if already processed
        if os.path.exists(save_path):
            continue
        
        try:
            # Load point cloud and convert to range map
            range_map = load_kitti_points_as_images(bin_path)
            
            # Convert to tensor
            range_map = torch.from_numpy(range_map).unsqueeze(0).to(device="cuda")
            # range_map = range_map[:, [0]]  # Select specific channel if needed
            
            # Encode
            latent = encode_image(vae_model, range_map, device="cuda")
            
            # Save
            torch.save(latent.cpu(), save_path)
            
        except Exception as e:
            print(f"Error processing {bin_path}: {e}")
            continue

print("All KITTI sequences processed successfully!")