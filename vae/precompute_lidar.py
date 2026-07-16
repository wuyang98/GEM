import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from vae_v17_kitti import load_points_as_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def render(x):
    range_mask = (x[:, [0]] >= 1.45) & (x[:, [0]] < 80.0)
    xyz = to_xyz(x[:, [0]]) * range_mask

    # save points
    points = xyz
    a = points[0].reshape(3, -1)
    a = a.transpose(1, 0)
    point1 = a.float().reshape((65536 * 3),)

    return point1

def get_hdl64e_linear_ray_angles(H: int = 64, W: int = 1024):
    h_up, h_down = 3, -25
    w_left, w_right = 180, -180
    elevation = 1 - torch.arange(H) / H  # [0, 1]
    elevation = elevation * (h_up - h_down) + h_down  # [-25, 3]
    azimuth = 1 - torch.arange(W) / W  # [0, 1]
    azimuth = azimuth * (w_left - w_right) + w_right  # [-180, 180]
    [elevation, azimuth] = torch.meshgrid([elevation, azimuth], indexing="ij")
    angles = torch.stack([elevation, azimuth])[None].deg2rad()
    return angles

def to_xyz(metric: torch.Tensor) -> torch.Tensor:
    assert metric.dim() == 4
    mask = (metric >= 1.45) & (metric <= 80.0)
    ray_angles = get_hdl64e_linear_ray_angles(64, 1024)
    phi = ray_angles[:, [0]].to(device=metric.device)
    theta = ray_angles[:, [1]].to(device=metric.device)
    grid_x = metric * phi.cos() * theta.cos()
    grid_y = metric * phi.cos() * theta.sin()
    grid_z = metric * phi.sin()
    xyz = torch.cat((grid_x, grid_y, grid_z), dim=1)
    xyz = xyz * mask.float()
    return xyz

def load_kitti_points_as_images(bin_path):
    range_map = load_points_as_images(bin_path)
    return range_map

kitti_root = "path/dataset/kitti_odometry/dataset"
output_root = "path/dataset_processed/kitti_odometry_gt_test"

SPLIT_SEQUENCES = {
    "TRAIN": ["00", "01", "02", "03", "04", "05", "06", "07"],
    "TEST": ["08", "09", "10"]
}

target_split = "TEST"
sequences = SPLIT_SEQUENCES[target_split]

print(f"Processing KITTI {target_split} split: {sequences}")

for sequence in tqdm(sequences, desc="Processing Sequences"):
    print(f"\nProcessing sequence: {sequence}")

    seq_output_dir = os.path.join(output_root, "sequences", sequence)
    os.makedirs(seq_output_dir, exist_ok=True)

    velo_dir = os.path.join(kitti_root, "sequences", sequence, "velodyne")
    
    if not os.path.exists(velo_dir):
        print(f"Velodyne directory not found: {velo_dir}")
        continue
    
    velo_files = sorted([f for f in os.listdir(velo_dir) if f.endswith('.bin')])
    velo_files = velo_files[0:500]

    print(f"Processing {len(velo_files)} frames in sequence {sequence}")
    
    for velo_file in tqdm(velo_files, desc=f"Sequence {sequence}", leave=False):
        bin_path = os.path.join(velo_dir, velo_file)
        output_filename = velo_file.replace('.bin', '.bin')
        save_path = os.path.join(seq_output_dir, output_filename)

        if os.path.exists(save_path):
            continue

        try:
            gt_point_rangemap = load_kitti_points_as_images(bin_path)
            gt_point_rangemap = torch.from_numpy(gt_point_rangemap).unsqueeze(0).to(device=device)
            gt_point_tosave = render(gt_point_rangemap)

            with open(save_path, 'wb') as f:
                gt_point_tosave.cpu().numpy().tofile(f)
                
        except Exception as e:
            print(f"Error processing {bin_path}: {e}")
            continue

print("All KITTI sequences processed successfully!")