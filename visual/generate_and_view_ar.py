import argparse
from pathlib import Path
import einops
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
from basic_module import CustomPreDataset
import utils.inference
import utils.render
import datetime
import numpy as np
from vae import VAE
from torch.utils.data import DataLoader
import os
from itertools import islice
import glob

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
    mask = (metric >= 1.45) & (metric <= 80)
    ray_angles = get_hdl64e_linear_ray_angles(64, 1024)
    phi = ray_angles[:, [0]].to(device=metric.device)
    theta = ray_angles[:, [1]].to(device=metric.device)
    grid_x = metric * phi.cos() * theta.cos()
    grid_y = metric * phi.cos() * theta.sin()
    grid_z = metric * phi.sin()
    xyz = torch.cat((grid_x, grid_y, grid_z), dim=1)
    xyz = xyz * mask.float()
    return xyz

def load_vae_model(model_path, device="cuda"):
    # Ensure model architecture matches training
    model = VAE(input_dim=1, output_dim=1)
    # model = nn.DataParallel(model) 
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    # model = model.module
    model = model.to(device)
    return model

def load_sorted_data(data_dir, device="cuda"):
    """
    Load data and pose files in sorted order.
    """
    data_dir = Path(data_dir)

    all_pt_files = sorted(list(data_dir.glob('*.pt')))

    # Separate point cloud data files from pose files
    data_files = []
    pose_files = []

    for pt_file in all_pt_files:
        if '_egopose.pt' in pt_file.name:
            pose_files.append(pt_file)
        else:
            data_files.append(pt_file)

    # Load the first 5 point cloud data files
    x_past_list = []
    name_past = []
    for i in range(min(5, len(data_files))):
        data_file = data_files[i]
        data_tensor = torch.load(str(data_file), map_location=device)
        # Ensure data dimensions are [1, 64, 1024] or [64, 1024], then expand to [1, 1, 64, 1024]
        if data_tensor.dim() == 2:  # [64, 1024]
            data_tensor = data_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 64, 1024]
        elif data_tensor.dim() == 3:  # [C, 64, 1024]
            data_tensor = data_tensor.unsqueeze(0)  # [1, C, 64, 1024]
        elif data_tensor.dim() == 4 and data_tensor.shape[0] != 1:
            data_tensor = data_tensor[:1]  # [1, C, 64, 1024]

        x_past_list.append(data_tensor)
        name_past.append(data_file.stem)

    # Stack point cloud data
    x_past = torch.cat(x_past_list, dim=0)  # [5, C, 64, 1024]
    x_past = x_past.unsqueeze(0)  # [1, 5, C, 64, 1024]

    # Load pose files and split into past and future in sorted order
    egopose_past_list = []
    egopose_future_list = []
    name_future = []

    for i in range(min(10, len(pose_files))):
        pose_file = pose_files[i]
        pose_matrix = torch.load(str(pose_file), map_location=device)

        # Extract [:3, 3] — the last column of the first three rows (xyz coordinates)
        extracted_xyz = pose_matrix[:3, 3]  # [3]

        if i < 5:  # first 5 as past
            egopose_past_list.append(extracted_xyz)
        else:  # remaining as future
            egopose_future_list.append(extracted_xyz)
            name_future.append(pose_file.stem)

    # Stack pose data
    egopose_past = torch.stack(egopose_past_list).unsqueeze(0)  # [1, 5, 3]
    egopose_future = torch.stack(egopose_future_list).unsqueeze(0)  # [1, 5, 3]
    
    return x_past, egopose_past, egopose_future, name_past, name_future

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    # =================================================================================
    # Load pre-trained model
    # =================================================================================
    model_path = "vae/checkpoints/vae_epoch_80.pth"
    vae = load_vae_model(model_path, device=args.device)
    vae.requires_grad_(False)
    vae.eval()

    ddpm, lidar_utils, _ = utils.inference.setup_model(args.ckpt, device=args.device)

    # =================================================================================
    # Load sorted data
    # =================================================================================
    x_past, egopose_past, egopose_future, name_past, name_future = load_sorted_data(
        args.input_dir, device=args.device
    )
    
    print(f"Data shapes - x_past: {x_past.shape}, egopose_past: {egopose_past.shape}, egopose_future: {egopose_future.shape}")
    
    # Ensure data types and dimensions are correct
    x_past = x_past.to(args.device)
    egopose_past = egopose_past.to(args.device)
    egopose_future = egopose_future.to(args.device)
    
    # =================================================================================
    # Sampling (reverse diffusion)
    # =================================================================================
    xs = ddpm.sample(
        batch_size=1,
        num_steps=args.sampling_steps,
        x_past=x_past,
        egopose_past=egopose_past,
        egopose_future=egopose_future,
        frames=args.frames,
        rng=torch.Generator(device=args.device).manual_seed(0),
    ).clamp(-1, 1)

    with torch.no_grad():
        sample = xs[0]  # [5, C, H, W]
        recon = vae.decoder(sample)  # predicted future frames
        past_gt = vae.decoder(x_past.squeeze(dim=0))  # ground-truth past frames
        torch.cuda.empty_cache()
    for i in range(sample.shape[0]):  # sample shape: [5, C, H, W]
        single_sample = sample[i]  # [C, H, W]
        sample_path = Path('./ar21') / f"sample_{i:03d}.pt"
        torch.save(single_sample.cpu(), sample_path)

    def render_pred(x, name_future, sample_number):
        depth = x
        if depth.numel() > 0:
            metric = depth
            img = einops.rearrange(metric, "B C H W -> B 1 (C H) W")
            img = utils.render.colorize(img / 80) / 255
            mask = (metric > 1.45) & (metric < 80.0)
            xyz = to_xyz(metric) * mask

            xyz = xyz / 80

            z_min, z_max = -2 / 80, 0.5 / 80
            z = (xyz[:, [2]] - z_min) / (z_max - z_min)
            colors = utils.render.colorize(z.clamp(0, 1), cm.GnBu_r) / 255
            R, t = utils.render.make_Rt(pitch=torch.pi / 3, yaw=torch.pi / 4, z=0.57)
            bev = 1 - utils.render.render_point_clouds(
                points=einops.rearrange(xyz, "B C H W -> B (H W) C"),
                colors= 1 - einops.rearrange(colors, "B C H W -> B (H W) C"),
                # t=torch.tensor([0, 0, 1.0]).to(xyz),
                R=R.to(xyz),
                t=t.to(xyz),
        )
        return bev, img


    bev, img = render_pred(recon, name_future, 0)
    name_bev = "pred_future.png"
    save_image(bev, name_bev, nrow=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default='path/diffusion_0000120000.pth')
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--input_dir", type=str, default="./ar20", help="Directory containing input .pt files")
    parser.add_argument("--output_dir", type=str, default="./ar20", help="Directory to save output .pt files")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--sampling_steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=88) # 88, 77
    args = parser.parse_args()
    args.device = torch.device(args.device)
    main(args)