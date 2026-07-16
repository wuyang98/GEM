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

def preprocess(batch):
    x_frame, egopose, name = batch[0], batch[1], batch[2]
    x_past = x_frame[:, 0:5]
    x_future = x_frame[:, 5:10]
    egopose_past = egopose[:, 0:5]
    egopose_future = egopose[:, 5:10]
    name_past = name[0:5]
    name_future = name[5:10]

    return x_past, x_future, egopose_past, egopose_future, name_past, name_future

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    root_dirs=['path/dataset_processed/kitti_odometry_vae_train']
    dataset = CustomPreDataset(
        root_dirs=root_dirs,
        split="TEST",
        sequence_length=10,
        frame_step=6    
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )

    # =================================================================================
    # Load pre-trained model
    # =================================================================================
    model_path = "vae/checkpoints/vae_epoch_80.pth"
    vae = load_vae_model(model_path, device=args.device)
    vae.requires_grad_(False)
    vae.eval()

    ddpm, lidar_utils, _ = utils.inference.setup_model(args.ckpt, device=args.device)

    # =================================================================================
    # Sampling (reverse diffusion)
    # =================================================================================
    
    # --- Initialize sample_number ---
    sample_number = 0
    for batch in dataloader:
        x_past, x_future, egopose_past, egopose_future, name_past, name_future = preprocess(batch)
        x_past = x_past.to(args.device)
        x_future = x_future.to(args.device)
        egopose_past = egopose_past.to(args.device)
        egopose_future = egopose_future.to(args.device)

        xs = ddpm.sample(
            batch_size=args.batch_size,
            num_steps=args.sampling_steps,
            x_past = x_past,
            egopose_past = egopose_past,
            egopose_future = egopose_future,
            frames=args.frames,
            rng=torch.Generator(device=args.device).manual_seed(0),
        ).clamp(-1, 1)

        with torch.no_grad():
            sample = xs[0]
            recon = vae.decoder(sample)
            future_gt = vae.decoder(x_future.squeeze(dim=0))
            past_gt = vae.decoder(x_past.squeeze(dim=0))
            torch.cuda.empty_cache()

        def render_pred(x, name_future, sample_number):
            depth = x
            if depth.numel() > 0:
                metric = depth
                # img = einops.rearrange(metric, "B C H W -> B 1 (C H) W")
                # img = utils.render.colorize(img / 80) / 255
                mask = (metric > 1.45) & (metric < 80.0)
                xyz = to_xyz(metric) * mask

                x_axi = xyz[:, [0]]
                y_axi = xyz[:, [1]]
                z_axi = xyz[:, [2]]
                roof_mask = (x_axi > -2.0) & (x_axi < 2.0) & (y_axi > -1.8) & (y_axi < 1.8) & (z_axi > -2.0) & (z_axi < 15.0)
                xyz = xyz * (~roof_mask)

                xyz = xyz / 80

                z_min, z_max = -2 / 80, 0.5 / 80
                z = (xyz[:, [2]] - z_min) / (z_max - z_min)
                colors = utils.render.colorize(z.clamp(0, 1), cm.GnBu_r) / 255
                R, t = utils.render.make_Rt(pitch=torch.pi / 3, yaw=torch.pi / 4, z=0.5)
                bev = 1 - utils.render.render_point_clouds(
                    points=einops.rearrange(xyz, "B C H W -> B (H W) C"),
                    colors= 1 - einops.rearrange(colors, "B C H W -> B (H W) C"),
                    R=R.to(xyz),
                    t=t.to(xyz),
            )
            return bev

        def render_gt(x, name_future, sample_number):
            depth = x
            if depth.numel() > 0:
                metric = depth
                mask = (metric > 1.0) & (metric < 80.0)
                xyz = to_xyz(metric) * mask
                        
                xyz = xyz / 80
                z_min, z_max = -2 / 80, 0.5 / 80
                z = (xyz[:, [2]] - z_min) / (z_max - z_min)
                colors = utils.render.colorize(z.clamp(0, 1), cm.GnBu_r) / 255
                R, t = utils.render.make_Rt(pitch=torch.pi / 3, yaw=torch.pi / 4, z=0.5)
                bev = 1 - utils.render.render_point_clouds(
                    points=einops.rearrange(xyz, "B C H W -> B (H W) C"),
                    colors= 1 - einops.rearrange(colors, "B C H W -> B (H W) C"),
                    R=R.to(xyz),
                    t=t.to(xyz),
                    )
            return bev

        # --- Create subfolder named by sample_number ---
        sample_folder = f"./fp/sample_{sample_number:04d}"
        os.makedirs(sample_folder, exist_ok=True)
        # ------------------------------------------

        bev = render_pred(recon, name_future, sample_number)
        # --- Save to subfolder ---
        name_bev = os.path.join(sample_folder, "pred_future.png")
        save_image(bev, name_bev, nrow=5)
        # ---------------------

        # --- Save name.txt ---
        name_txt_path = os.path.join(sample_folder, f"name.txt")
        with open(name_txt_path, 'w') as f:
            f.write("name_past:\n")
            for name in name_past:
                f.write(f"{name}\n")
            f.write("\nname_future:\n")
            for name in name_future:
                f.write(f"{name}\n")
        # ----------------------

        print(sample_number)
        sample_number = sample_number + 1
        if sample_number == 1:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default='path/diffusion_0001200000.pth')
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--sampling_steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=88)
    args = parser.parse_args()
    args.device = torch.device(args.device)
    main(args)