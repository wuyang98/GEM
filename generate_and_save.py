import argparse
from pathlib import Path
import torch
from basic_module import CustomPreDataset
import utils.inference
from vae import VAE
from torch.utils.data import DataLoader
import os

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
    model = VAE(input_dim=1, output_dim=1)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
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
            torch.cuda.empty_cache()


        def render_pred(x, name_future, sample_number):
            depth = x
            if depth.numel() > 0:
                metric = depth
                mask = (metric > 1.45) & (metric < 80.0)
                xyz = to_xyz(metric) * mask

                points = xyz
                for k in range(points.shape[0]):
                    name_point = f"{args.output_dir}/{sample_number:04d}_{name_future[k][0]}"
                    name_point = os.path.splitext(name_point)[0] + '.bin'
                    a = points[k].reshape(3, -1).transpose(1, 0)
                    point1 = a.float().reshape(-1)
                    os.makedirs(os.path.dirname(name_point), exist_ok=True)
                    with open(name_point, 'wb') as f:
                        point1.cpu().numpy().tofile(f)

        render_pred(recon, name_future, sample_number)


        sample_number = sample_number + 1
        if args.num_samples and sample_number >= args.num_samples:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default='path/diffusion_0001200000.pth')
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--sampling_steps", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="path/results/kitti_odometry_3s", help="Directory to save output .bin files")
    parser.add_argument("--num_samples", type=int, default=6701, help="Number of samples to generate (default: all)")
    parser.add_argument("--seed", type=int, default=88)
    args = parser.parse_args()
    args.device = torch.device(args.device)
    main(args)