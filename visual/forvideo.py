import torch
from vae import VAE, load_points_as_images
import matplotlib.pyplot as plt
import numpy as np
import einops
import utils.render
import matplotlib.cm as cm
import os
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

def render(x):

    range_mask = (x[:, [0]] >= 1.45) & (x[:, [0]] <= 80.0)
    xyz = to_xyz(x[:, [0]]) * range_mask
    xyz /= 80.0
    
    z_min, z_max = -2 / 80.0, 0.5 / 80.0
    z = (xyz[:, [2]] - z_min) / (z_max - z_min)
    colors = utils.render.colorize(z.clamp(0, 1), cm.GnBu_r) / 255
    R, t = utils.render.make_Rt(pitch=torch.pi / 3, yaw=torch.pi / 4, z=0.57)
    bev = 1 - utils.render.render_point_clouds(
        points=einops.rearrange(xyz, "B C H W -> B (H W) C"),
        colors=1 - einops.rearrange(colors, "B C H W -> B (H W) C"),
        R=R.to(xyz),
        t=t.to(xyz),
    )

    return bev

def load_vae_model(model_path, device="cuda"):
    # Ensure model architecture matches training
    model = VAE(input_dim=1, output_dim=1)
    # model = nn.DataParallel(model) 
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    # model = model.module
    model = model.to(device)
    model.eval()
    return model

def decode_latent(model, latent_vector, device="cuda"):
    with torch.no_grad():
        latent_vector = latent_vector.to(device)
        recon = model.decoder(latent_vector)
    return recon


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configure paths (modify as needed)
    model_path = "vae/checkpoints/vae_epoch_80.pth"

    base_dir = "./"
    ar_folders = [f for f in os.listdir(base_dir) if f.startswith('ar') and os.path.isdir(os.path.join(base_dir, f))]

    # Collect all .pt files excluding egopose files
    all_pt_files = []
    for folder in ar_folders:
        folder_path = os.path.join(base_dir, folder)
        pt_files = glob.glob(os.path.join(folder_path, "*.pt"))
        filtered_pt_files = [f for f in pt_files if "egopose" not in os.path.basename(f)]
        all_pt_files.extend(filtered_pt_files)

    vae_model = load_vae_model(model_path, device)

    save_dir = "./for_video_gen_long"
    os.makedirs(save_dir, exist_ok=True)

    for idx, pt_file_path in enumerate(all_pt_files):
        try:
            latent = torch.load(pt_file_path, map_location=device)
            if latent.dim() == 3:
                latent = latent.unsqueeze(0)

            # Get filename without extension
            base_name = os.path.splitext(os.path.basename(pt_file_path))[0]

            # Build test_point_path
            test_point_path = f"path/dataset/kitti_odometry/dataset/sequences/08/velodyne/{base_name}.bin"

            # Check if test_point_path exists
            if not os.path.exists(test_point_path):
                print(f"Test point file does not exist: {test_point_path}")
                continue

            print(f"Processing {idx+1}/{len(all_pt_files)}: {pt_file_path}")

            # Load test point
            test_point = load_points_as_images(test_point_path)
            test_point = torch.from_numpy(test_point).unsqueeze(0).to(device=device)
            test_point = test_point[:,[0]]

            # Reconstruct image
            recon = decode_latent(vae_model, latent, device)
            point_original = render(test_point)
            point_original = point_original.squeeze().permute(1, 2, 0).cpu().numpy()

            point_recon = render(recon)
            point_recon = point_recon.squeeze().permute(1, 2, 0).cpu().numpy()

            original = test_point[0].permute(1, 2, 0).cpu().numpy()
            recon = recon[0].permute(1, 2, 0).cpu().numpy()

            # original depth map
            fig, ax = plt.subplots()
            ax.imshow(original[..., 0], cmap='GnBu_r')
            ax.axis('off')
            plt.savefig(f"{save_dir}/original_depth_{idx+1:03d}.png", dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # original rendered
            fig, ax = plt.subplots()
            ax.imshow(point_original)
            ax.axis('off')
            plt.savefig(f"{save_dir}/original_rendered_{idx+1:03d}.png", dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

            # recon depth map
            fig, ax = plt.subplots()
            ax.imshow(recon[..., 0], cmap='GnBu_r')
            ax.axis('off')
            plt.savefig(f"{save_dir}/recon_depth_{idx+1:03d}.png", dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

            # recon rendered
            fig, ax = plt.subplots()
            ax.imshow(point_recon)
            ax.axis('off')
            plt.savefig(f"{save_dir}/recon_rendered_{idx+1:03d}.png", dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
            
        except Exception as e:
            print(f"Error processing {pt_file_path}: {str(e)}")
            continue
    
    print('ok')
