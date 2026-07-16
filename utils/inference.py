from pathlib import Path

import torch

from models.diffusion import (
    ContinuousTimeGaussianDiffusion,
    DiscreteTimeGaussianDiffusion,
    GaussianDiffusion,
)
from models.video_unet import EfficientUNet
# from models.video_unet import LiDARGenRefineNet
from utils.lidar import LiDARUtility, get_hdl64e_linear_ray_angles
from utils.option import Config

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # return sum(p.numel() for p in model.parameters())


def setup_model(
    ckpt: str | Path | dict,
    device: torch.device | str = "cpu",
    ema: bool = True,
    show_info: bool = True,
    compile_denoiser: bool = False,
) -> tuple[GaussianDiffusion, LiDARUtility, Config]:
    if isinstance(ckpt, (str, Path)):
        ckpt = torch.load(ckpt, map_location="cpu")
    cfg = Config(**ckpt["cfg"])

    # in_channels = [0, 0]
    # if cfg.data['train_depth']:
    #     in_channels[0] = 1
    # if cfg.data['train_reflectance']:
    #     in_channels[1] = 1
    # in_channels = sum(in_channels)
    
    in_channels = [0, 0, 0]
    if cfg.data['train_depth']:
        in_channels[0] = 1
    if cfg.data['train_reflectance']:
        in_channels[1] = 1
    if cfg.data['train_vae']:
        in_channels[2] = 3
    in_channels = sum(in_channels)

    if cfg.model['architecture'] == "video":
        model = EfficientUNet(
            in_channels=in_channels,
            resolution=cfg.data['resolution'],
            base_channels=cfg.model['base_channels'],
            temb_channels=cfg.model['temb_channels'],
            channel_multiplier=cfg.model['channel_multiplier'],
            num_residual_blocks=cfg.model['num_residual_blocks'],
            gn_num_groups=cfg.model['gn_num_groups'],
            gn_eps=cfg.model['gn_eps'],
            attn_num_heads=cfg.model['attn_num_heads'],
            coords_encoding=cfg.model['coords_encoding'],
            # ring=True,
        )
    else:
        raise ValueError(f"Unknown: {cfg.model.architecture}")

    if cfg.diffusion['timestep_type'] == "discrete":
        ddpm = DiscreteTimeGaussianDiffusion(
            model=model,
            loss_type=cfg.diffusion['loss_type'],
            num_training_steps=cfg.diffusion['num_training_steps'],
            prediction_type=cfg.diffusion['prediction_type'],
            noise_schedule=cfg.diffusion['noise_schedule'],
        )
    elif cfg.diffusion['timestep_type'] == "continuous":
        ddpm = ContinuousTimeGaussianDiffusion(
            model=model,
            loss_type=cfg.diffusion['loss_type'],
            prediction_type=cfg.diffusion['prediction_type'],
            noise_schedule=cfg.diffusion['noise_schedule'],
        )
    else:
        raise ValueError(f"Unknown: {cfg.diffusion.timestep_type}")

    state_dict = ckpt["ema_weights"] if ema else ckpt["weights"]
    ddpm.load_state_dict(state_dict)
    ddpm.eval()
    ddpm.to(device)

    if compile_denoiser:
        ddpm.model = torch.compile(ddpm.model)

    lidar_utils = LiDARUtility(
        resolution=cfg.data['resolution_recon'],
        depth_format=cfg.data['depth_format'],
        min_depth=2.0,
        max_depth=75.0,
        ray_angles=get_hdl64e_linear_ray_angles(*cfg.data['resolution_recon']),
    )
    lidar_utils.eval()
    lidar_utils.to(device)

    if show_info:
        print(
            *[
                f"resolution: {model.resolution}",
                f"denoiser: {model.__class__.__name__}",
                f"diffusion: {ddpm.__class__.__name__}",
                f'#steps:  {ckpt["global_step"]:,}',
                f"#params: {count_parameters(ddpm):,}",
            ],
            sep="\n",
        )

    return ddpm, lidar_utils, cfg


def load_model(
    ckpt: str | Path | dict,
    device: torch.device | str = "cpu",
    ema: bool = True,
    show_info: bool = True,
    compile_denoiser: bool = False,
) -> tuple[GaussianDiffusion, LiDARUtility, Config]:
    if isinstance(ckpt, (str, Path)):
        ckpt = torch.load(ckpt, map_location="cpu")
    cfg = Config(**ckpt["cfg"])

    channels = [
        1 if cfg.data['train_depth'] else 0,
        1 if cfg.data['train_reflectance'] else 0,
        3 if cfg.data['train_vae'] else 0,
    ]
    in_channels = sum(channels)

    if cfg.model['architecture'] == "video":
        model = EfficientUNet(
            in_channels=sum(channels),
            resolution=cfg.data['resolution'],
            base_channels=cfg.model['base_channels'],
            temb_channels=cfg.model['temb_channels'],
            channel_multiplier=cfg.model['channel_multiplier'],
            num_residual_blocks=cfg.model['num_residual_blocks'],
            gn_num_groups=cfg.model['gn_num_groups'],
            gn_eps=cfg.model['gn_eps'],
            attn_num_heads=cfg.model['attn_num_heads'],
            coords_encoding=cfg.model['coords_encoding'],
            # ring=True,
        )
    else:
        raise ValueError(f"Unknown: {cfg.model['architecture']}")

    if cfg.diffusion['timestep_type'] == "discrete":
        ddpm = DiscreteTimeGaussianDiffusion(
            model=model,
            loss_type=cfg.diffusion['loss_type'],
            num_training_steps=cfg.diffusion['num_training_steps'],
            prediction_type=cfg.diffusion['prediction_type'],
            noise_schedule=cfg.diffusion['noise_schedule'],
        )
    elif cfg.diffusion['timestep_type'] == "continuous":
        ddpm = ContinuousTimeGaussianDiffusion(
            model=model,
            loss_type=cfg.diffusion['loss_type'],
            prediction_type=cfg.diffusion['prediction_type'],
            noise_schedule=cfg.diffusion['noise_schedule'],
        )
    else:
        raise ValueError(f"Unknown: {cfg.diffusion.timestep_type}")

    state_dict = ckpt["ema_weights"] if ema else ckpt["weights"]
    ddpm.load_state_dict(state_dict)
    return ddpm

def setup_rng(seeds: list[int], device: torch.device | str):
    return [torch.Generator(device=device).manual_seed(i) for i in seeds]
