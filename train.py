import dataclasses
import datetime
import json
import os
import warnings
from pathlib import Path
import datasets as ds
import einops
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from ema_pytorch import EMA
from rich import print
from simple_parsing import ArgumentParser
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from basic_module import CustomPreDataset
import utils.inference
import utils.option
import utils.render
import utils.training
from models.diffusion import (
    ContinuousTimeGaussianDiffusion,
    DiscreteTimeGaussianDiffusion,
)
from models.video_unet import EfficientUNet
from utils.lidar import LiDARUtility, get_hdl64e_linear_ray_angles
import torch.nn as nn
from vae import VAE
import matplotlib.pyplot as plt
import numpy as np

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.automatic_dynamic_shapes = False

def load_vae_model(model_path, device="cuda"):
    # Ensure model architecture matches training
    model = VAE(input_dim=1, output_dim=1)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    return model

def train(cfg: utils.option.Config):
    torch.backends.cudnn.benchmark = True
    project_dir = Path(cfg.training.output_dir) / cfg.data.dataset / cfg.data.projection

    # =================================================================================
    # Initialize accelerator
    # =================================================================================

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with=["tensorboard"],
        project_dir=project_dir,
        dynamo_backend=cfg.training.dynamo_backend,
        split_batches=True,
        step_scheduler_with_optimizer=True,
    )
    if accelerator.is_main_process:
        print(cfg)
        os.makedirs(project_dir, exist_ok=True)
        project_name = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        accelerator.init_trackers(project_name=project_name)
        tracker = accelerator.get_tracker("tensorboard")
        json.dump(
            dataclasses.asdict(cfg),
            open(Path(tracker.logging_dir) / "training_config.json", "w"),
            indent=4,
        )
    device = accelerator.device

    # v17_kitti vae
    model_path = "vae/checkpoints/vae_epoch_80.pth"
    vae = load_vae_model(model_path, device)
    vae.requires_grad_(False)
    vae.eval()

    # =================================================================================
    # Setup models
    # =================================================================================

    channels = [
        1 if cfg.data.train_depth else 0,
        1 if cfg.data.train_reflectance else 0,
        3 if cfg.data.train_vae else 0,
    ]

    if cfg.model.architecture == "video":
        model = EfficientUNet(
            in_channels=sum(channels),
            resolution=cfg.data.resolution,
            base_channels=cfg.model.base_channels,
            temb_channels=cfg.model.temb_channels,
            channel_multiplier=cfg.model.channel_multiplier,
            num_residual_blocks=cfg.model.num_residual_blocks,
            gn_num_groups=cfg.model.gn_num_groups,
            gn_eps=cfg.model.gn_eps,
            attn_num_heads=cfg.model.attn_num_heads,
            coords_encoding=cfg.model.coords_encoding,
            # ring=True,
        )
    else:
        raise ValueError(f"Unknown: {cfg.model.architecture}")

    if accelerator.is_main_process:
        print(f"number of parameters: {utils.inference.count_parameters(model):,}")
        print(f"number of vae: {utils.inference.count_parameters(vae):,}")

    if cfg.diffusion.timestep_type == "discrete":
        ddpm = DiscreteTimeGaussianDiffusion(
            model=model,
            prediction_type=cfg.diffusion.prediction_type,
            loss_type=cfg.diffusion.loss_type,
            noise_schedule=cfg.diffusion.noise_schedule,
            num_training_steps=cfg.diffusion.num_training_steps,
        )
    elif cfg.diffusion.timestep_type == "continuous":
        ddpm = ContinuousTimeGaussianDiffusion(
            model=model,
            prediction_type=cfg.diffusion.prediction_type,
            loss_type=cfg.diffusion.loss_type,
            noise_schedule=cfg.diffusion.noise_schedule,
        )
    else:
        raise ValueError(f"Unknown: {cfg.diffusion.timestep_type}")
    
    ddpm.train()
    ddpm.to(device)

    if accelerator.is_main_process:
        ddpm_ema = EMA(
            ddpm,
            beta=cfg.training.ema_decay,
            update_every=cfg.training.ema_update_every,
            update_after_step=cfg.training.lr_warmup_steps
            * cfg.training.gradient_accumulation_steps,
        )
        ddpm_ema.to(device)

    lidar_utils = LiDARUtility(
        resolution=cfg.data.resolution_recon,
        depth_format=cfg.data.depth_format,
        min_depth=cfg.data.min_depth,
        max_depth=cfg.data.max_depth,
        # ray_angles=ddpm.model.coords,
        ray_angles=get_hdl64e_linear_ray_angles(*cfg.data.resolution_recon)
    )
    lidar_utils.to(device)

    # =================================================================================
    # Setup optimizer & dataloader
    # =================================================================================

    optimizer = torch.optim.AdamW(
        ddpm.parameters(),
        lr=cfg.training.lr,
        betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
        weight_decay=cfg.training.adam_weight_decay,
        eps=cfg.training.adam_epsilon,
    )

    root_dirs=['path/dataset_processed/kitti_odometry_vae_train']
    dataset = CustomPreDataset(
        root_dirs=root_dirs,
        split="TRAIN",
        sequence_length=10,
        frame_step=6    
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size_train,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    lr_scheduler = utils.training.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps
        * cfg.training.gradient_accumulation_steps,
        num_training_steps=cfg.training.num_steps
        * cfg.training.gradient_accumulation_steps,
    )

    ddpm, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        ddpm, optimizer, dataloader, lr_scheduler
    )

    # =================================================================================
    # Utility
    # =================================================================================

    def preprocess(batch):
        x_frame, egopose = batch[0], batch[1]
        x_past = x_frame[:, 0:5]
        x_future = x_frame[:, 5:10]
        egopose_past = egopose[:, 0:5]
        egopose_future = egopose[:, 5:10]

        return x_past, x_future, egopose_past, egopose_future

    @torch.inference_mode()
    def log_images(image, tag: str = "name", global_step: int = 0):
        out = dict()
        depth = image
        if depth.numel() > 0:
            metric = depth
            mask = (metric >= lidar_utils.min_depth) & (metric <= lidar_utils.max_depth)
            out[f"{tag}/depth/orig"] = utils.render.colorize(metric / lidar_utils.max_depth)
            xyz = lidar_utils.to_xyz(metric) * mask

            xyz = xyz / 80

            z_min, z_max = -2 / 80, 0.5 / 80
            z = (xyz[:, [2]] - z_min) / (z_max - z_min)
            colors = utils.render.colorize(z.clamp(0, 1), cm.RdBu_r) / 255
            R, t = utils.render.make_Rt(pitch=torch.pi / 3, yaw=torch.pi / 4, z=0.7)
            bev = utils.render.render_point_clouds(
                points=einops.rearrange(xyz, "B C H W -> B (H W) C"),
                colors=einops.rearrange(colors, "B C H W -> B (H W) C"),
                # t=torch.tensor([0, 0, 1.0]).to(xyz),
                R=R.to(xyz),
                t=t.to(xyz),
            )
            out[f"{tag}/bev"] = bev.mul(255).clamp(0, 255).byte()

        if mask.numel() > 0:
            out[f"{tag}/mask"] = utils.render.colorize(mask, cm.binary_r)
        tracker.log_images(out, step=global_step)

    # =================================================================================
    # Training loop
    # =================================================================================

    progress_bar = tqdm(
        range(cfg.training.num_steps),
        desc="training",
        dynamic_ncols=True,
        disable=not accelerator.is_main_process,
    )

    global_step = 0
    while global_step < cfg.training.num_steps:
        ddpm.train()
        for batch in dataloader:
            x_past, x_future, egopose_past, egopose_future = preprocess(batch)
            with accelerator.accumulate(ddpm):
                loss = ddpm(x_0=x_past, x_future=x_future, 
                            egopose_past=egopose_past, egopose_future=egopose_future)
                accelerator.backward(loss)
                # for name, param in ddpm.named_parameters():
                #     if param.grad is None:
                #         print(name)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            log = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
            if accelerator.is_main_process:
                ddpm_ema.update()
                log["ema/decay"] = ddpm_ema.get_current_decay()

                if global_step == 1:
                    with torch.no_grad():
                        # only show the firt frames
                        x_all = torch.cat([x_past, x_future], dim=1)
                        x_first = x_all[1]
                        recon = vae.decoder(x_first)
                        torch.cuda.empty_cache()
                    log_images(recon, "image", global_step)

                if global_step % cfg.training.steps_save_image == 0:
                    ddpm_ema.ema_model.eval()
                    sample = ddpm_ema.ema_model.sample(
                        batch_size=cfg.training.batch_size_eval,
                        num_steps=cfg.diffusion.num_sampling_steps,
                        x_past = x_past,
                        egopose_past = egopose_past,
                        egopose_future = egopose_future,
                        frames=cfg.training.frames,
                        rng=torch.Generator(device=device).manual_seed(0),
                    )
                    with torch.no_grad():
                        # only show the firt samples
                        x_all = torch.cat([x_past, sample], dim=1)
                        sample_first = x_all[0]
                        recon = vae.decoder(sample_first)
                        torch.cuda.empty_cache()
                    log_images(recon, "sample", global_step)

                if global_step % cfg.training.steps_save_model == 0:
                    save_dir = Path(tracker.logging_dir) / "models"
                    save_dir.mkdir(exist_ok=True, parents=True)
                    torch.save(
                        {
                            "cfg": dataclasses.asdict(cfg),
                            "weights": ddpm_ema.online_model.state_dict(),
                            "ema_weights": ddpm_ema.ema_model.state_dict(),
                            "global_step": global_step,
                        },
                        save_dir / f"diffusion_{global_step:010d}.pth",
                    )

            accelerator.log(log, step=global_step)
            progress_bar.update(1)

            if global_step >= cfg.training.num_steps:
                break

    accelerator.end_training()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(utils.option.Config, dest="cfg")
    train(parser.parse_args().cfg)
