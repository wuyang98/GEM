# based on v13: more training data used
import torch
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Union
from basic_module import CircularConv2d, LinearAttention, VectorQuantizer, GeoConverter, NLayerDiscriminator, LiDARNLayerDiscriminator, LiDARNLayerDiscriminatorV2

try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError("Please install Mamba: pip install mamba-ssm")


def square_dist_loss(x, y):
    return torch.sum((x - y) ** 2, dim=1, keepdim=True)

def l1(x, y):
    return torch.abs(x - y)

def square_dist_loss(x, y):
    return torch.sum((x - y) ** 2, dim=1, keepdim=True)

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

VERSION2DISC = {'v0': NLayerDiscriminator, 'v1': LiDARNLayerDiscriminator, 'v2': LiDARNLayerDiscriminatorV2}


class VAEDataset(Dataset):
    def __init__(self, root_configs: Dict[str, Dict[str, List[int]]], splits: str):

        self.root_configs = root_configs
        self.splits = splits
        self.file_paths = []
        self.load_path()

    def load_path(self):
        self.file_paths = []

        for root_dir, config in self.root_configs.items():
            root_path = Path(root_dir)

            if self.splits not in config:
                raise ValueError(f"splits '{self.splits}' not found in config for root: {root_dir}")

            splits = config[self.splits]
            self._load_kitti_odometry_paths(root_path, splits)

    def _load_kitti_odometry_paths(self, root_path: Path, splits: List[int]):
        """Load KITTI Odometry dataset paths."""
        split_dirs = [f"{split:02d}" for split in splits]
        
        for split_dir in split_dirs:
            velodyne_dir = root_path / split_dir / "velodyne"
            if velodyne_dir.exists():
                self.file_paths += sorted(velodyne_dir.glob("*.bin"))
            else:
                print(f"Warning: Directory {velodyne_dir} does not exist")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        point_path = self.file_paths[idx]
        return load_points_as_images(point_path)
    
def scatter(array, index, value):
    for (h, w), v in zip(index, value):
        array[h, w] = v
    return array

def load_points_as_images(
    point_path: str,
    scan_unfolding: bool = True,
    H: int = 64,
    W: int = 1024,
):
    # load xyz & intensity and add depth & mask
    points = np.fromfile(point_path, dtype=np.float32).reshape((-1, 4))
    xyz = points[:, :3]  # xyz
    x = xyz[:, [0]]
    y = xyz[:, [1]]
    z = xyz[:, [2]]
    depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)
    mask = (depth >= 1.45) & (depth <= 80)
    points = np.concatenate([depth, mask], axis=1)

    h_up, h_down = np.deg2rad(3), np.deg2rad(-25)
    elevation = np.arcsin(z / depth) + abs(h_down)
    grid_h = 1 - elevation / (h_up - h_down)
    grid_h = np.floor(grid_h * H).clip(0, H - 1).astype(np.int32)

    # horizontal grid
    azimuth = -np.arctan2(y, x)  # [-pi,pi]
    grid_w = (azimuth / np.pi + 1) / 2 % 1  # [0,1]
    grid_w = np.floor(grid_w * W).clip(0, W - 1).astype(np.int32)

    grid = np.concatenate((grid_h, grid_w), axis=1)

    # projection
    order = np.argsort(-depth.squeeze(1))
    proj_points = np.zeros((H, W, 2), dtype=points.dtype)
    proj_points = scatter(proj_points, grid[order], points[order])
    proj_points = proj_points.transpose(2, 0, 1)
    proj_points = proj_points.astype(np.float32)
    proj_points = proj_points[[0]] * proj_points[[1]]

    return proj_points

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x, temb=None):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_
    
class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""

    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)
    
    def forward(self, x, temb=None):
        return super().forward(x)

class MambaBlock(nn.Module):
    def __init__(self, in_channels, d_state=16, d_conv=4, expand=2, **kwargs):
        super().__init__()
        self.norm = Normalize(in_channels)
        # Reduce Mamba parameter count and memory usage
        self.mamba = Mamba(
            d_model=in_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            **kwargs
        )

    def forward(self, x, temb=None):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        h = self.norm(x)
        # Reshape to [B, L, C] where L = H * W
        h = h.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        h = self.mamba(h)
        # Reshape back to [B, C, H, W]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x + h  # residual

def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none", "mamba"], f'attn_type {attn_type} unknown'
    # print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity()
    elif attn_type == "mamba":
        return MambaBlock(in_channels)
    else:
        return LinAttnBlock(in_channels)

DOWNSAMPLE_STRIDE2KERNEL_DICT = {(1, 2): (3, 3), (1, 4): (3, 5), (2, 1): (3, 3), (2, 2): (3, 3)}
DOWNSAMPLE_STRIDE2PAD_DICT = {(1, 2): (0, 1, 1, 1), (1, 4): (1, 1, 1, 1), (2, 1): (1, 1, 1, 1), (2, 2): (0, 1, 0, 1)}


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, stride):
        super().__init__()
        self.with_conv = with_conv
        self.stride = stride
        if self.with_conv:
            k, p = DOWNSAMPLE_STRIDE2KERNEL_DICT[stride], DOWNSAMPLE_STRIDE2PAD_DICT[stride]
            self.conv = CircularConv2d(in_channels, in_channels, kernel_size=k, stride=stride, padding=p)

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)  # modified for lidar
        return x

UPSAMPLE_STRIDE2KERNEL_DICT = {(1, 2): (1, 5), (1, 4): (1, 7), (2, 1): (5, 1), (2, 2): (3, 3)}
UPSAMPLE_STRIDE2PAD_DICT = {(1, 2): (2, 2, 0, 0), (1, 4): (3, 3, 0, 0), (2, 1): (0, 0, 2, 2), (2, 2): (1, 1, 1, 1)}


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, stride):
        super().__init__()
        self.with_conv = with_conv
        self.stride = stride
        if self.with_conv:
            k, p = UPSAMPLE_STRIDE2KERNEL_DICT[stride], UPSAMPLE_STRIDE2PAD_DICT[stride]
            self.conv = CircularConv2d(in_channels, in_channels, kernel_size=k, padding=p)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=self.stride, mode='bilinear', align_corners=True)
        if self.with_conv:
            x = self.conv(x)
        return x

UNIFORM_KERNEL2PAD_DICT = {(3, 3): (1, 1, 1, 1), (1, 4): (1, 2, 0, 0)}


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, kernel_size=(3, 3), conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        pad = UNIFORM_KERNEL2PAD_DICT[kernel_size]

        self.norm1 = Normalize(in_channels)
        self.conv1 = CircularConv2d(in_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=pad)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = CircularConv2d(out_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=pad)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CircularConv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=1,
                                                    padding=pad)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

class Encoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Channel multiplier (fewer levels but more capacity per level)
        ch_mult = [1, 2, 4]  # Originally [1,2,4,8]
        num_res_blocks = 2
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.temb_ch = 0
        ch = 64
        self.ch = ch
        dropout = 0.0

        # Downsample strides (keep height at 32, width 1024 -> 256)
        strides = [[1,2], [1,2]]  # Only downsample width twice

        attn_levels = [0]  # Use Mamba only at the shallowest level
        attn_type = "mamba"
        deep_attn_type = "none"  # No attention at deep levels

        self.conv_in = CircularConv2d(input_dim, ch, kernel_size=3, stride=1, padding=1)

        self.down = nn.ModuleList()
        block_in = ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=ch * ch_mult[i_level],
                                         dropout=dropout))
                block_in = ch * ch_mult[i_level]
                if i_level in attn_levels:
                    block.append(make_attn(block_in, attn_type))
                # Deep levels: attention fully removed to save memory
                # elif i_level > 0:
                #     block.append(make_attn(block_in, deep_attn_type))

            # Downsample layer (no downsampling at the last stage)
            down = nn.Module()
            down.block = block
            if i_level < len(strides):
                down.downsample = Downsample(block_in, True, tuple(strides[i_level]))
            self.down.append(down)
        
        # Mid blocks (attention removed)
        self.mid_block1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        # self.mid_attn = make_attn(block_in, deep_attn_type)  # fully removed
        self.mid_block2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # Output layer (target dimensions 3 x 32 x 256)
        self.norm_out = Normalize(block_in)
        z_channels = 3
        self.conv_out = CircularConv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

        self.quantize = VectorQuantizer(16384, z_channels, beta=0.25)
        self.quant_conv = nn.Conv2d(z_channels, z_channels, 1)

    def forward(self, x):
        h = self.conv_in(x) # B, 1, 32, 1024 --> B, 64, 32, 1024

        # Downsampling path
        for i_level in range(self.num_resolutions):
            for block in self.down[i_level].block:
                h = block(h, temb=None)
            if hasattr(self.down[i_level], 'downsample'):
                h = self.down[i_level].downsample(h)

        # Mid blocks (attention removed)
        h = self.mid_block1(h, temb=None)
        # h = self.mid_attn(h)  # fully removed
        h = self.mid_block2(h, temb=None)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h) # B, 256, 32, 256

        # Quantization
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        quant = torch.tanh(quant)
        
        return quant, emb_loss, info
    
class Decoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        ch_mult = [4, 2, 1]  # Symmetric to encoder
        num_res_blocks = 2
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.temb_ch = 0
        block_in = 256  # Matches encoder's final channel count

        # Upsample strides (symmetric to encoder)
        strides = [[1,2], [1,2]]  # Upsample width twice

        attn_type = "none"  # Decoder also removes attention to save memory

        # Post-quantization processing (input channels adjusted to 3)
        self.post_quant_conv = nn.Conv2d(3, block_in, 1)  # 3 -> 256

        # Mid blocks
        self.mid_block1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=0.0)
        self.mid_block2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=0.0)

        self.up = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=ch_mult[i_level]*64,
                                         dropout=0.0))
                block_in = ch_mult[i_level]*64
            up = nn.Module()
            up.block = block

            # Upsample layer (no upsampling at the first level)
            if i_level < len(strides):
                up.upsample = Upsample(block_in, True, tuple(strides[i_level]))
            self.up.append(up)
        
        # Output layer
        self.norm_out = Normalize(block_in)
        self.conv_out = CircularConv2d(block_in, output_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, quant):
        h = self.post_quant_conv(quant)

        # Mid blocks (attention removed)
        h = self.mid_block1(h, temb=None)
        h = self.mid_block2(h, temb=None)

        # Upsampling path
        for i_level in range(self.num_resolutions):
            for block in self.up[i_level].block:
                h = block(h, temb=None)
            if hasattr(self.up[i_level], 'upsample'):
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class VAE(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(Encoder(input_dim))
        self.decoder = nn.SyncBatchNorm.convert_sync_batchnorm(Decoder(output_dim))
        
        # Discriminator
        self.discriminator = nn.SyncBatchNorm.convert_sync_batchnorm(LiDARNLayerDiscriminatorV2(
            input_nc=input_dim, 
            n_layers=4, 
            ndf=128,
            use_actnorm=True
        ).apply(weights_init))

    def forward(self, x):
        quant, qloss, info = self.encoder(x)
        xrec = self.decoder(quant)
        
        return xrec, qloss, info
    
    def training_step(self, batch, optimizer_idx):
        data = batch
        xrec, qloss, _ = self(data)
        
        if optimizer_idx == 0:

            pixel_loss = l1(data, xrec).mean()

            logits_fake = self.discriminator(xrec.detach())
            g_loss = -torch.mean(logits_fake)

            total_loss = 0.8*pixel_loss + 0.2*g_loss + qloss.mean()

            return xrec, total_loss

        else:
            logits_real = self.discriminator(data.detach())
            logits_fake = self.discriminator(xrec.detach())
            d_loss = 0.5 * (F.relu(1.0 - logits_real).mean() + F.relu(1.0 + logits_fake).mean())
            return xrec, d_loss


def train_vae(config):
    os.makedirs("vae/fig", exist_ok=True)
    os.makedirs("vae/checkpoints", exist_ok=True)

    # debug
    # local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # global_rank = int(os.environ.get("RANK", "0"))
    # world_size = int(os.environ.get("WORLD_SIZE", "1"))
    # # os.environ['MASTER_ADDR'] = 'localhost'  # master node address (local)
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '5679'       # communication port (custom)
    # torch.cuda.set_device(local_rank)
    # device = torch.device("cuda", local_rank)
    # dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=global_rank)

    # train
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")
    print(f"Requested local_rank: {local_rank}")
    if local_rank >= available_gpus:
        raise RuntimeError(f"local_rank {local_rank} exceeds available GPU count {available_gpus}")
    actual_device = local_rank % torch.cuda.device_count()
    torch.cuda.set_device(actual_device)
    device = torch.device("cuda", actual_device)
    import datetime
    dist.init_process_group(
        backend='nccl', 
        init_method='env://', 
        world_size=world_size, 
        rank=global_rank,
        timeout=datetime.timedelta(seconds=600)  # 10-minute timeout
    )
    

    root_configs = {
        "path/to/kitti_odometry/dataset/sequences": {
            "TRAIN": [0, 1, 2, 3, 4, 5, 6, 7],
            "TEST": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            "ALL": [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        }
    }

    train_dataset = VAEDataset(root_configs=root_configs, splits='ALL')
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    model = VAE(
        input_dim=config['input_dim'],
        output_dim=config['output_dim'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_parameters(model)
    print("="*50)
    print(f"VAE model size:",
          f"Total params: {total_params:,} (~{total_params/1e6:.2f}M)",
          f"Trainable params: {trainable_params:,} (~{trainable_params/1e6:.2f}M)",
          f"Non-trainable params: {total_params - trainable_params:,}")
    
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True
    )
    #------------------------------------------------------------------------
    original_model = model.module
    ae_params = list(original_model.encoder.parameters()) + \
                list(original_model.decoder.parameters())
    disc_params = list(original_model.discriminator.parameters())

    optimizer_ae = optim.Adam(ae_params, lr=config['lr'])
    optimizer_disc = optim.Adam(disc_params, lr=config['lr']*0.2)
    optimizers = [optimizer_ae, optimizer_disc]

    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler_ae = CosineAnnealingLR(
        optimizer_ae,
        T_max=config['epochs'],
        eta_min=config['lr']*0.01
    )
    scheduler_disc = CosineAnnealingLR(
        optimizer_disc,
        T_max=config['epochs']//2,
        eta_min=config['lr']*0.004
    )
    schedulers = [scheduler_ae, scheduler_disc]
    
    for epoch in range(config['epochs']):
        train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0
        if global_rank == 0:
            current_lr_ae = optimizer_ae.param_groups[0]['lr']
            current_lr_disc = optimizer_disc.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{config['epochs']} - LR: AE={current_lr_ae:.2e}, Disc={current_lr_disc:.2e}")

        for batch_idx, data in enumerate(train_loader):
            total_loss = 0
            # data = data_process(data=data)
            data = data.to(device, non_blocking=True)

            for optimizer_idx in range(2):  # 0: autoencoder, 1: discriminator
                optimizers[optimizer_idx].zero_grad()

                xrec, loss = model.module.training_step(data, optimizer_idx)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters() if optimizer_idx == 0 else disc_params, 
                    max_norm=1.0
                )
                
                optimizers[optimizer_idx].step()
                total_loss += loss.item()

            if global_rank == 0 and batch_idx % 2 == 0:
                avg_loss = total_loss / (batch_idx + 1) / config['batch_size']
                print(f'Epoch {epoch+1}/{config["epochs"]} | '
                      f'Batch {batch_idx}/{len(train_loader)} | '
                      f'Loss: {avg_loss:.4f}')
            if global_rank == 0 and batch_idx % 1600 == 0:
                with torch.no_grad():
                    # Take the first sample for visualization
                    original = data[0].permute(1, 2, 0).cpu().numpy()
                    recon = xrec[0].permute(1, 2, 0).cpu().numpy()
                    
                    # original depth map
                    plt.figure(figsize=(15, 5), dpi=300)
                    plt.subplot(1, 2, 1)
                    plt.imshow(original[..., 0], cmap='jet')
                    plt.title(f"original (Epoch {epoch+1})")
                    
                    # recon depth map
                    plt.subplot(1, 2, 2)
                    plt.imshow(recon[..., 0], cmap='jet')
                    plt.title(f"recon (Epoch {epoch+1})")
                    
                    plt.tight_layout()
                    plt.savefig(f"vae/fig/recon_depth_epoch_{epoch+1}_{batch_idx+1}.png")
                    plt.close()

        for scheduler in schedulers:
            scheduler.step()
        if local_rank == 0 and (epoch+1) % config['save_interval'] == 0:
        # if (epoch + 1) % config['save_interval'] == 0:
            torch.save(model.module.state_dict(), f"vae/checkpoints/vae_epoch_{epoch+1}.pth")

    
    print("Training complete!")
    return model


if __name__ == "__main__":
    config = {
        'input_dim': 1,
        'output_dim': 1,
        'batch_size': 8,
        'epochs': 80,
        'lr': 4e-4,
        'num_workers': 16,
        'save_interval': 10
    }
    
    # Start training
    trained_model = train_vae(config)

    # Save final model
    torch.save(trained_model.state_dict(), "vae/checkpoints/vae_final.pth")