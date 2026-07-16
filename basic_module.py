import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from einops import rearrange, repeat
from functools import partial
import functools
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset
import os
import glob
from nuscenes import NuScenes


class CircularConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        if 'padding' in kwargs:
            self.is_pad = True
            if isinstance(kwargs['padding'], int):
                h1 = h2 = v1 = v2 = kwargs['padding']
            elif isinstance(kwargs['padding'], tuple):
                h1, h2, v1, v2 = kwargs['padding']
            else:
                raise NotImplementedError
            self.h_pad, self.v_pad = (h1, h2, 0, 0), (0, 0, v1, v2)
            del kwargs['padding']
        else:
            self.is_pad = False

        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        if self.is_pad:
            if sum(self.h_pad) > 0:
                x = nn.functional.pad(x, self.h_pad, mode="circular")  # horizontal pad
            if sum(self.v_pad) > 0:
                x = nn.functional.pad(x, self.v_pad, mode="constant")  # vertical pad
        x = self._conv_forward(x, self.weight, self.bias)
        return x
    

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)
    
class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)
    
class GeoConverter(nn.Module):
    def __init__(self, curve_length=4, bev_only=False, dataset_config=dict()):
        super().__init__()
        self.curve_length = curve_length
        self.coord_dim = 3 if not bev_only else 2
        self.convert_fn = self.batch_range2bev if bev_only else self.batch_range2xyz

        fov = dataset_config['fov']
        self.fov_up = fov[0] / 180.0 * np.pi  # field of view up in rad
        self.fov_down = fov[1] / 180.0 * np.pi  # field of view down in rad
        self.fov_range = abs(self.fov_down) + abs(self.fov_up)  # get field of view total in rad
        self.depth_scale = dataset_config['depth_scale']
        self.depth_min, self.depth_max = dataset_config['depth_range']
        self.log_scale = dataset_config['log_scale']
        self.size = dataset_config['size']
        self.register_conversion()

    def register_conversion(self):
        scan_x, scan_y = np.meshgrid(np.arange(self.size[1]), np.arange(self.size[0]))
        scan_x = scan_x.astype(np.float64) / self.size[1]
        scan_y = scan_y.astype(np.float64) / self.size[0]

        yaw = (np.pi * (scan_x * 2 - 1))
        pitch = ((1.0 - scan_y) * self.fov_range - abs(self.fov_down))

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('cos_yaw', torch.cos(to_torch(yaw)))
        self.register_buffer('sin_yaw', torch.sin(to_torch(yaw)))
        self.register_buffer('cos_pitch', torch.cos(to_torch(pitch)))
        self.register_buffer('sin_pitch', torch.sin(to_torch(pitch)))

    def batch_range2xyz(self, imgs):
        batch_depth = (imgs * 0.5 + 0.5) * self.depth_scale
        if self.log_scale:
            batch_depth = torch.exp2(batch_depth) - 1
        batch_depth = batch_depth.clamp(self.depth_min, self.depth_max)

        batch_x = self.cos_yaw * self.cos_pitch * batch_depth
        batch_y = -self.sin_yaw * self.cos_pitch * batch_depth
        batch_z = self.sin_pitch * batch_depth
        batch_xyz = torch.cat([batch_x, batch_y, batch_z], dim=1)

        return batch_xyz
    
    def batch_range2bev(self, imgs):
        batch_depth = (imgs * 0.5 + 0.5) * self.depth_scale
        if self.log_scale:
            batch_depth = torch.exp2(batch_depth) - 1
        batch_depth = batch_depth.clamp(self.depth_min, self.depth_max)

        batch_x = self.cos_yaw * self.cos_pitch * batch_depth
        batch_y = -self.sin_yaw * self.cos_pitch * batch_depth
        batch_bev = torch.cat([batch_x, batch_y], dim=1)

        return batch_bev

    def curve_compress(self, batch_coord):
        compressed_batch_coord = F.avg_pool2d(batch_coord, (1, self.curve_length))

        return compressed_batch_coord

    def forward(self, input):
        input = input / 2. + .5  # [-1, 1] -> [0, 1]

        input_coord = self.convert_fn(input)
        if self.curve_length > 1:
            input_coord = self.curve_compress(input_coord)

        return input_coord

class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=1, output_nc=1, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, output_nc, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


class LiDARNLayerDiscriminator(nn.Module):
    """Modified PatchGAN discriminator from Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=1, output_nc=1, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(LiDARNLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = (4, 4)
        sequence = [CircularConv2d(input_nc, ndf, kernel_size=kw, stride=(1, 2), padding=(1, 2, 1, 2)), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                CircularConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=(1, 2), bias=use_bias, padding=(1, 2, 1, 2)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            CircularConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, bias=use_bias, padding=(1, 2, 1, 2)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            CircularConv2d(ndf * nf_mult, output_nc, kernel_size=kw, stride=1, padding=(1, 2, 1, 2))]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


class LiDARNLayerDiscriminatorV2(nn.Module):
    """Modified PatchGAN discriminator from Pix2Pix (larger receptive field)
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=1, output_nc=1, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(LiDARNLayerDiscriminatorV2, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = (4, 4)
        sequence = [CircularConv2d(input_nc, ndf, kernel_size=kw, stride=(1, 2), padding=(1, 2, 1, 2)), nn.LeakyReLU(0.2, True),
                    CircularConv2d(ndf, ndf, kernel_size=kw, stride=(1, 2), padding=(1, 2, 1, 2)), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                CircularConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=(2, 2), bias=use_bias, padding=(1, 2, 1, 2)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            CircularConv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, bias=use_bias, padding=(1, 2, 1, 2)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            CircularConv2d(ndf * nf_mult, output_nc, kernel_size=kw, stride=1, padding=(1, 2, 1, 2))]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)
    
    
class CustomPreDataset(Dataset):
    # KITTI sequence splits
    SPLIT_SEQUENCES = {
        "TRAIN": ["00", "01", "02", "03", "04", "05", "06", "07"],
        "TEST": ["08", "09", "10"],
        "ALL": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    }
    
    def __init__(self, root_dirs, split="train", sequence_length=10, frame_step=6):
        """
        Initialize the dataset, supporting loading from multiple root directories.
        :param root_dirs: list of data root directories
        :param split: data split ("train", "val", "test", "train_small", "trainval", "all")
        :param sequence_length: number of consecutive frames (default 10)
        :param frame_step: frame interval (default 2, i.e. read every other frame)
        """
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
        self.root_dirs = root_dirs
        self.split = split
        
        if split in self.SPLIT_SEQUENCES:
            self.target_sequences = self.SPLIT_SEQUENCES[split]
        else:
            raise ValueError(f"Unknown split: {split}. Available splits: {list(self.SPLIT_SEQUENCES.keys())}")
        
        self.seq_len = sequence_length
        self.frame_step = frame_step
        self.scene_paths = []
        
        for root_dir in self.root_dirs:
            sequences_dir = os.path.join(root_dir, 'sequences')
            if os.path.exists(sequences_dir):
                for seq_name in self.target_sequences:
                    seq_path = os.path.join(sequences_dir, seq_name)
                    if os.path.exists(seq_path):
                        self.scene_paths.append(seq_path)
        
        self.samples = []
        self._build_samples()

    def _build_samples(self):
        """Build sample indices."""
        for scene_path in self.scene_paths:
            seq_name = os.path.basename(scene_path)

            if seq_name not in self.target_sequences:
                continue
                
            point_cloud_files = sorted(
                [f for f in glob.glob(os.path.join(scene_path, '*.pt')) 
                 if '_egopose' not in os.path.basename(f)],
                key=lambda x: self._extract_frame_number(os.path.basename(x))
            )
            
            min_frames_needed = (self.seq_len - 1) * self.frame_step + 1

            if len(point_cloud_files) >= min_frames_needed:
                max_start_idx = len(point_cloud_files) - min_frames_needed + 1
                for start_idx in range(max_start_idx):
                    self.samples.append((scene_path, start_idx))

    def _extract_frame_number(self, filename):
        """Extract frame number from filename for sorting."""
        # Handle filenames like "000000.pt"
        name_without_ext = os.path.splitext(filename)[0]
        if name_without_ext.isdigit():
            return int(name_without_ext)
        else:
            return filename

    def __len__(self):
        """Return total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Load seq_len consecutive frames, sampling every frame_step frames.
        :return:
            - point_cloud_seq: tensor of shape (seq_len, ...)
            - ego_pose_seq: tensor of shape (seq_len, 3) containing xyz coordinates
            - file_paths: List[str] of length seq_len, each formatted as "sequence_name/filename.pt"
        """
        scene_path, start_idx = self.samples[idx]
        sequence_name = os.path.basename(scene_path)

        point_cloud_files = sorted(
            [f for f in glob.glob(os.path.join(scene_path, '*.pt')) 
            if '_egopose' not in os.path.basename(f)],
            key=lambda x: self._extract_frame_number(os.path.basename(x))
        )
        
        point_cloud_seq = []
        ego_pose_seq = []  # stores xyz coordinates only
        file_paths = []
        
        for i in range(self.seq_len):
            actual_idx = start_idx + i * self.frame_step

            if actual_idx >= len(point_cloud_files):
                raise IndexError(f"Index out of bounds for scene {scene_path}")
            
            point_cloud_path = point_cloud_files[actual_idx]
            try:
                point_cloud_data = torch.load(point_cloud_path)
                point_cloud_seq.append(point_cloud_data)
            except Exception as e:
                print(f"Error loading point cloud {point_cloud_path}: {e}")
                raise

            lidar_filename = os.path.basename(point_cloud_path)
            ego_pose_filename = os.path.splitext(lidar_filename)[0] + '_egopose.pt'
            ego_pose_path = os.path.join(scene_path, ego_pose_filename)
            
            try:
                ego_pose_data = torch.load(ego_pose_path)
                # Extract translation (xyz) from pose matrix
                if ego_pose_data.shape == (4, 4):
                    xyz_coords = ego_pose_data[:3, 3]
                elif ego_pose_data.shape == (3, 4):
                    xyz_coords = ego_pose_data[:, 3]
                else:
                    xyz_coords = ego_pose_data
                
                ego_pose_seq.append(xyz_coords)
            except Exception as e:
                print(f"Error loading ego pose {ego_pose_path}: {e}")
                raise

            file_paths.append(f"{sequence_name}/{lidar_filename}")

        point_cloud_seq = torch.cat(point_cloud_seq, dim=0)
        ego_pose_seq = torch.stack(ego_pose_seq, dim=0)

        return point_cloud_seq, ego_pose_seq, file_paths

        
    
    def get_sequence_info(self):
        """Return dataset sequence information."""
        seq_counts = {}
        for scene_path, _ in self.samples:
            seq_name = os.path.basename(scene_path)
            seq_counts[seq_name] = seq_counts.get(seq_name, 0) + 1
        return seq_counts


class CustomDataset(Dataset):
    """
    Each `__getitem__` call returns a sequence of `sequence_length` frames.
    Sequences are guaranteed not to span different splits.
    """
    def __init__(
        self,
        dataroot: str,
        version: str = 'v1.0-trainval',
        sequence_length: int = 6,
        frame_stride: int = 1,
    ):
        """
        Args:
            dataroot: dataset root directory
            version: dataset version (e.g., 'v1.0-trainval', 'v1.0-mini')
            sequence_length: number of consecutive frames (default 6)
            frame_stride: stride between frames (default 1, consecutive frames)
        """
        self.dataroot = Path(dataroot)
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.sequence_length = sequence_length
        self.frame_stride = frame_stride
        # self.sequence_start_indices stores all valid sequence start indices
        self.sequence_start_indices = self._build_sequence_indices()

    def _get_samples_in_scene(self, scene_token: str) -> list:
        scene = self.nusc.get('scene', scene_token)
        samples = []
        current_token = scene['first_sample_token']
        
        while current_token:
            sample = self.nusc.get('sample', current_token)
            samples.append(sample)
            current_token = sample['next']
        
        return samples

    def _build_sequence_indices(self) -> list:
        """Build list of valid sequence start indices (scene_token, start_sample_token)."""
        indices = []
        required_span = (self.sequence_length - 1) * self.frame_stride + 1
        
        for scene in self.nusc.scene:
            samples = self._get_samples_in_scene(scene['token'])
            if len(samples) < required_span:
                continue
                
            for start_idx in range(0, len(samples) - required_span + 1):
                start_token = samples[start_idx]['token']
                indices.append((scene['token'], start_token))
                
        return indices
    
    def __len__(self) -> int:
        return len(self.sequence_start_indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load consecutive frame point cloud sequences and convert to image representations."""
        scene_token, start_token = self.sequence_start_indices[idx]
        sequence_images = []
        
        current_token = start_token
        for _ in range(self.sequence_length):
            sample = self.nusc.get('sample', current_token)
            lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            pc_path = self.dataroot / lidar_data['filename']
            
            # Custom point cloud processing (replaces LidarPointCloud)
            img_representation = load_points_as_images(pc_path)

            sequence_images.append(torch.tensor(img_representation, dtype=torch.float32))

            next_sample_token = sample['next']
            for _ in range(self.frame_stride - 1):
                if next_sample_token: 
                    next_sample = self.nusc.get('sample', next_sample_token)
                    next_sample_token = next_sample['next']
            current_token = next_sample_token if next_sample_token else sample['next']
        
        return torch.stack(sequence_images, dim=0) 
    
def load_points_as_images(
    point_path: str,
    scan_unfolding: bool = True,
    H: int = 32,
    W: int = 1024,
):
    # load xyz & intensity and add depth & mask
    points = np.fromfile(point_path, dtype=np.float32).reshape((-1, 5))
    xyz = points[:, :3]  # xyz
    x = xyz[:, [0]]
    y = xyz[:, [1]]
    z = xyz[:, [2]]
    depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)
    points = depth

    h_up, h_down = np.deg2rad(10), np.deg2rad(-30)
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
    proj_points = np.zeros((H, W, 1), dtype=points.dtype)
    proj_points = scatter(proj_points, grid[order], points[order])
    proj_points = proj_points.transpose(2, 0, 1)
    proj_points = proj_points.astype(np.float32)

    return proj_points

def scatter(array, index, value):
    for (h, w), v in zip(index, value):
        array[h, w] = v
    return array