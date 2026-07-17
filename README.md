# GEM: Generating LiDAR World Model via Deformable Mamba

**project page**: [https://wuyang98.github.io/GEM/](https://wuyang98.github.io/GEM/)

![64 line reconstruction and generation](https://raw.githubusercontent.com/wuyang98/gem/main/visual/page56_1.gif)

## 1. Environment Setup

```bash
conda env create -f environment.yaml
conda activate gem
```

> **Note:** `mamba-ssm` may require CUDA 12.1 and a compatible GPU. Make sure your CUDA driver version matches before installing.

---

## 2. Dataset Preparation

This project uses the [KITTI Odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). Download the velodyne laser data and calibration files and organize them as:

```
kitti_odometry/dataset/
├── sequences/
│   ├── 00/
│   │   ├── velodyne/       # raw .bin point cloud files
│   │   └── calib.txt
│   ├── 01/
│   ...
└── poses/
    ├── 00.txt
    ├── 01.txt
    ...
```

---

## 3. Train VAE

Edit the dataset path in `vae/vae_v17_kitti.py`:

```python
root_configs = {
    "path/kitti_odometry/dataset/sequences": {
        "TRAIN": [0, 1, 2, 3, 4, 5, 6, 7],
        ...
    }
}
```

Then launch distributed training:

```bash
torchrun --nproc_per_node=<NUM_GPUS> vae/vae_v17_kitti.py
```

Checkpoints are saved to `vae/checkpoints/` and visualizations to `vae/fig/`.

Alternatively, download the pretrained VAE checkpoint: [vae_epoch_80.pth](https://huggingface.co/fengchen7/gem/blob/main/vae_epoch_80.pth)

---

## 4. Precompute Dataset

### 4.1 Precompute VAE Latents

Edit the paths in `vae/precompute_vae_onlylidar.py`:

```python
kitti_root = "path/dataset/kitti_odometry/dataset"
output_root = "path/dataset_processed/kitti_odometry_vae_train"
```

Then run:

```bash
python vae/precompute_vae_onlylidar.py
```

This encodes each LiDAR frame into a VAE latent and saves it as a `.pt` file.

### 4.2 Precompute Ego Poses

Edit the paths in `vae/precompute_vae_lidar_pose.py`:

```python
kitti_root = "path/dataset/kitti_odometry/dataset"
poses_root = "path/dataset/kitti_odometry/dataset/poses"
output_root = "path/dataset_processed/kitti_odometry_vae_train"
```

Then run:

```bash
python vae/precompute_vae_lidar_pose.py
```

This saves per-frame ego pose matrices as `*_egopose.pt` files alongside the latents.

After both steps, the processed dataset directory should look like:

```
kitti_odometry_vae_train/
└── sequences/
    ├── 08/
    │   ├── 000000.pt
    │   ├── 000000_egopose.pt
    │   ├── 000001.pt
    │   ├── 000001_egopose.pt
    │   ...
```

---

## 5. Train Diffusion Model

Edit the dataset path in `train.py`:

```python
root_dirs = ["path/dataset_processed/kitti_odometry_vae_train"]
```

Also set the VAE checkpoint path:

```python
model_path = "vae/checkpoints/vae_epoch_80.pth"
```

Then run:

```bash
accelerate launch train.py
```

Checkpoints and TensorBoard logs are saved to `logs/diffusion/kitti/spherical-1024/<timestamp>/`.

Alternatively, download the pretrained diffusion checkpoint: [diffusion_0001200000.pth](https://huggingface.co/fengchen7/gem/blob/main/diffusion_0001200000.pth)

---

## 6. Generate Test Samples

Edit the dataset path in `generate_and_save.py`:

```python
root_dirs = ["path/dataset_processed/kitti_odometry_vae_train"]
```

Then run:

```bash
python generate_and_save.py \
    --ckpt logs/diffusion/kitti/spherical-1024/<timestamp>/models/diffusion_0001200000.pth \
    --output_dir ./results \
    --num_samples 6701 \
    --sampling_steps 256 \
    --seed 88
```

Generated point clouds are saved as `.bin` files under `--output_dir` with the structure:

```
results/
├── 0000_08/
│   ├── 000036.bin
│   ├── 000042.bin
│   ...
├── 0001_08/
...
```

---

## 7. Prepare Ground Truth Point Clouds

Edit the paths in `vae/precompute_lidar.py`:

```python
kitti_root = "path/dataset/kitti_odometry/dataset"
output_root = "path/dataset_processed/kitti_odometry_gt_test"
```

Then run:

```bash
python vae/precompute_lidar.py
```

This converts raw KITTI LiDAR frames to the same xyz `.bin` format as the generated samples, for fair comparison.

---

## 8. Evaluate

```bash
python test_multiframe_kitti.py \
    --pred-dir ./results \
    --gt-dir path/dataset_processed/kitti_odometry_gt_test \
    --output ./results_kitti.txt
```

Metrics reported:
- **Chamfer Distance** (CD)
- **Chamfer Distance Inner** (CD-inner)
- **L1 Error** (mean and median)
- **AbsRel Error** (mean and median)

This project is undergoing internal business review by the relevant department. Where possible, I will continue to open-source the non-commercial parts. Currently, the 64-line content has already been open-sourced.
If you are interested, you can contact me through e-mail for more information.

---

## Acknowledgements

This work builds upon the following open-source projects:

- [R2DM](https://github.com/kazuto1011/r2dm)
- [LiDM](https://github.com/hancyran/LiDAR-Diffusion)
- [4D-Occ](https://github.com/tarashakhurana/4d-occ-forecasting)

We sincerely thank the authors for making their code publicly available.

---

## Citation

If you find it helpful, we would appreciate it if you could cite our work involved in this project.

```bibtex
@inproceedings{wu2026gem,
  title={GEM: Generating LiDAR World Model via Deformable Mamba},
  author={Wu, Yang and Liu, Zhaojiang and Meng, Qiang and Liu, Youquan and Weng, Renliang and Qian, Jianjun and Yang, Jian and Xie, Jin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={24227--24236},
  year={2026}
}

@inproceedings{wu2025weathergen,
  title={WeatherGen: A unified diverse weather generator for LiDAR point clouds via spider mamba diffusion},
  author={Wu, Yang and Zhu, Yun and Zhang, Kaihua and Qian, Jianjun and Xie, Jin and Yang, Jian},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={17019--17028},
  year={2025}
}
```
