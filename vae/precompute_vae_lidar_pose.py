import os
import numpy as np
import torch

def parse_calibration(filename):
    """Read calibration file."""
    calib = {}
    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]
        
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        
        calib[key] = pose
    
    calib_file.close()
    return calib

def parse_poses(filename, calibration):
    """Read pose file and transform to the LiDAR coordinate system."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Pose file not found: {filename}")
        
    file = open(filename)
    poses = []
    
    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)
    
    for line in file:
        values = [float(v) for v in line.strip().split()]
        
        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0
        
        # Transform to global pose in LiDAR coordinate system
        poses.append(Tr_inv @ (pose @ Tr))
    
    file.close()
    return poses

def save_kitti_egopose(kitti_root, poses_root, output_root):
    """
    Save KITTI odometry global egoproes as .pt files.

    Args:
        kitti_root: KITTI dataset root directory (contains the sequences directory)
        poses_root: Root directory of pose.txt files
        output_root: Output directory
    """
    # all
    # sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

    # train
    sequences = ["00", "01", "02", "03", "04", "05", "06", "07"]

    # test
    # sequences = ["08", "09", "10"]
    
    for sequence in sequences:
        print(f"Processing sequence {sequence}...")
        
        # Create output directory
        seq_output_dir = os.path.join(output_root, "sequences", sequence)
        os.makedirs(seq_output_dir, exist_ok=True)
        
        # Read calibration file (from kitti_root)
        calib_path = os.path.join(kitti_root, "sequences", sequence, "calib.txt")
        if not os.path.exists(calib_path):
            print(f"Calibration file not found: {calib_path}")
            continue
            
        calib = parse_calibration(calib_path)
        
        # Read pose file (from poses_root)
        pose_path = os.path.join(poses_root, f"{sequence}.txt")
        if not os.path.exists(pose_path):
            print(f"Pose file not found: {pose_path}")
            continue
            
        poses = parse_poses(pose_path, calib)
        
        # Read point cloud file name list (from kitti_root)
        velo_dir = os.path.join(kitti_root, "sequences", sequence, "velodyne")
        if not os.path.exists(velo_dir):
            print(f"Velodyne directory not found: {velo_dir}")
            continue
            
        velo_names = sorted(os.listdir(velo_dir))
        
        # Ensure pose count matches point cloud count
        if len(poses) != len(velo_names):
            print(f"Warning: pose count ({len(poses)}) != velodyne count ({len(velo_names)}) in sequence {sequence}")
            min_count = min(len(poses), len(velo_names))
            poses = poses[:min_count]
            velo_names = velo_names[:min_count]
        
        # Save egopose per frame
        for i, (pose, velo_name) in enumerate(zip(poses, velo_names)):
            # Generate output filename: point cloud name + egopose.pt
            output_filename = velo_name.replace('.bin', '_egopose.pt')
            output_path = os.path.join(seq_output_dir, output_filename)
            
            # Convert 4x4 pose matrix to torch tensor and save
            pose_tensor = torch.from_numpy(pose.astype(np.float32))
            torch.save(pose_tensor, output_path)
            
            if i % 100 == 0:
                print(f"  Saved {i+1}/{len(poses)} egoposes for sequence {sequence}")
        
        print(f"Finished processing sequence {sequence}: {len(poses)} egoposes saved")

# Usage example
if __name__ == "__main__":
    # Set paths
    kitti_root = "path/dataset/kitti_odometry/dataset"  # KITTI dataset root directory
    poses_root = "path/dataset/kitti_odometry/dataset/poses"  # pose.txt file directory
    output_root = "path/dataset_processed/kitti_odometry_vae_train"  # Output directory

    # Create output root directory
    os.makedirs(output_root, exist_ok=True)
    
    # Execute save operation
    save_kitti_egopose(kitti_root, poses_root, output_root)
    
    print("All egoposes saved successfully!")