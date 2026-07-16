import torch
import numpy as np
import argparse
import os
import glob
from evaluation_utils import compute_chamfer_distance, compute_chamfer_distance_inner, compute_ray_errors

def load_point_cloud(bin_path):
    """
    Load point cloud data from a .bin file.
    Assumes each point consists of x, y, z, intensity, ring (5 floats of 4 bytes each),
    or pure xyz data (3 floats). Only the first 3 dimensions (x, y, z) are returned.
    """
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Point cloud file not found: {bin_path}")

    points = np.fromfile(bin_path, dtype=np.float32)

    if points.size % 5 == 0:
        points = points.reshape((-1, 5))
        points = points[:, :3]  # keep xyz only
    elif points.size % 3 == 0:
        points = points.reshape((-1, 3))
    else:
        raise ValueError(f"Point cloud file {bin_path} size ({points.size}) is not a multiple of 3 or 5. Data format might be incorrect.")

    return points

def find_matching_files(pred_dir, gt_dir):
    """
    Find matching .bin file pairs between pred and gt directories.

    pred_dir structure:
        pred_dir/0000_08/000440.bin

    gt_dir structure:
        gt_dir/sequences/08/000440.bin
    """
    matched_pairs = []

    pred_seq_dirs = glob.glob(os.path.join(pred_dir, "*_*"))

    for pred_seq_dir in pred_seq_dirs:
        seq_name_with_prefix = os.path.basename(pred_seq_dir)  # e.g., "0000_08"
        parts = seq_name_with_prefix.split('_')
        if len(parts) != 2:
            print(f"Warning: Skipping invalid pred sequence folder name: {seq_name_with_prefix}")
            continue
        _, seq_number = parts  # e.g., "08"

        gt_seq_dir = os.path.join(gt_dir, "sequences", seq_number)

        if not os.path.exists(gt_seq_dir):
            print(f"Warning: GT sequence directory not found: {gt_seq_dir}. Skipping.")
            continue

        pred_files = glob.glob(os.path.join(pred_seq_dir, "*.bin"))

        for pred_file in pred_files:
            filename = os.path.basename(pred_file)  # e.g., "000440.bin"
            gt_file = os.path.join(gt_seq_dir, filename)

            if os.path.exists(gt_file):
                matched_pairs.append((pred_file, gt_file))
            else:
                print(f"Warning: GT file not found for prediction {pred_file}. Skipping.")

    if not matched_pairs:
        raise ValueError("No matching file pairs found between prediction and ground truth directories.")

    matched_pairs.sort()
    return matched_pairs

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.origin:
        origin = np.array(args.origin, dtype=np.float32)
    else:
        print("Warning: Using default origin [0.0, 0.0, 0.0]. Please verify this is correct for your data.")
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    origin_tensor = torch.from_numpy(origin)
    print(f"Using sensor origin: {origin}")

    try:
        print(f"Searching for matching point cloud files...")
        matched_pairs = find_matching_files(args.pred_dir, args.gt_dir)
        print(f"Found {len(matched_pairs)} matching file pairs.")

        metrics = {
            "count": 0,
            "chamfer_distance": 0.0,
            "chamfer_distance_inner": 0.0,
            "l1_error": 0.0,
            "absrel_error": 0.0
        }
        individual_errors = {
            "l1_errors": [],
            "absrel_errors": []
        }

        for i, (pred_path, gt_path) in enumerate(matched_pairs):
            print(f"Processing pair {i+1}/{len(matched_pairs)}: {pred_path} <-> {gt_path}")

            try:
                pred_points = load_point_cloud(pred_path)
                gt_points = load_point_cloud(gt_path)

                pred_tensor = torch.from_numpy(pred_points)
                gt_tensor = torch.from_numpy(gt_points)

                cd = compute_chamfer_distance(pred_tensor, gt_tensor, device)
                cd_inner = compute_chamfer_distance_inner(pred_tensor, gt_tensor, device)
                l1_error, absrel_error = compute_ray_errors(pred_tensor, gt_tensor, origin_tensor, device)

                metrics["count"] += 1
                metrics["chamfer_distance"] += cd
                metrics["chamfer_distance_inner"] += cd_inner
                metrics["l1_error"] += l1_error
                metrics["absrel_error"] += absrel_error

                individual_errors["l1_errors"].append(l1_error)
                individual_errors["absrel_errors"].append(absrel_error)

                print(f"  -> CD: {cd:.6f}, CD_inner: {cd_inner:.6f}, L1: {l1_error:.6f}, AbsRel: {absrel_error:.6f}")

            except Exception as e:
                print(f"Error processing pair ({pred_path}, {gt_path}): {e}")

        if metrics["count"] > 0:
            avg_cd = metrics["chamfer_distance"] / metrics["count"]
            avg_cd_inner = metrics["chamfer_distance_inner"] / metrics["count"]
            avg_l1 = metrics["l1_error"] / metrics["count"]
            avg_absrel = metrics["absrel_error"] / metrics["count"]

            l1_errors_array = np.array(individual_errors["l1_errors"])
            absrel_errors_array = np.array(individual_errors["absrel_errors"])

            median_l1 = np.median(l1_errors_array)
            median_absrel = np.median(absrel_errors_array)

            print("\n" + "="*72)
            print("Final Average Metrics:")
            print("="*72)
            print(f"Total Pairs Evaluated: {metrics['count']}")
            print("-" * 33)
            print(f"Average Chamfer Distance:     {avg_cd:.6f}")
            print(f"Average Chamfer Distance Inner: {avg_cd_inner:.6f}")
            print(f"Average L1 Error:             {avg_l1:.6f}")
            print(f"Average AbsRel Error:         {avg_absrel:.6f}")
            print("="*72)

            print("\n" + "="*72)
            print("Final Median Metrics:")
            print("="*72)
            print(f"Total Pairs Evaluated: {metrics['count']}")
            print("-" * 33)
            print(f"Median L1 Error:              {median_l1:.6f}")
            print(f"Median AbsRel Error:          {median_absrel:.6f}")
            print("="*72)

            if args.output:
                with open(args.output, 'a') as f:
                    f.write("Point Cloud Evaluation Results (Averaged)\n")
                    f.write("="*50 + "\n")
                    f.write(f"Pred Dir: {args.pred_dir}\n")
                    f.write(f"GT Dir: {args.gt_dir}\n")
                    f.write(f"Origin: {origin.tolist()}\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Total Pairs Evaluated: {metrics['count']}\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"Average Chamfer Distance:     {avg_cd:.6f}\n")
                    f.write(f"Average Chamfer Distance Inner: {avg_cd_inner:.6f}\n")
                    f.write(f"Average L1 Error:             {avg_l1:.6f}\n")
                    f.write(f"Average AbsRel Error:         {avg_absrel:.6f}\n")
                    f.write(f"Median L1 Error:              {median_l1:.6f}\n")
                    f.write(f"Median AbsRel Error:          {median_absrel:.6f}\n")
                    f.write("*"*97 + "\n")
                print(f"\nResults saved to {args.output}")

        else:
            print("No pairs were successfully evaluated.")

    except Exception as e:
        print(f"An error occurred during folder evaluation: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate point clouds in folders using standard metrics.")
    
    parser.add_argument("--pred-dir", type=str, default='path/results/kitti_odometry_3',
                        help="Path to the directory containing predicted point cloud .bin files")
    parser.add_argument("--gt-dir", type=str, default='path/dataset_processed/kitti_odometry_gt_test',
                        help="Path to the directory containing ground truth point cloud .bin files") 
    parser.add_argument("--origin", type=float, nargs=3, metavar=('X', 'Y', 'Z'), 
                        default=[0.0, 0.0, 0.0],
                        help="Sensor origin coordinates (x, y, z). Default is [0.0, 0.0, 0.0].")
    parser.add_argument("--output", type=str, default="./results_kitti.txt",
                        help="Path to save the final averaged results to a text file.")

    args = parser.parse_args()
    main(args)