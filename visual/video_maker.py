import imageio
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import glob

def create_video_from_folders():
    base_dir = "./for_video_gen_long"
    
    if not os.path.exists(base_dir):
        print(f"Directory does not exist: {base_dir}")
        return
    
    original_depth_files = sorted(glob.glob(os.path.join(base_dir, "original_depth_*.png")))
    original_rendered_files = sorted(glob.glob(os.path.join(base_dir, "original_rendered_*.png")))
    recon_depth_files = sorted(glob.glob(os.path.join(base_dir, "recon_depth_*.png")))
    recon_rendered_files = sorted(glob.glob(os.path.join(base_dir, "recon_rendered_*.png")))
    
    # Use the shortest file count to ensure consistent image counts across all types
    min_count = min(len(original_depth_files), len(original_rendered_files), 
                   len(recon_depth_files), len(recon_rendered_files))
    
    print(f"Found {min_count} image sets")
    
    scene_data = {
        'original_depth': [],
        'original_rendered': [],
        'recon_depth': [],
        'recon_rendered': []
    }
    
    timestamps = []
    current_time = -2.4
    for i in range(min_count):
        timestamps.append(f"{current_time:.1f}s")
        current_time += 0.6
    
    for i in range(min_count):
        depth_img = np.array(Image.open(original_depth_files[i]))
        rendered_img = np.array(Image.open(original_rendered_files[i]))
        
        if len(depth_img.shape) == 2:  # grayscale
            depth_img = np.stack([depth_img, depth_img, depth_img], axis=2)
        if len(rendered_img.shape) == 2:  # grayscale
            rendered_img = np.stack([rendered_img, rendered_img, rendered_img], axis=2)
        
        if depth_img.shape[2] == 4:
            depth_img = depth_img[:, :, :3]
        if rendered_img.shape[2] == 4:
            rendered_img = rendered_img[:, :, :3]
        
        # Add timestamp to original rendered image
        rendered_pil = Image.fromarray(rendered_img)
        draw = ImageDraw.Draw(rendered_pil)
        
        font = None
        font_names = [
            "times.ttf", 
            "Times New Roman.ttf", 
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux path
            "/System/Library/Fonts/Times.ttc",  # macOS path
            "C:/Windows/Fonts/times.ttf",  # Windows path
            "C:/Windows/Fonts/timesbd.ttf",  # Windows bold path
        ]
        
        for font_path in font_names:
            try:
                font = ImageFont.truetype(font_path, size=48)
                print(f"Successfully loaded font: {font_path}")
                break
            except:
                continue
        
        # Use default font if all preset paths fail
        if font is None:
            font = ImageFont.load_default()
            print("Using default font")
        
        bbox = draw.textbbox((0, 0), timestamps[i], font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Add timestamp in the top-right corner (20px margin from right and top)
        text_x = rendered_pil.width - text_width - 20
        text_y = 20
        
        draw.text((text_x, text_y), timestamps[i], fill=(0, 0, 0), font=font)
        
        rendered_img_with_timestamp = np.array(rendered_pil)
        
        depth_resized = np.array(Image.fromarray(depth_img).resize((rendered_img_with_timestamp.shape[1], depth_img.shape[0]), Image.LANCZOS))
        
        scene_data['original_depth'].append(depth_resized)
        scene_data['original_rendered'].append(rendered_img_with_timestamp)
        
        recon_depth_img = np.array(Image.open(recon_depth_files[i]))
        recon_rendered_img = np.array(Image.open(recon_rendered_files[i]))
        
        if len(recon_depth_img.shape) == 2:  # grayscale
            recon_depth_img = np.stack([recon_depth_img, recon_depth_img, recon_depth_img], axis=2)
        if len(recon_rendered_img.shape) == 2:  # grayscale
            recon_rendered_img = np.stack([recon_rendered_img, recon_rendered_img, recon_rendered_img], axis=2)
        
        if recon_depth_img.shape[2] == 4:
            recon_depth_img = recon_depth_img[:, :, :3]
        if recon_rendered_img.shape[2] == 4:
            recon_rendered_img = recon_rendered_img[:, :, :3]
        
        # Add the same timestamp to reconstructed rendered image
        recon_rendered_pil = Image.fromarray(recon_rendered_img)
        draw_recon = ImageDraw.Draw(recon_rendered_pil)
        
        font_recon = None
        for font_path in font_names:
            try:
                font_recon = ImageFont.truetype(font_path, size=48)
                print(f"Successfully loaded font for recon: {font_path}")
                break
            except:
                continue
        
        # Use default font if all preset paths fail
        if font_recon is None:
            font_recon = ImageFont.load_default()
            print("Using default font for recon")
        
        bbox_recon = draw_recon.textbbox((0, 0), timestamps[i], font=font_recon)
        text_width_recon = bbox_recon[2] - bbox_recon[0]
        text_height_recon = bbox_recon[3] - bbox_recon[1]
        
        text_x_recon = recon_rendered_pil.width - text_width_recon - 20
        text_y_recon = 20
        
        draw_recon.text((text_x_recon, text_y_recon), timestamps[i], fill=(0, 0, 0), font=font_recon)
        
        recon_rendered_img_with_timestamp = np.array(recon_rendered_pil)
        
        # Resize reconstructed depth image to match reconstructed rendered image width
        recon_depth_resized = np.array(Image.fromarray(recon_depth_img).resize((recon_rendered_img_with_timestamp.shape[1], recon_depth_img.shape[0]), Image.LANCZOS))
        
        scene_data['recon_depth'].append(recon_depth_resized)
        scene_data['recon_rendered'].append(recon_rendered_img_with_timestamp)
    
    max_frames = len(scene_data['original_rendered'])
    print(f"Creating video with {max_frames} frames")
    
    writer = imageio.get_writer("samples.mp4", mode="I", fps=5)
    
    for frame_idx in range(max_frames):
        original_depth = scene_data['original_depth'][frame_idx]
        original_rendered = scene_data['original_rendered'][frame_idx]
        recon_depth = scene_data['recon_depth'][frame_idx]
        recon_rendered = scene_data['recon_rendered'][frame_idx]
        
        # Stack depth image above rendered image to create original and reconstructed composites
        original_combined = np.vstack([original_depth, original_rendered])
        recon_combined = np.vstack([recon_depth, recon_rendered])
        
        assert len(original_combined.shape) == 3 and original_combined.shape[2] == 3, f"Image shape error: {original_combined.shape}"
        assert len(recon_combined.shape) == 3 and recon_combined.shape[2] == 3, f"Image shape error: {recon_combined.shape}"
        
        # Calculate target dimensions (use maximum values)
        target_height = max(original_combined.shape[0], recon_combined.shape[0])
        target_width = max(original_combined.shape[1], recon_combined.shape[1])
        
        # Resize both composite images to the same dimensions
        original_resized = np.array(Image.fromarray(original_combined).resize((target_width, target_height)))
        recon_resized = np.array(Image.fromarray(recon_combined).resize((target_width, target_height)))
        
        # Create separator (2-pixel wide white line)
        separator = np.ones((target_height, 2, 3), dtype=np.uint8) * 255
        
        # Horizontally concatenate with separator
        combined_frame = np.hstack([original_resized, separator, recon_resized])

        writer.append_data(combined_frame)
    
    writer.close()
    print("Video created successfully!")

if __name__ == "__main__":
    create_video_from_folders()
