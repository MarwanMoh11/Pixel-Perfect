import os
import argparse
import glob
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import sys

# Ensure src module is discoverable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.esrgan import RRDBNet

def upscale_image(img_path, output_path, model, device):
    """
    Upscale a single image (handling RGBA)
    """
    img = Image.open(img_path)
    
    has_alpha = False
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        img = img.convert('RGBA')
        rgb = img.convert('RGB')
        alpha = img.split()[-1]
        has_alpha = True
    else:
        img = img.convert('RGB')
        rgb = img
        alpha = None

    try:
        # Convert RGB to tensor [1, 3, H, W], normalized to [0, 1]
        rgb_tensor = TF.to_tensor(rgb).unsqueeze(0).to(device)

        # Forward pass
        with torch.no_grad():
            out_tensor = model(rgb_tensor)
        
        # Post-process to image
        out_tensor = out_tensor.squeeze(0).cpu().clamp(0, 1)
        out_rgb = TF.to_pil_image(out_tensor)
    except torch.cuda.OutOfMemoryError:
        print("    CUDA OOM: Falling back to Nearest Neighbor for this image.")
        # Free memory
        torch.cuda.empty_cache()
        # Fallback to simple nearest neighbor scaling for RGB
        out_rgb = rgb.resize((rgb.width * 4, rgb.height * 4), Image.NEAREST)

    # Handle Alpha by 4x nearest neighbor scaling
    if has_alpha:
        w, h = alpha.size
        # ESRGAN scales by 4x
        out_alpha = alpha.resize((w * 4, h * 4), Image.NEAREST)
        out_rgb.putalpha(out_alpha)
    
    # Save image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_rgb.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help="Path to input graphics folder")
    parser.add_argument('--output_dir', required=True, help="Path to save upscaled graphics")
    parser.add_argument('--checkpoint', required=True, help="Path to trained RRDBNet checkpoint (.pth)")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=8, gc=32)
    state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    # The saved checkpoint might be nested in 'generator' if it was a full training state, 
    # but the user said they provided RRDBNet_epoch_20.pth which is probably just the model.
    if 'generator' in state_dict:
        model.load_state_dict(state_dict['generator'])
    else:
        model.load_state_dict(state_dict)

    model.to(args.device)
    model.eval()

    print(f"Upscaling assets from {args.input_dir} to {args.output_dir}...")
    
    # Walk through all directories
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith('.png'):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(input_path, args.input_dir)
                output_path = os.path.join(args.output_dir, rel_path)
                
                print(f"Processing {rel_path}...")
                try:
                    upscale_image(input_path, output_path, model, args.device)
                except Exception as e:
                    print(f"Failed to process {rel_path}: {e}")

    print("Done!")

if __name__ == '__main__':
    main()
