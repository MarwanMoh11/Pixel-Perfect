import os
import glob
from PIL import Image

def fix_colorkeys():
    orig_dir = '/home/marwan/AML/Pixel-Perfect/mario_clone/resources/graphics'
    upscaled_dir = '/home/marwan/AML/Pixel-Perfect/mario_clone/resources/graphics_upscaled'
    fixed_dir = '/home/marwan/AML/Pixel-Perfect/mario_clone/resources/graphics_fixed'
    
    os.makedirs(fixed_dir, exist_ok=True)
    
    # We will identify the background color for each image by looking at pixel (0,0) of the original
    # For text_images.png, it's a blue. For most others it's black or magenta.
    # We will force any pixel in the upscaled image that corresponds to the original background color
    # to be EXACTLY the background color.

    for orig_path in glob.glob(os.path.join(orig_dir, '*.png')):
        filename = os.path.basename(orig_path)
        upscaled_path = os.path.join(upscaled_dir, filename)
        fixed_path = os.path.join(fixed_dir, filename)
        
        if not os.path.exists(upscaled_path):
            print(f"Skipping {filename}, no upscaled version.")
            # Just copy the original if missing (like level_1.png if it OOM'd)
            os.system(f"cp {orig_path} {fixed_path}")
            continue
            
        print(f"Fixing colorkeys for {filename}...")
        orig_img = Image.open(orig_path).convert('RGB')
        up_img = Image.open(upscaled_path).convert('RGB')
        
        # Determine background color (assume top-left pixel)
        bg_color = orig_img.getpixel((0, 0))
        
        orig_pixels = orig_img.load()
        up_pixels = up_img.load()
        
        w, h = orig_img.size
        
        for y in range(h):
            for x in range(w):
                if orig_pixels[x, y] == bg_color:
                    # Force the 4x4 block to exact bg color
                    for dy in range(4):
                        for dx in range(4):
                            up_pixels[x*4 + dx, y*4 + dy] = bg_color
                            
        # Save fixed image
        up_img.save(fixed_path)
        
    print("Colorkeys fixed!")

if __name__ == '__main__':
    fix_colorkeys()
