import os
import glob
import re

components_dir = '/home/marwan/AML/Pixel-Perfect/mario_clone/data/components/'
py_files = glob.glob(os.path.join(components_dir, '*.py'))

for file_path in py_files:
    with open(file_path, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    new_lines = []
    
    skip = False
    for i, line in enumerate(lines):
        if 'def reload_graphics(self, new_gfx):' in line:
            skip = True
            continue
            
        if skip:
            if line.strip() == '' or 'self.sprite_sheet =' in line or 'self.setup_frames()' in line or 'self.load_images_from_sheet()' in line or 'self.image =' in line or '# Mario image' in line:
                # Need to be careful not to skip too much
                if line.strip() == '' and (i + 1 < len(lines) and lines[i+1].strip() == ''):
                    skip = False
                continue
            else:
                skip = False

        if 'self.frames = []' in line and 'def setup_frames' not in line:
            # We injected this right after def setup_frames
            continue
            
        new_lines.append(line)

    with open(file_path, 'w') as f:
        f.write('\n'.join(new_lines))
