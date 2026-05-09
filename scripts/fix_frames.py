import os
import glob
import re

components_dir = '/home/marwan/AML/Pixel-Perfect/mario_clone/data/components/'
py_files = glob.glob(os.path.join(components_dir, '*.py'))

for file_path in py_files:
    if file_path.endswith('mario.py') or file_path.endswith('__init__.py'):
        continue
        
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    new_lines = []
    
    in_init = False
    added_frames = False
    
    for line in lines:
        if 'def __init__' in line:
            in_init = True
            added_frames = False
            new_lines.append(line)
            continue
            
        if in_init:
            if 'self.frames = []' in line:
                added_frames = True
            
            # If we hit an empty line or end of init (another def or lower indent)
            if 'self.sprite_sheet =' in line and not added_frames:
                new_lines.append(line)
                indent = line.split('self')[0]
                new_lines.append(indent + 'self.frames = []\n')
                added_frames = True
                continue
                
            if 'self.setup_frames()' in line or 'self.create_frames()' in line:
                if not added_frames:
                    indent = line.split('self')[0]
                    new_lines.append(indent + 'self.frames = []\n')
                    added_frames = True
                    
            if line.strip().startswith('def ') and 'def __init__' not in line:
                in_init = False
                
        new_lines.append(line)

    with open(file_path, 'w') as f:
        f.writelines(new_lines)
