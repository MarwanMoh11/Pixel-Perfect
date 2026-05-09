"""
Refactor the Mario game engine to use 4x upscaled sprite sheets.

This script is IDEMPOTENT — it can be safely run multiple times.
It first strips any previous refactoring, then applies the 4x coordinate
mapping exactly once, and injects near-black pixel cleanup to fix
ESRGAN colorkey artifacts.
"""
import os
import glob
import re


def refactor_get_image():
    components_dir = '/home/marwan/AML/Pixel-Perfect/mario_clone/data/components/'
    py_files = glob.glob(os.path.join(components_dir, '*.py'))

    for file_path in py_files:
        with open(file_path, 'r') as f:
            content = f.read()

        # --- Phase 1: STRIP any previous refactoring ---
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped == 'orig_w, orig_h = width, height':
                continue
            if stripped == 'x, y, width, height = x*4, y*4, width*4, height*4':
                continue
            # Remove any previously injected import
            if stripped == 'from .. import tools as _tools':
                continue
            # Remove any previously injected cleanup calls
            if '_tools.clean_near_black(image)' in stripped or '_tools.clean_near_color(image' in stripped:
                continue
            # Revert SRCALPHA
            if 'pg.SRCALPHA).convert_alpha()' in line:
                line = line.replace('pg.SRCALPHA).convert_alpha()', 'convert()')
            # Revert smoothscale
            if 'pg.transform.smoothscale(' in line:
                line = line.replace('pg.transform.smoothscale(', 'pg.transform.scale(')
            
            # Revert orig_w/orig_h back to rect.width/rect.height
            if 'orig_w*c.SIZE_MULTIPLIER' in line or 'orig_h*c.SIZE_MULTIPLIER' in line:
                line = line.replace('orig_w*c.SIZE_MULTIPLIER', 'rect.width*c.SIZE_MULTIPLIER')
                line = line.replace('orig_h*c.SIZE_MULTIPLIER', 'rect.height*c.SIZE_MULTIPLIER')
            if 'orig_w*c.BRICK_SIZE_MULTIPLIER' in line or 'orig_h*c.BRICK_SIZE_MULTIPLIER' in line:
                line = line.replace('orig_w*c.BRICK_SIZE_MULTIPLIER', 'rect.width*c.BRICK_SIZE_MULTIPLIER')
                line = line.replace('orig_h*c.BRICK_SIZE_MULTIPLIER', 'rect.height*c.BRICK_SIZE_MULTIPLIER')
            if 'orig_w*2.9' in line or 'orig_h*2.9' in line:
                line = line.replace('orig_w*2.9', 'rect.width*2.9')
                line = line.replace('orig_h*2.9', 'rect.height*2.9')
            if 'orig_w*3' in line or 'orig_h*3' in line:
                line = line.replace('orig_w*3', 'rect.width*3')
                line = line.replace('orig_h*3', 'rect.height*3')
            cleaned_lines.append(line)
        content = '\n'.join(cleaned_lines)

        # --- Phase 2: Ensure tools import exists ---
        # Check if the file already imports tools
        has_tools_import = 'from .. import tools' in content or 'from .. import setup, tools' in content
        if not has_tools_import:
            # Add import after the last 'from ..' import line
            import_lines = content.split('\n')
            insert_idx = 0
            for i, line in enumerate(import_lines):
                if line.startswith('from ..'):
                    insert_idx = i + 1
            import_lines.insert(insert_idx, 'from .. import tools as _tools')
            content = '\n'.join(import_lines)
            tools_ref = '_tools'
        else:
            tools_ref = 'tools'

        # --- Phase 3: APPLY the 4x refactoring + cleanup exactly once ---
        new_content = []
        in_get_image = False
        for line in content.split('\n'):
            if re.match(r'^\s*def get_image\(self,\s*x,\s*y,\s*width,\s*height\):', line):
                new_content.append(line)
                indent = line.split('def')[0] + '    '
                new_content.append(indent + 'orig_w, orig_h = width, height')
                new_content.append(indent + 'x, y, width, height = x*4, y*4, width*4, height*4')
                in_get_image = True
                continue

            if in_get_image:
                # Replace rect.width/height with orig_w/orig_h in scale calls
                if 'rect.width*c.SIZE_MULTIPLIER' in line or 'rect.height*c.SIZE_MULTIPLIER' in line:
                    line = line.replace('rect.width*c.SIZE_MULTIPLIER', 'orig_w*c.SIZE_MULTIPLIER')
                    line = line.replace('rect.height*c.SIZE_MULTIPLIER', 'orig_h*c.SIZE_MULTIPLIER')
                if 'rect.width*c.BRICK_SIZE_MULTIPLIER' in line or 'rect.height*c.BRICK_SIZE_MULTIPLIER' in line:
                    line = line.replace('rect.width*c.BRICK_SIZE_MULTIPLIER', 'orig_w*c.BRICK_SIZE_MULTIPLIER')
                    line = line.replace('rect.height*c.BRICK_SIZE_MULTIPLIER', 'orig_h*c.BRICK_SIZE_MULTIPLIER')
                if 'rect.width*2.9' in line or 'rect.height*2.9' in line:
                    line = line.replace('rect.width*2.9', 'orig_w*2.9')
                    line = line.replace('rect.height*2.9', 'orig_h*2.9')
                if 'rect.width*3' in line or 'rect.height*3' in line:
                    line = line.replace('rect.width*3', 'orig_w*3')
                    line = line.replace('rect.height*3', 'orig_h*3')

                # Replace surface creation with SRCALPHA
                if 'image = pg.Surface([width, height]).convert()' in line:
                    line = line.replace('image = pg.Surface([width, height]).convert()', 'image = pg.Surface([width, height], pg.SRCALPHA).convert_alpha()')
                elif 'image = pg.Surface([width, height])' in line and 'convert' not in line:
                    line = line.replace('image = pg.Surface([width, height])', 'image = pg.Surface([width, height], pg.SRCALPHA).convert_alpha()')

                # Use smoothscale for high-quality downsampling!
                if 'pg.transform.scale(' in line:
                    line = line.replace('pg.transform.scale(', 'pg.transform.smoothscale(')

                # Note: We no longer inject clean_near_black because SRCALPHA natively
                # preserves the transparent backgrounds and anti-aliased halos without
                # letting uninitialized Pygame memory cause static rainbow noise!
                
                # If we encounter another def, we are out of get_image
                if re.match(r'^\s*def ', line) and 'get_image' not in line:
                    in_get_image = False

            new_content.append(line)

        with open(file_path, 'w') as f:
            f.write('\n'.join(new_content))

    print("Refactoring complete (idempotent).")


if __name__ == '__main__':
    refactor_get_image()
