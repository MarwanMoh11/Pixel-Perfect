import os
import glob
import re

components_dir = '/home/marwan/AML/Pixel-Perfect/mario_clone/data/components/'
py_files = glob.glob(os.path.join(components_dir, '*.py'))

def inject():
    for file_path in py_files:
        with open(file_path, 'r') as f:
            content = f.read()

        if 'def reload_graphics(self, new_gfx):' in content:
            continue

        lines = content.split('\n')
        new_lines = []
        
        # We need to find class names and their sprite_sheet
        current_class = None
        sprite_sheet_key = None
        has_setup_frames = False
        class_to_sheet = {}
        class_to_setup = {}
        
        for i, line in enumerate(lines):
            class_match = re.match(r'^class (\w+)\(', line)
            if class_match:
                current_class = class_match.group(1)
                class_to_sheet[current_class] = None
                class_to_setup[current_class] = None
                
            sheet_match = re.search(r"self\.sprite_sheet\s*=\s*setup\.GFX\['(.*?)'\]", line)
            if sheet_match and current_class:
                class_to_sheet[current_class] = sheet_match.group(1)

            if re.match(r'^\s*def setup_frames\(self\):', line) and current_class:
                class_to_setup[current_class] = 'setup_frames'
                # inject frames clearance
                new_lines.append(line)
                indent = line.split('def')[0] + '    '
                new_lines.append(indent + 'self.frames = []')
                continue
            
            if re.match(r'^\s*def load_images_from_sheet\(self\):', line) and current_class:
                class_to_setup[current_class] = 'load_images_from_sheet'

            new_lines.append(line)
            
        content = '\n'.join(new_lines)
        
        # Now append reload_graphics to each class
        final_lines = []
        in_class = False
        current_class = None
        
        for i, line in enumerate(content.split('\n')):
            class_match = re.match(r'^class (\w+)\(', line)
            if class_match:
                # If we were in a class, append reload_graphics before starting new class
                if in_class and current_class and class_to_sheet.get(current_class):
                    sheet = class_to_sheet[current_class]
                    setup_func = class_to_setup[current_class]
                    if setup_func:
                        final_lines.append(f'    def reload_graphics(self, new_gfx):')
                        final_lines.append(f'        self.sprite_sheet = new_gfx[\'{sheet}\']')
                        final_lines.append(f'        self.{setup_func}()')
                        if setup_func == 'setup_frames':
                            final_lines.append(f'        self.image = self.frames[self.frame_index]')
                        elif current_class == 'Mario':
                            final_lines.append(f'        # Mario image is updated in animation()')
                        final_lines.append('')
                        final_lines.append('')
                
                current_class = class_match.group(1)
                in_class = True
                
            final_lines.append(line)
            
        # Append for the last class
        if in_class and current_class and class_to_sheet.get(current_class):
            sheet = class_to_sheet[current_class]
            setup_func = class_to_setup.get(current_class)
            if setup_func:
                final_lines.append(f'    def reload_graphics(self, new_gfx):')
                final_lines.append(f'        self.sprite_sheet = new_gfx[\'{sheet}\']')
                final_lines.append(f'        self.{setup_func}()')
                if setup_func == 'setup_frames':
                    final_lines.append(f'        self.image = self.frames[self.frame_index]')
                elif current_class == 'Mario':
                    final_lines.append(f'        # Mario image is updated in animation()')
                final_lines.append('')
                final_lines.append('')

        with open(file_path, 'w') as f:
            f.write('\n'.join(final_lines))

if __name__ == '__main__':
    inject()
