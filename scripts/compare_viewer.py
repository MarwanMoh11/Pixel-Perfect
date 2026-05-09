#!/usr/bin/env python3
"""
Side-by-side comparison: True Original (1x) vs Nearest-Neighbor (4x) vs AI-Upscaled (ESRGAN 4x).

Shows three versions side-by-side:
1. Small, centered view representing original NES resolution
2. Nearest-Neighbor upscaled to modern screen height
3. AI-Upscaled (ESRGAN) to modern screen height

Usage:
    python3 scripts/compare_viewer.py

Controls:
    LEFT/RIGHT arrows  - scroll through the level
    UP/DOWN arrows     - zoom in/out
    F                  - toggle fullscreen
    ESC                - quit
"""

import sys
import os
import pygame as pg
import numpy as np

MARIO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mario_clone'))
RESOURCES = os.path.join(MARIO_DIR, 'resources')
# Both 1x and Nearest panels use the original (which are pre-scaled 4x nearest-neighbor)
NEAREST_GFX_DIR = os.path.join(RESOURCES, 'graphics_original')
UPSCALED_GFX_DIR = os.path.join(RESOURCES, 'graphics')

if not os.path.isdir(NEAREST_GFX_DIR):
    print("ERROR: 'resources/graphics_original' not found.")
    sys.exit(1)


# ─── Helper: clean near-black pixels ────────────────────────────────────
def clean_near_black(surface, threshold=20):
    """Snap near-black pixels to pure black for clean colorkey.
    Keep threshold low (20) to avoid eroding dark brown/red sprite outlines.
    """
    arr = pg.surfarray.pixels3d(surface)
    channel_sum = arr.astype(np.uint16).sum(axis=2)
    mask = channel_sum < threshold
    arr[mask] = 0
    del arr
    return surface


def clean_near_color(surface, color, tolerance=20):
    """Snap pixels near a target color to that exact color."""
    arr = pg.surfarray.pixels3d(surface)
    target = np.array(color[:3], dtype=np.int16)
    diff = np.abs(arr.astype(np.int16) - target).sum(axis=2)
    mask = diff < tolerance
    arr[mask] = target.astype(np.uint8)
    del arr
    return surface


# ─── Load all graphics from a directory ─────────────────────────────────
def load_gfx(directory):
    graphics = {}
    for pic in os.listdir(directory):
        name, ext = os.path.splitext(pic)
        if ext.lower() in ('.png', '.jpg', '.bmp'):
            img = pg.image.load(os.path.join(directory, pic))
            if img.get_alpha():
                img = img.convert_alpha()
            else:
                img = img.convert()
            graphics[name] = img
    return graphics


# ─── Extract a sprite from a 4x sprite sheet ───────────────────────────
def get_sprite(sheet, nes_x, nes_y, nes_w, nes_h, scale, colorkey=(0, 0, 0),
               is_esrgan=False):
    """Extract sprite using NES-scale coordinates from a 4x sheet."""
    x, y, w, h = nes_x * 4, nes_y * 4, nes_w * 4, nes_h * 4
    
    # Use convert_alpha to preserve transparency natively for ESRGAN
    image = pg.Surface([w, h], pg.SRCALPHA).convert_alpha()
    image.blit(sheet, (0, 0), (x, y, w, h))

    if not is_esrgan:
        # Original graphics don't have alpha channels and have weird background colors
        # (like brown (156, 74, 0) or dark brown (19, 11, 2)).
        # We manually key these out for the viewer demo.
        arr = pg.surfarray.pixels3d(image)
        alpha = pg.surfarray.pixels_alpha(image)
        
        # Color key 1: Black
        mask_black = (arr[:, :, 0] == 0) & (arr[:, :, 1] == 0) & (arr[:, :, 2] == 0)
        # Color key 2: Dark Brown around Mushroom
        mask_dbrown = (arr[:, :, 0] == 19) & (arr[:, :, 1] == 11) & (arr[:, :, 2] == 2)
        # Color key 3: Light Brown around Brick
        mask_lbrown = (arr[:, :, 0] == 156) & (arr[:, :, 1] == 74) & (arr[:, :, 2] == 0)
        
        alpha[mask_black | mask_dbrown | mask_lbrown] = 0
        del arr
        del alpha
    else:
        # For ESRGAN, the transparency is already perfectly handled by the alpha channel!
        # We DO NOT want to use colorkey=(0,0,0) because that will erase the smoothed black outlines!
        pass

    # Scale the sprite using smoothscale for ESRGAN, scale for Original
    new_w = int(nes_w * scale)
    new_h = int(nes_h * scale)
    if is_esrgan:
        image = pg.transform.smoothscale(image, (new_w, new_h))
    else:
        image = pg.transform.scale(image, (new_w, new_h))
        
    return image


# ─── Build a composite level frame ─────────────────────────────────────
def build_level_surface(gfx, target_height, is_esrgan=False):
    """Render the full level background scaled to a specific height."""
    bg = gfx['level_1']
    bg_rect = bg.get_rect()

    # The background image is 4x NES resolution.
    NES_WIDTH = bg_rect.width / 4
    NES_HEIGHT = bg_rect.height / 4

    # Calculate scale factor relative to NES resolution
    scale_factor = target_height / NES_HEIGHT
    final_w = int(NES_WIDTH * scale_factor)
    final_h = int(target_height)

    bg_scaled = pg.transform.scale(bg, (final_w, final_h))

    # ── Overlay some key sprites for visual comparison ──
    # Sizes relative to the NES resolution
    sheet_mario = gfx.get('mario_bros')
    sheet_tiles = gfx.get('tile_set')
    sheet_items = gfx.get('item_objects')
    sheet_enemies = gfx.get('enemies')

    if sheet_mario:
        # Mario standing (NES coord: 178, 32, size: 12x16)
        mario_img = get_sprite(sheet_mario, 178, 32, 12, 16, scale_factor,
                               is_esrgan=is_esrgan)
        mario_x = int(110 * scale_factor / 2.679) # Approximate level positioning
        mario_y = int(224 * scale_factor / 2.679) - mario_img.get_height()
        bg_scaled.blit(mario_img, (mario_x, mario_y))

    if sheet_tiles:
        # Bricks at their level positions
        brick_positions_x = [858, 944, 1030]
        for bx in brick_positions_x:
            brick = get_sprite(sheet_tiles, 16, 0, 16, 16, scale_factor,
                               is_esrgan=is_esrgan)
            bx_screen = int(bx * scale_factor / 2.679)
            by_screen = int(136 * scale_factor / 2.679)
            bg_scaled.blit(brick, (bx_screen, by_screen))

    if sheet_items:
        # Question blocks (NES coord: 0, 16, size: 16x16)
        qblock_positions = [(685, 136), (901, 136), (987, 136), (943, 72)]
        for qx, qy in qblock_positions:
            qblock = get_sprite(sheet_items, 0, 16, 16, 16, scale_factor,
                                is_esrgan=is_esrgan)
            qx_screen = int(qx * scale_factor / 2.679)
            qy_screen = int(qy * scale_factor / 2.679)
            bg_scaled.blit(qblock, (qx_screen, qy_screen))

    if sheet_enemies:
        # Goomba (NES coord: 0, 4, size: 16x16)
        goomba = get_sprite(sheet_enemies, 0, 4, 16, 16, scale_factor,
                            is_esrgan=is_esrgan)
        gx = int(640 * scale_factor / 2.679)
        gy = int(224 * scale_factor / 2.679) - goomba.get_height()
        bg_scaled.blit(goomba, (gx, gy))

    return bg_scaled, final_w


def main():
    pg.init()

    # Get display info for sizing
    display_info = pg.display.Info()
    screen_w = min(display_info.current_w - 100, 1800)
    screen_h = min(display_info.current_h - 100, 800)

    DIVIDER = 6
    HEADER = 50
    FOOTER = 35
    panel_w = (screen_w - 2 * DIVIDER) // 3
    panel_h = screen_h - HEADER - FOOTER

    screen = pg.display.set_mode((screen_w, screen_h), pg.RESIZABLE)
    pg.display.set_caption("Pixel-Perfect: 1x Original vs 4x Nearest vs 4x ESRGAN")
    clock = pg.time.Clock()
    fullscreen = False

    # Load graphics
    print("Loading original/nearest graphics...")
    nearest_gfx = load_gfx(NEAREST_GFX_DIR)
    print("Loading ESRGAN upscaled graphics...")
    upscaled_gfx = load_gfx(UPSCALED_GFX_DIR)

    # Build full level surfaces
    # For Panel 1 (1x Original), we scale it so its height is 224 * 2 = 448 (a standard small window size)
    # or just fit to panel_w but keeping NES aspect ratio. Let's use exactly 2.5x NES resolution.
    NES_BASE_HEIGHT = 224
    PANEL_1_SCALE = 2.0
    panel_1_h = int(NES_BASE_HEIGHT * PANEL_1_SCALE)

    print("Rendering 1x original level (small window mode)...")
    true_orig_level, true_orig_w = build_level_surface(nearest_gfx, panel_1_h, is_esrgan=False)
    
    print("Rendering Nearest-Neighbor stretched level...")
    nearest_level, nearest_w = build_level_surface(nearest_gfx, panel_h, is_esrgan=False)
    
    print("Rendering ESRGAN upscaled stretched level...")
    upsc_level, upsc_w = build_level_surface(upscaled_gfx, panel_h, is_esrgan=True)

    # Fonts
    try:
        font = pg.font.SysFont('DejaVu Sans', 18, bold=True)
        small_font = pg.font.SysFont('DejaVu Sans', 13)
    except Exception:
        font = pg.font.SysFont(None, 20, bold=True)
        small_font = pg.font.SysFont(None, 16)

    viewport_x_nes = 0
    scroll_speed_nes = 2
    max_scroll_nes = (true_orig_w / PANEL_1_SCALE) - (panel_w / PANEL_1_SCALE)

    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
                elif event.key == pg.K_f:
                    fullscreen = not fullscreen
                    if fullscreen:
                        screen = pg.display.set_mode((0, 0), pg.FULLSCREEN)
                        screen_w, screen_h = screen.get_size()
                    else:
                        screen_w = min(display_info.current_w - 100, 1800)
                        screen_h = min(display_info.current_h - 100, 800)
                        screen = pg.display.set_mode((screen_w, screen_h),
                                                     pg.RESIZABLE)
                    panel_w = (screen_w - 2 * DIVIDER) // 3
                    panel_h = screen_h - HEADER - FOOTER

                    # Re-render nearest and upscaled for new height
                    nearest_level, nearest_w = build_level_surface(nearest_gfx, panel_h, is_esrgan=False)
                    upsc_level, upsc_w = build_level_surface(upscaled_gfx, panel_h, is_esrgan=True)
                    max_scroll_nes = (true_orig_w / PANEL_1_SCALE) - (panel_w / PANEL_1_SCALE)
                    viewport_x_nes = min(viewport_x_nes, max_scroll_nes)

        keys = pg.key.get_pressed()
        move = 0
        if keys[pg.K_RIGHT]:
            move = scroll_speed_nes
        if keys[pg.K_LEFT]:
            move = -scroll_speed_nes
            
        if keys[pg.K_LSHIFT] or keys[pg.K_RSHIFT]:
            move *= 4

        viewport_x_nes = max(0, min(viewport_x_nes + move, max_scroll_nes))

        # ── Draw ──
        screen.fill((15, 15, 25))

        # Panel 1: TRUE ORIGINAL (Small view, centered in panel)
        true_orig_view = pg.Surface((panel_w, panel_h))
        true_orig_view.fill((15, 15, 25))
        # Center vertically
        y_offset = (panel_h - panel_1_h) // 2
        # Calculate scaled viewport x
        v_x_1 = int(viewport_x_nes * PANEL_1_SCALE)
        true_orig_view.blit(true_orig_level, (-v_x_1, y_offset))
        
        # Draw a border around the small game area
        pg.draw.rect(true_orig_view, (100, 100, 120), 
                     (0, y_offset - 2, panel_w, panel_1_h + 4), 2)
        screen.blit(true_orig_view, (0, HEADER))

        pg.draw.rect(screen, (255, 200, 50),
                      (panel_w, 0, DIVIDER, screen_h))

        # Panel 2: NEAREST NEIGHBOR (Stretched)
        nearest_scale = panel_h / NES_BASE_HEIGHT
        v_x_nearest = int(viewport_x_nes * nearest_scale)
        nearest_view = pg.Surface((panel_w, panel_h))
        nearest_view.blit(nearest_level, (-v_x_nearest, 0))
        screen.blit(nearest_view, (panel_w + DIVIDER, HEADER))

        pg.draw.rect(screen, (255, 200, 50),
                      (panel_w * 2 + DIVIDER, 0, DIVIDER, screen_h))

        # Panel 3: UPSCALED (ESRGAN Stretched)
        upsc_scale = panel_h / NES_BASE_HEIGHT
        v_x_upsc = int(viewport_x_nes * upsc_scale)
        upsc_view = pg.Surface((panel_w, panel_h))
        upsc_view.blit(upsc_level, (-v_x_upsc, 0))
        screen.blit(upsc_view, (panel_w * 2 + DIVIDER * 2, HEADER))

        # Header labels
        label_bg = pg.Surface((screen_w, HEADER))
        label_bg.fill((15, 15, 25))
        label_bg.set_alpha(220)
        screen.blit(label_bg, (0, 0))

        lbl_true = font.render("ORIGINAL (Actual NES Size)", True,
                                (255, 100, 100))
        lbl_nearest = font.render("NEAREST-NEIGHBOR (Stretched)", True,
                                (255, 255, 100))
        lbl_upsc = font.render("AI-UPSCALED (ESRGAN Stretched)", True,
                                (100, 255, 130))
        
        screen.blit(lbl_true,
                     (panel_w // 2 - lbl_true.get_width() // 2, 12))
        screen.blit(lbl_nearest,
                     (panel_w + DIVIDER + panel_w // 2 -
                      lbl_nearest.get_width() // 2, 12))
        screen.blit(lbl_upsc,
                     (panel_w * 2 + DIVIDER * 2 + panel_w // 2 -
                      lbl_upsc.get_width() // 2, 12))

        # Footer
        progress = viewport_x_nes / max(max_scroll_nes, 1)
        bar_w = screen_w - 200
        bar_x = 100
        bar_y = screen_h - 22

        controls = small_font.render(
            "← → Scroll   |   SHIFT + ← → Fast Scroll   |   F Fullscreen   |   ESC Quit",
            True, (150, 150, 160))
        screen.blit(controls,
                     (screen_w // 2 - controls.get_width() // 2,
                      screen_h - FOOTER + 2))

        # Progress bar
        pg.draw.rect(screen, (50, 50, 60), (bar_x, bar_y, bar_w, 6),
                      border_radius=3)
        pg.draw.rect(screen, (255, 200, 50),
                      (bar_x, bar_y, int(bar_w * progress), 6),
                      border_radius=3)

        pg.display.flip()
        clock.tick(60)

    pg.quit()
    print("Comparison viewer closed.")


if __name__ == '__main__':
    main()
