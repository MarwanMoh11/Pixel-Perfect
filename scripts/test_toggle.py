import pygame as pg
import sys
import threading
import time

def inject_keypress():
    time.sleep(2)
    print("Injecting T keydown...")
    pg.event.post(pg.event.Event(pg.KEYDOWN, key=pg.K_t))
    time.sleep(0.1)
    pg.event.post(pg.event.Event(pg.KEYUP, key=pg.K_t))
    print("Injected T key!")
    time.sleep(2)
    print("Quitting...")
    pg.event.post(pg.event.Event(pg.QUIT))

threading.Thread(target=inject_keypress, daemon=True).start()

sys.path.append('/home/marwan/AML/Pixel-Perfect/mario_clone')
import mario_level_1
try:
    mario_level_1.main()
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
