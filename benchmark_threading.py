
import time
import numpy as np
import cv2 as cv
from rompar.rompar import Rompar, GridLine

# Mock Config
class Config:
    def __init__(self):
        self.radius = 6
        self.inverted = False
        self.default_radius = None
        self.img_display_blank_image = False
        self.img_display_original = False
        self.img_display_grid = True # Important
        self.img_display_peephole = False
        self.img_display_data = False
        self.bit_thresh_div = 1
        self.pix_thresh_min = 0
        self.pix_thresh_max = 255
        self.pix_thresh_local = False
        self.pix_thresh_max_delta = 10
        self.dilate = 0
        self.erode = 0
        self.threads = 4

def benchmark_threading():
    print("Testing Threaded Redraw...")
    W, H = 4000, 4000
    r = Rompar(Config(), img_fn="test.png", group_cols=10, group_rows=10)
    # Mock image load
    r.img_original = np.zeros((H, W, 3), dtype=np.uint8)
    r.img_target = r.img_original.copy()
    r.img_grid = np.zeros((H, W, 3), dtype=np.uint8)
    r.img_peephole = np.zeros((H, W), dtype=np.uint8)
    r.img_data_cache = np.zeros((H, W, 3), dtype=np.uint8)
    r.img_shape # Force init properties
    
    # Init huge grid: 400 cols x 400 rows = 160k pts
    r._grid_lines_v = [GridLine(x, x) for x in range(0, W, 10)]
    r._grid_lines_h = [GridLine(y, y) for y in range(0, H, 10)]
    
    r._Rompar__data = np.zeros((r.bit_height, r.bit_width), dtype=bool)
    
    print("Running Full Redraw (Threaded)...")
    r.grid_dirty = True
    start = time.time()
    r.redraw_grid(viewport=None)
    full_time = time.time() - start
    print(f"Full Redraw (Threaded, 160k pts): {full_time:.4f}s")
    
    # We will run this again after implementation to verify speedup

if __name__ == "__main__":
    benchmark_threading()
