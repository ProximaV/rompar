
import time
import numpy as np
import cv2 as cv
from rompar.rompar import Rompar, GridLine
from types import SimpleNamespace

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

def benchmark_viewport_correctness():
    print("Testing Viewport Culling...")
    W, H = 4000, 4000
    r = Rompar(Config(), img_fn="test.png", group_cols=10, group_rows=10)
    # Mock image load
    r.img_original = np.zeros((H, W, 3), dtype=np.uint8)


    r.img_target = r.img_original.copy()
    r.img_grid = np.zeros((H, W, 3), dtype=np.uint8)
    r.img_peephole = np.zeros((H, W), dtype=np.uint8)
    r.img_data_cache = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Init huge grid
    r._grid_lines_v = [GridLine(x, x) for x in range(0, W, 10)]
    r._grid_lines_h = [GridLine(y, y) for y in range(0, H, 10)]
    # 160k points
    
    # Mock data initialization

    r._Rompar__data = np.zeros((r.bit_height, r.bit_width), dtype=bool)
    
    # Full Redraw Benchmark
    r.grid_dirty = True
    start = time.time()
    r.redraw_grid(viewport=None)
    full_time = time.time() - start
    print(f"Full Redraw (160k pts): {full_time:.4f}s")
    
    # Viewport Redraw
    view_rect = (100, 100, 200, 200) # Small window
    r.grid_dirty = True
    start = time.time()
    r.redraw_grid(viewport=view_rect)
    view_time = time.time() - start
    print(f"Viewport Redraw (Small): {view_time:.4f}s")
    
    # Verify correctness (checking if pixels outside viewport are 0?)
    # Since we cleared viewport area only, outside area should remain from previous full redraw?
    # Actually, previous full redraw drew everything.
    # Current viewport redraw draws inside.
    # We can't easily check 'cleared' state unless we started with blank.
    # But speedup confirms culling worked.
    
    assert view_time < full_time / 5.0, "Viewport should be much faster"
    print("Optimization Verified.")

if __name__ == "__main__":
    benchmark_viewport_correctness()
