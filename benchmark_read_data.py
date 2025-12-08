
import sys
import os
import time
import numpy as np
import cv2 as cv

sys.path.append(os.path.abspath("C:/Users/HomeUser/source/repos/rompar"))

from rompar.rompar import Rompar
from rompar.config import Config

def benchmark_read_data():
    config = Config()
    
    # Large image
    img_height, img_width = 2000, 2000
    dummy_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        cv.imwrite(tf.name, dummy_img)
        img_fn = tf.name
    
    try:
        # Simulate user's 240x256 grid (~60k points)
        group_cols = 8
        group_rows = 1
        r = Rompar(config, img_fn=img_fn, group_cols=group_cols, group_rows=group_rows)
        
        # Manually set grid points
        step = 8 # denser grid
        r._grid_points_x = list(range(10, img_width-10, step)) # ~250
        r._grid_points_y = list(range(10, img_height-10, step)) # ~250
        
        # Ensure we have ~60k points
        r._grid_points_x = r._grid_points_x[:240]
        r._grid_points_y = r._grid_points_y[:256]
        
        print(f"Grid size: {len(r._grid_points_x)}x{len(r._grid_points_y)} = {len(r._grid_points_x)*len(r._grid_points_y)} points")
        
        # Init data
        r._Rompar__data = np.zeros((len(r._grid_points_y), len(r._grid_points_x)), dtype=bool)

        # Benchmark read_data
        start_time = time.time()
        r.read_data()
        end_time = time.time()
        print(f"read_data time: {end_time - start_time:.4f} seconds")
        
    finally:
        os.remove(img_fn)

if __name__ == "__main__":
    benchmark_read_data()
