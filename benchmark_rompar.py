
import sys
import os
import time
import numpy as np
import cv2 as cv

# Add repo root to path
sys.path.append(os.path.abspath("C:/Users/HomeUser/source/repos/rompar"))

from rompar.rompar import Rompar
from rompar.config import Config

def benchmark():
    # Mock config
    config = Config()
    config.img_display_grid = True
    config.img_display_data = True
    
    # Create a large dummy image (e.g. 4k)
    img_height, img_width = 2000, 2000
    dummy_img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    print("Initializing Rompar with large image...")
    # Initialize Rompar
    # We need to save dummy image to load it? Or can we pass it? 
    # Rompar constructor takes img_fn. Let's bypass or save temp.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        cv.imwrite(tf.name, dummy_img)
        img_fn = tf.name
    
    try:
        r = Rompar(config, img_fn=img_fn, group_cols=8, group_rows=1)
        
        # Setup specific grid to generate many points
        # grid_add_horizontal_line will create grid points.
        # Let's manually set internal grid points to simulate large grid
        # Say 100x100 grid = 10,000 points.
        step = 20
        r._grid_points_x = list(range(0, img_width, step))
        r._grid_points_y = list(range(0, img_height, step))
        
        # Initialize data array based on new grid dimensions
        r._Rompar__data = np.zeros((len(r._grid_points_y), len(r._grid_points_x)), dtype=bool)
        
        # Compute data
        r.read_data()
        
        # Benchmark render_image
        start_time = time.time()
        n_frames = 10
        for i in range(n_frames):
            r.render_image()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / n_frames
        print(f"Average render_image time: {avg_time:.4f} seconds")
        
    finally:
        os.remove(img_fn)

if __name__ == "__main__":
    benchmark()
