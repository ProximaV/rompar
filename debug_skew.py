
import numpy as np
from rompar.rompar import Rompar, GridLine, ImgXY

# Mock Config
class Config:
    def __init__(self):
        self.radius = 10
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

def debug_interaction():
    print("Initializing Rompar...")
    H, W = 100, 100
    r = Rompar(Config(), img_fn="test.png", group_cols=10, group_rows=10)
    # Mock image
    r.img_original = np.zeros((H, W, 3), dtype=np.uint8)
    r.img_target = r.img_original.copy()
    r.img_grid = np.zeros((H, W, 3), dtype=np.uint8)
    r.img_shape # Force init
    
    # Add one vertical line at x=50
    # Add one horizontal line at y=50
    r._grid_lines_v = [GridLine(50, 50)]
    r._grid_lines_h = [GridLine(50, 50)]
    
    print("\n--- Testing Hit Test ---")
    # Test Hit Body V
    hit = r.grid_hit_test(ImgXY(50, 20), threshold=5) # Should hit index 0, both
    print(f"Hit(50, 20): {hit}")
    
    # Test Hit Start V
    hit = r.grid_hit_test(ImgXY(50, 2), threshold=5) # Should hit index 0, start
    print(f"Hit(50, 2): {hit}")
    
    # Test Hit End V
    hit = r.grid_hit_test(ImgXY(50, 98), threshold=5) # Should hit index 0, end
    print(f"Hit(50, 98): {hit}")

    # Test Hit Mid V (Midpoint approx 50, 50)
    hit = r.grid_hit_test(ImgXY(50, 50), threshold=5) # Should hit index 0, both (mid-handle)
    print(f"Hit(50, 50): {hit}")
    
    # Test Miss
    hit = r.grid_hit_test(ImgXY(60, 20), threshold=5)
    print(f"Hit(60, 20): {hit}")

    print("\n--- Testing BitXY ---")
    # Test exact intersection
    try:
        b = r.imgxy_to_bitxy(ImgXY(50, 50))
        print(f"BitXY(50, 50): {b}")
    except Exception as e:
        print(f"BitXY(50, 50) FAILED: {e}")

    # Test slightly off (within delta=5)
    try:
        b = r.imgxy_to_bitxy(ImgXY(52, 52))
        print(f"BitXY(52, 52): {b}")
    except Exception as e:
        print(f"BitXY(52, 52) FAILED: {e}")

    # Test Far off
    try:
        b = r.imgxy_to_bitxy(ImgXY(60, 60))
        print(f"BitXY(60, 60): {b}")
    except Exception as e:
        print(f"BitXY(60, 60): Expected failure caught: {e}")

if __name__ == "__main__":
    debug_interaction()
