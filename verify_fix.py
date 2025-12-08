
import sys
import os

# Adjust path to include the repo root
sys.path.append(r'c:\Users\HomeUser\source\repos\rompar')

from rompar import Rompar
from rompar.config import Config
import pathlib

# Create a mock config
config = Config()

# Mock file existence for Rompar init
img_fn = r'c:\Users\HomeUser\source\repos\rompar\small.png'
# Ensure image exists or mock cv? Rompar uses cv.imread.
# small.png exists in file listing.

try:
    # Initialize Rompar
    # It requires img_fn, group_cols, group_rows
    # We can pass dummy values
    romp = Rompar(config, img_fn=img_fn, group_cols=8, group_rows=8)
    
    print("Rompar initialized successfully.")

    # Test select_toggle_v
    print("Testing select_toggle_v...")
    idx = 0
    romp.select_toggle_v(idx)
    assert idx in romp.selected_indices_v, "Index should be selected"
    assert romp.selected_line_v == idx, "Primary selection should be set"
    
    romp.select_toggle_v(idx)
    assert idx not in romp.selected_indices_v, "Index should be deselected"
    assert romp.selected_line_v != idx, "Primary selection should be cleared or changed"

    print("select_toggle_v passed.")

    # Test select_toggle_h
    print("Testing select_toggle_h...")
    idx = 0
    romp.select_toggle_h(idx)
    assert idx in romp.selected_indices_h, "Index should be selected"
    assert romp.selected_line_h == idx, "Primary selection should be set"

    romp.select_toggle_h(idx)
    assert idx not in romp.selected_indices_h, "Index should be deselected"
    assert romp.selected_line_h != idx, "Primary selection should be cleared or changed"

    print("select_toggle_h passed.")
    print("ALL TESTS PASSED")

except Exception as e:
    print(f"FAILED with error: {e}")
    import traceback
    traceback.print_exc()
