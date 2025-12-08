
import sys
import os
import shutil

# Adjust path to include the repo root
pass
import sys
import unittest.mock
mock_cv = unittest.mock.Mock()
mock_img = unittest.mock.Mock()
mock_img.shape = (100, 100, 3)
mock_cv.imread.return_value = mock_img
mock_cv.IMREAD_COLOR = 1
sys.modules["cv2"] = mock_cv

mock_numpy = unittest.mock.Mock()
mock_numpy.zeros.return_value = mock_img
mock_numpy.copy.return_value = mock_img
mock_numpy.array.side_effect = lambda x: unittest.mock.Mock(tolist=lambda: x, __iter__=lambda: iter(x), size=len(x))
# We need basic array ops for _calculate_grid_intersections?
# It uses vector math (Ve - Vs) etc.
# This might be too hard to mock fully if grid math runs.
# But for verify_undo, `_calculate_grid_intersections` is called by `redraw_grid`.
# We can mock `_calculate_grid_intersections` on the instance?
sys.modules["numpy"] = mock_numpy

sys.path.append(r'c:\Users\HomeUser\source\repos\rompar')

from rompar.rompar import Rompar
from rompar.config import Config
from rompar.history import MoveColumnCommand

# Mock file existence
img_fn = r'c:\Users\HomeUser\source\repos\rompar\small.png'

def verification():
    config = Config()
    romp = Rompar(config, img_fn=img_fn, group_cols=8, group_rows=8)
    # Mock calculation to avoid numpy issues
    romp._calculate_grid_intersections = unittest.mock.Mock(return_value=(unittest.mock.Mock(size=0), unittest.mock.Mock(size=0)))
    
    # Setup initial grid
    # Add vertical lines at 10, 20, 30
    romp.add_bit_column(10)
    romp.add_bit_column(20)
    romp.add_bit_column(30)
    
    print(f"Initial lines: {[l.start for l in romp._grid_lines_v]}")
    
    # Select lines 0, 1 (10, 20)
    romp.select_toggle_v(0)
    romp.select_toggle_v(1)
    
    start_selection = list(romp.selected_indices_v) # Should be [0, 1]
    start_idx = 0
    print(f"Start selection: {start_selection}")
    
    # Simulate Drag
    # Move by +5 pixels
    # We simulate what UI does: incremental moves calling move_bit_column with push_history=False
    # But for simplicity, let's just do one big move
    dx = 5
    print(f"Moving by {dx}...")
    romp.move_bit_column(0, dx, relative=True, push_history=False)
    
    print(f"Lines after move: {[l.start for l in romp._grid_lines_v]}")
    # Should be [15, 25, 30]
    
    # Now UI pushes command
    cmd = MoveColumnCommand(romp, start_idx, dx, relative=True, indices=start_selection)
    
    # UI populates final indices
    cmd.final_indices = list(romp.selected_indices_v)
    cmd.final_idx = romp.Edit_x
    print(f"Command final indices: {cmd.final_indices}")
    
    romp.history.push(cmd)
    
    # Verify State before Undo
    lines = [l.start for l in romp._grid_lines_v]
    if lines != [15.0, 25.0, 30.0]:
         print("ERROR: Move didn't work as expected?")
    
    # Undo
    print("Undoing...")
    romp.history.undo()
    
    lines_after_undo = [l.start for l in romp._grid_lines_v]
    print(f"Lines after undo: {lines_after_undo}")
    
    if lines_after_undo == [10.0, 20.0, 30.0]:
         print("Undo SUCCESS")
    else:
         print("Undo FAILED")
         
    # Redo
    print("Redoing...")
    romp.history.redo()
    lines_after_redo = [l.start for l in romp._grid_lines_v]
    print(f"Lines after redo: {lines_after_redo}")

    if lines_after_redo == [15.0, 25.0, 30.0]:
         print("Redo SUCCESS")
    else:
         print("Redo FAILED")

verification()
