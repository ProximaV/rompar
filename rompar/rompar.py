from collections import namedtuple
import cv2 as cv
import os.path
import json
import numpy
import time
import pathlib
import concurrent.futures

BLACK  = (0x00, 0x00, 0x00)
BLUE   = (0xff, 0x00, 0x00)
GREEN  = (0x00, 0xff, 0x00)
RED    = (0x00, 0x00, 0xff)
YELLOW = (0x00, 0xff, 0xff)
WHITE  = (0xff, 0xff, 0xff)

from .history import (History, Command, MoveColumnCommand, MoveRowCommand, 
                      ToggleBitCommand, AddColumnCommand, DeleteColumnCommand, 
                      AddRowCommand, DeleteRowCommand)


ImgXY = namedtuple('ImgXY', ['x', 'y'])
BitXY = namedtuple('BitXY', ['x', 'y'])

class GridLine:
    def __init__(self, start, end):
        self.start = float(start)
        self.end = float(end)

    def get_at(self, pos, max_pos):
        if max_pos == 0: return self.start
        t = pos / max_pos
        return self.start + (self.end - self.start) * t

    def to_json(self):
        return {"start": self.start, "end": self.end}

    def __repr__(self):
        return f"GridLine({self.start}, {self.end})"

class Rompar(object):
    def mode_is_grid(self):
        return self.is_grid_mode

    def set_grid_mode(self, is_grid_mode):
        self.is_grid_mode = is_grid_mode
        self.grid_dirty = True

    @property
    def Edit_x(self):
         return self.selected_line_v if self.selected_line_v is not None else -1
    @Edit_x.setter
    def Edit_x(self, val):
         self.selected_line_v = val if val >= 0 else None
         
         # Sync with multi-select (Backward compatibility)
         self.selected_indices_v.clear()
         if val >= 0:
             self.selected_indices_v.add(val)
             
         if val >= 0: self.selected_handle = 'both'
         self.grid_dirty = True

    @property
    def Edit_y(self):
         return self.selected_line_h if self.selected_line_h is not None else -1
    @Edit_y.setter
    def Edit_y(self, val):
         self.selected_line_h = val if val >= 0 else None
         
         # Sync with multi-select
         self.selected_indices_h.clear()
         if val >= 0:
             self.selected_indices_h.add(val)

         if val >= 0: self.selected_handle = 'both'
         self.grid_dirty = True

    def select_toggle_v(self, idx):
        if idx in self.selected_indices_v:
            self.selected_indices_v.remove(idx)
            if self.selected_line_v == idx:
                self.selected_line_v = next(iter(self.selected_indices_v)) if self.selected_indices_v else None
        else:
            self.selected_indices_v.add(idx)
            self.selected_line_v = idx
        self.grid_dirty = True

    def select_toggle_h(self, idx):
        if idx in self.selected_indices_h:
            self.selected_indices_h.remove(idx)
            if self.selected_line_h == idx:
                self.selected_line_h = next(iter(self.selected_indices_h)) if self.selected_indices_h else None
        else:
            self.selected_indices_h.add(idx)
            self.selected_line_h = idx
        self.grid_dirty = True

    def __init__(self, config, *, img_fn=None, grid_json=None,
                 group_cols=0, group_rows=0, grid_dir_path=None,
                 annotate=None):
        self.is_grid_mode = False # State flag controlled by UI
        self.img_fn = pathlib.Path(img_fn).expanduser().absolute() \
                      if img_fn else None
        self.config = config
        self.annotate = annotate
        
        # Allow skipping of process_target_image if nothing changed.
        self.__process_cache = None

        # Pixels between cols and rows
        self.step_x, self.step_y = (0, 0)
        # Number of rows/cols per bit grouping
        self.group_cols, self.group_rows = (group_cols, group_rows)

        self.Search_HEX = None

        # Selection state for skew edition
        self.selected_line_v = None
        self.selected_line_h = None
        self.selected_indices_v = set()
        self.selected_indices_h = set()
        self.selected_handle = 'both' # 'start', 'end', 'both'



        # Global
        self._grid_lines_v = [] # Vertical lines (stored as x positions)
        self._grid_lines_h = [] # Horizontal lines (stored as y positions)

        if grid_json:
            self.load_json(grid_json, grid_dir_path)

        # Then need critical args
        if not self.img_fn:
            raise Exception("Filename required")
        if not self.group_cols:
            raise Exception("cols required")
        if not self.group_rows:
            raise Exception("rows required")

        #load img as numpy ndarray dimensions (height, width, channels)
        self.img_original = cv.imread(str(self.img_fn), cv.IMREAD_COLOR)
        assert self.img_original is not None, "Failed to load %s" % (self.img_fn,)
        print ('Image is %dx%d; %d channels' %
               (self.img_width, self.img_height, self.img_channels))

        # Image buffers
        self.img_target = numpy.copy(self.img_original)
        self.img_grid = numpy.zeros(self.img_original.shape, numpy.uint8)
        self.img_peephole = numpy.zeros(self.img_original.shape, numpy.uint8)

        # Caching
        self.grid_dirty = True
        self.data_dirty = True
        self.is_grid_mode = False
        self.img_data_cache = numpy.zeros(self.img_original.shape, numpy.uint8)
        
        self.history = History()
        
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.config.threads)

        self.__process_target_image()

        self.__data = numpy.ndarray((self.bit_height, self.bit_width), dtype=bool)

        if not (grid_json and grid_json['data'] and
                self.__parse_grid_bit_data(grid_json['data'])):
            self.read_data()

    def load_json(self, grid_json, grid_dir_path):
        if self.img_fn is None:
            # TODO: If absolute paths are supported in the json
            # file, then saving the grid file should probably not
            # write in a relative path. For now, all img_fn paths
            # should be relative.
            self.img_fn = pathlib.Path(grid_json.get('img_fn'))
            if not self.img_fn.is_absolute():
                # Eventually, it would be nice if this class was
                # unaware of the file system so it can be used
                # with any data source (Files, Databases, etc) in
                # any system configuration (desktop, web backend,
                # etc). But that is not the case yet, so the
                # current grid file's directory is necessary to
                # calculate the relative path of img_fn when
                # saving and loading a grid's configuration.
                self.img_fn = grid_dir_path / self.img_fn
                self.img_fn = self.img_fn.resolve()
        if self.group_cols is None:
            self.group_cols = grid_json.get('group_cols')
            self.group_rows = grid_json.get('group_rows')

        def parse_legacy_list(l):
             res = []
             for item in l:
                  if isinstance(item, (int, float)):
                       res.append(GridLine(item, item))
                  elif isinstance(item, dict):
                       res.append(GridLine(item['start'], item['end']))
             return res
        
        # Use existing JSON keys 'grid_points_x' for backward compatibility
        self._grid_lines_v = parse_legacy_list(grid_json.get('grid_points_x', []))
        self._grid_lines_v.sort(key=lambda l: l.start)

        self._grid_lines_h = parse_legacy_list(grid_json.get('grid_points_y', []))
        self._grid_lines_h.sort(key=lambda l: l.start)

        self.config.update(grid_json['config'])

        print ('Grid points: %d x, %d y' % (len(self._grid_lines_v),
                                            len(self._grid_lines_h)))

        if len(self._grid_lines_v) > 1 and self.group_cols > 1:
            self.step_x = (self._grid_lines_v[self.group_cols - 1].start -
                           self._grid_lines_v[0].start) / \
                          (self.group_cols - 1)
        if len(self._grid_lines_h) > 1 and self.group_rows > 1:
            self.step_y = (self._grid_lines_h[self.group_rows - 1].start -
                           self._grid_lines_h[0].start) / \
                          (self.group_rows - 1)
        if hasattr(self.config, 'radius') and self.config.radius > 6:
             pass # kept from user
        elif self.config.default_radius:
            self.config.radius = self.config.default_radius
        else:
            if self.step_x:
                self.config.radius = int(self.step_x / 3)
            elif self.step_y:
                self.config.radius = int(self.step_y / 3)
        
        # Respect loaded radius if present
        if grid_json.get('radius'):
             self.config.radius = grid_json.get('radius')

    def shift_xy(self, dx, dy):
        """Move data points a relative amount relative to existing image. Used to align an old project to a new image"""
        # self.redraw()
        for line in self._grid_lines_v:
             line.start += dx
             line.end += dx
        for line in self._grid_lines_h:
             line.start += dy
             line.end += dy

    def __parse_grid_bit_data(self, data):
        if isinstance(data, list):
            try:
                data = "".join(data)
            except Exception as e:
                print("File 'data' field is in unknown format. Ignoring.")
                return False
        if not isinstance(data, str):
            print("File 'data' field is incompatible type '%s'. Ignoring." %\
                  str(type(data)))
            return False
        if (self.bit_height*self.bit_width) != len(data):
            print("Data length (%d) is different than the number of "
                  "grid intersections (%d). Ignoring data" %
                  (len(data), self.bit_height*self.bit_width))
            return False
        if set(data).difference({'0', '1'}):
            print("File 'data' contains not 0/1 characters: %s." %\
                  set(data).difference({'0', '1'}))
            return False

        bit_iter = (bit == '1' for bit in data)
        for bit_x in range(self.bit_width):
            for bit_y in range(self.bit_height):
                self.set_data(BitXY(bit_x, bit_y), next(bit_iter))
        return True

    def _calculate_grid_intersections(self):
        # Arrays of start/end
        Vs = numpy.array([l.start for l in self._grid_lines_v])
        Ve = numpy.array([l.end for l in self._grid_lines_v])
        Hs = numpy.array([l.start for l in self._grid_lines_h])
        He = numpy.array([l.end for l in self._grid_lines_h])
        
        H, W = self.img_height, self.img_width
        
        # Avoid division by zero if no lines
        if len(Vs) == 0: Vs = numpy.zeros(0)
        if len(Ve) == 0: Ve = numpy.zeros(0)
        if len(Hs) == 0: Hs = numpy.zeros(0)
        if len(He) == 0: He = numpy.zeros(0)

        # Broadcast to proper shapes
        # dv: (1, Nc), dh: (Nr, 1)
        dv = ((Ve - Vs) / H).reshape(1, -1) if H > 0 else numpy.zeros((1, len(Vs)))
        dh = ((He - Hs) / W).reshape(-1, 1) if W > 0 else numpy.zeros((len(Hs), 1))
        
        Vs = Vs.reshape(1, -1)
        Hs = Hs.reshape(-1, 1)
        
        # x = (Vs + dv*Hs) / (1 - dv*dh)
        # y = Hs + dh * x
        
        denom = 1.0 - dv * dh
        # Handle parallel case safety (though unlikely to be exactly 0 unless perfectly diagonal vs diagonal)
        denom[numpy.abs(denom) < 1e-9] = 1.0 
        
        grid_x = (Vs + dv * Hs) / denom
        grid_y = Hs + dh * grid_x
        
        return grid_x.astype(int), grid_y.astype(int)

    def redraw_grid(self, viewport=None, fast=False):
        if not self.grid_dirty and viewport is None:
            return
        
        # Clear viewport area if provided, else clear whole image
        if viewport:
             vx, vy, vw, vh = viewport
             vx = max(0, int(vx))
             vy = max(0, int(vy))
             vw = min(self.img_width - vx, int(vw))
             vh = min(self.img_height - vy, int(vh))
             self.img_grid[vy:vy+vh, vx:vx+vw] = 0
             self.img_peephole[vy:vy+vh, vx:vx+vw] = 0
        else:
             self.img_grid.fill(0)
             self.img_peephole.fill(0)

        t = time.time()
        
        # Draw skewed lines
        # Optimization: Culling. Check if line intersects viewport?
        # Vertical line: check x at top and bottom. If both outside range left or right, skip.
        # Conservatively: min(start, end) > vx+vw OR max(start, end) < vx -> Skip
        
        for i, l in enumerate(self._grid_lines_v):
             # Culling
             if viewport:
                  min_x, max_x = min(l.start, l.end), max(l.start, l.end)
                  if min_x > vx + vw or max_x < vx: continue
             
             is_selected = (self.mode_is_grid() and i in self.selected_indices_v)
             color = RED if is_selected else BLUE
             cv.line(self.img_grid, (int(l.start), 0), (int(l.end), self.img_height), color, 1)
             
             if self.mode_is_grid():
                  h_rad = 4 if is_selected else 2
                  h_col = GREEN if is_selected else YELLOW
                  cv.circle(self.img_grid, (int(l.start), 0), h_rad, h_col, -1)
                  cv.circle(self.img_grid, (int(l.end), self.img_height), h_rad, h_col, -1)

                  # Mid Handle only if selected
                  if is_selected and self.selected_handle == 'both':
                       my = self.img_height // 2
                       mx = int(l.get_at(my, self.img_height))
                       cv.rectangle(self.img_grid, (mx-4, my-4), (mx+4, my+4), GREEN, -1)

        for i, l in enumerate(self._grid_lines_h):
             # Culling
             if viewport:
                  # Skewed H-lines: y varies.
                  min_y, max_y = min(l.start, l.end), max(l.start, l.end)
                  if min_y > vy + vh or max_y < vy: continue

             is_selected = (self.mode_is_grid() and i in self.selected_indices_h)
             color = RED if is_selected else BLUE
             
             try:
                 y1, y2 = int(l.start), int(l.end)
                 y1 = max(-20000, min(20000, y1))
                 y2 = max(-20000, min(20000, y2))
                 cv.line(self.img_grid, (0, y1), (self.img_width, y2), color, 1)

                 if self.mode_is_grid():
                      h_rad = 4 if is_selected else 2
                      h_col = GREEN if is_selected else YELLOW
                      cv.circle(self.img_grid, (0, y1), h_rad, h_col, -1)
                      cv.circle(self.img_grid, (self.img_width, y2), h_rad, h_col, -1)

                      # Mid Handle
                      if is_selected and self.selected_handle == 'both':
                           mx = self.img_width // 2
                           my = int(l.get_at(mx, self.img_width))
                           my = max(-20000, min(20000, my))
                           cv.rectangle(self.img_grid, (mx-4, my-4), (mx+4, my+4), GREEN, -1)
             except (ValueError, OverflowError):
                 pass

        t = time.time()
        
        # Calculate intersection points
        mg_x, mg_y = self._calculate_grid_intersections()
        
        if mg_x.size == 0:
            self.grid_dirty = False
            return
            
        h, w = self.img_shape[:2]
        grid_x_safe = numpy.clip(mg_x, 0, w-1)
        grid_y_safe = numpy.clip(mg_y, 0, h-1)
        
        # mg_x, mg_y are already meshgrid-like (Nr, Nc)
        # Proceed with existing mask logic using these points
        
        # 2. State masks (Grid based)
        # Retrieve logical state from __data
        # data_state matches grid dimensions (H_grid, W_grid)
        data_state = self.__data.copy()
        if self.config.inverted:
            data_state = ~data_state
        
        # Viewport Culling Mask
        if viewport:
            # Select points roughly within viewport
            # mg_x, mg_y are (H_grid, W_grid)
            vx, vy, vw, vh = viewport
            # Add margin for radius
            margin = int(self.config.radius) + 5
            vx, vy = vx - margin, vy - margin
            vw, vh = vw + 2*margin, vh + 2*margin
            
            # Create boolean mask
            viewport_mask = (mg_x >= vx) & (mg_x < vx+vw) & \
                            (mg_y >= vy) & (mg_y < vy+vh)
                            
            # Apply to data_state?
            # Wait, data_state controls Blue/Green.
            # If we mask data_state, the False values become True? No.
            # If we mask, we want to set EVERYTHING to False outside viewport.
            # But grid_blue is ~data_state. So outside becomes True?
            
            # Better strategy: Apply viewport_mask to final grid_blue/green masks.
        else:
            viewport_mask = None
        
        # Initial assignment
        grid_green = data_state
        grid_blue = ~data_state
        
        # Determine Edit/Select Logic
        grid_white = None
        grid_red = None
        
        if self.Edit_x >= 0:
            # Create a bool mask for the grid
            grid_select_mask = numpy.zeros(data_state.shape, dtype=bool)
            
            # Use broadcasting or ranges to set True
            H_grid, W_grid = data_state.shape
            
            # bit_xy.x == self.Edit_x
            if 0 <= self.Edit_x < W_grid:
                grid_select_mask[:, self.Edit_x] = True
            
            # bit_xy.y == self.Edit_y and range
            if 0 <= self.Edit_y < H_grid:
                sx = self.Edit_x - (self.Edit_x % self.group_cols)
                start_x = max(0, sx)
                end_x = min(W_grid, sx + self.group_cols)
                grid_select_mask[self.Edit_y, start_x:end_x] = True
                
            # Filter selected from Blue/Green
            # grid_select_mask indicates "Edit/Selected"
            
            # Extract Selected Blue/Green to become Red/White
            # If was Green (ON) and Selected -> White
            grid_white = grid_green & grid_select_mask
            # If was Blue (OFF) and Selected -> Red
            grid_red = grid_blue & grid_select_mask
            
            # Remove Selected from Blue/Green
            grid_green = grid_green & ~grid_select_mask
            grid_blue = grid_blue & ~grid_select_mask
        
        # Apply Viewport Culling if active
        if viewport_mask is not None:
             grid_blue &= viewport_mask
             grid_green &= viewport_mask
             if grid_white is not None: grid_white &= viewport_mask
             if grid_red is not None: grid_red &= viewport_mask
        else:
             viewport_mask = numpy.ones(mg_x.shape, dtype=bool) # for peephole filtering
             
        # Prepare Peephole (All points)
        # We need (x, y) pairs for all grid points
        # mg_x, mg_y are (H_grid, W_grid) arrays of coordinates
        # Filter by viewport mask for efficient peephole drawing
        # Flatten and convert to python list for fast iteration
        
        # Optimization: use viewport_mask on mg_x/mg_y before ravel
        all_x = mg_x[viewport_mask].ravel().tolist()
        all_y = mg_y[viewport_mask].ravel().tolist()
        
        # Draw Peephole loops
        # radius + 1
        pr = int(self.config.radius) + 1
        
        # Use zip for fast iteration
        for x, y in zip(all_x, all_y):
            cv.circle(self.img_peephole, (x, y), pr, WHITE, -1)

        # Draw Colors
        # Define drawing properties
        rad = int(self.config.radius)
        thick = 2
        
        def draw_subset(xs, ys, color, radius, thickness, img):
            # Inner loop in python is the bottleneck
            # Splitting it across threads helps if GIL allows cv.circle to run
            # OpenCV's bind functions often give up GIL
            for x, y in zip(xs, ys):
                cv.circle(img, (x, y), radius, color, thickness)
        
        # with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.threads) as executor:
        # Re-use executor to save thread creation time
        if True: 
            executor = self.executor
            futures = []
            
            # Process each color group
            for grid_mask, color in [(grid_blue, BLUE), (grid_green, GREEN), (grid_red, RED), (grid_white, WHITE)]:
                if grid_mask is None or not numpy.any(grid_mask):
                    continue
                
                # Extract coordinates
                # This numpy access is fast
                xs = mg_x[grid_mask].ravel()
                ys = mg_y[grid_mask].ravel()
                
                n_points = len(xs)
                if n_points == 0: continue
                
                # Chunking
                # Adjust chunks based on count
                n_chunks = self.config.threads
                if n_points < 100000: n_chunks = 1
                
                chunk_size = (n_points + n_chunks - 1) // n_chunks
                
                if n_chunks == 1:
                     # Avoid threading overhead for single chunk
                     draw_subset(xs.tolist(), ys.tolist(), color, rad, thick, self.img_grid)
                else:
                    for i in range(0, n_points, chunk_size):
                        end = min(i + chunk_size, n_points)
                        
                        sub_x = xs[i:end].tolist()
                        sub_y = ys[i:end].tolist()
                        
                        futures.append(executor.submit(draw_subset, sub_x, sub_y, color, rad, thick, self.img_grid))
            
            # Wait for all
            concurrent.futures.wait(futures)

        self.grid_dirty = False

    def render_image(self, img_display=None, rgb=False, viewport=None, fast=False):
        t = time.time()
        if img_display is None:
             img_display = numpy.zeros(self.img_original.shape, numpy.uint8)
        if self.config.img_display_blank_image:
            img_display.fill(0)
        elif self.config.img_display_original:
            numpy.copyto(img_display, self.img_original)
        else:
            numpy.copyto(img_display, self.img_target)

        if self.config.img_display_grid:
            self.redraw_grid(viewport=viewport, fast=fast)
            cv.bitwise_or(img_display, self.img_grid, img_display)

        if self.config.img_display_peephole:
            cv.bitwise_and(img_display, self.img_peephole, img_display)

        if self.config.img_display_data:
            self.render_data_layer(img_display)

        if self.annotate:
            self.render_annotate(img_display)

        print("render_image time:", time.time()-t)

        if rgb:
            cv.cvtColor(img_display, cv.COLOR_BGR2RGB, img_display);

        return img_display

    def read_data(self, bit_pairs=None):
        process_redone = self.__process_target_image()
        if process_redone or bit_pairs is None:
            # Vectorized full update
            bit_pairs = self.iter_bitxy()
            
            # maximum possible value if all pixels are set
            maxval = (self.config.radius ** 2) * 255
            thresh = (maxval / self.config.bit_thresh_div)
            delta = int(self.config.radius // 2)
            
            print('read_data: computing (vectorized)')
            
            # Use boxFilter to sum pixels in window
            # boxFilter sums if normalize=False
            ksize = (delta * 2, delta * 2)
            
            # Handle edge case where radius/delta is small
            if ksize[0] < 1: ksize = (1, 1)
                
            # Compute sum for every pixel
            # ddepth=-1 means same depth as source (uint8), but sum can overflow uint8
            # So we use CV_32S or CV_64F. Source is uint8.
            sum_img = cv.boxFilter(self.img_target, cv.CV_32S, ksize, normalize=False, borderType=cv.BORDER_CONSTANT)
            
            # Now extract values at grid points
            # We assume grid points are valid coordinates
            # Note: boxFilter anchors at center. 
            # Slice in original code: [y-delta : y+delta]. 
            # Center of slicing window of size 2*delta is at 'delta' offset from top-left.
            # Grid point is at 'y'.
            # If window is [y-delta, y+delta], the center relative to y is 0 (if even size?).
            # OpenCV boxFilter anchor default is (-1,-1) i.e. center.
            # If ksize is even (2*delta), center is at (ksize-1)/2.
            # Example delta=2, ksize=4. center index 1.5 -> 1 or 2.
            # We need to match precise sum window. 
            # Current code: y-delta to y+delta. Length 2*delta.
            # Center of this range is y.
            # So boxFilter centered at y should match.
            
            # Use advanced indexing to get all values at once
            # bit_xy in bit_pairs gives us bit coordinates. 
            # We need to map to img coordinates.
            # But since we are iterating ALL bits (bit_pairs is basically all), 
            # we can just use _grid_points.
            
            # To handle potential non-uniform grid or partial updates (if logic falls through),
            # we should be careful. 
            # But here we are in the "full update" block mostly.
            # The optimization is most valuable for full update.
            
            # Construct meshgrid of coordinates
            # Only valid if grid is uniform? No, iter_bitxy iterates all combinations.
            # So we can use broadcasting.
            # Construct meshgrid of coordinates using skewed lines
            mg_x, mg_y = self._calculate_grid_intersections()
            
            if mg_x.size == 0:
                return

            # Map to integer coordinates (they should be ints already)
            
            # Validate bounds
            h, w = self.img_shape[:2]
            grid_x_safe = numpy.clip(mg_x, 0, w-1)
            grid_y_safe = numpy.clip(mg_y, 0, h-1)
            
            # Extract sums
            # Note: we are assigning to self.__data which is (bit_height, bit_width)
            # self.__data[y, x] corresponds to grid_y[y], grid_x[x]
            
            try:
                # If we have full mesh arrays, directly indexing works
                values = sum_img[grid_y_safe, grid_x_safe]
                
                # values is (grid_h, grid_w, channels) e.g. (240, 256, 3)
                # We need scalar sum per grid point.
                if len(values.shape) == 3:
                     # Sum across channels
                     values = values.sum(axis=2)
                
                # Check threshold
                new_data = values > thresh
                
                # Update __data
                # We can assign directly since we processed the full grid
                self.__data[:] = new_data
                self.data_dirty = True
                self.grid_dirty = True
                
            except Exception as e:
                print(f"Vectorized read_data failed: {e}. Falling back to iterative.")
                # Fallback logic could go here or we re-raise
                # Let's revert to iterative if this fails to be safe, easier to just copy-paste original loop
                # Re-using original loop code below for non-full update anyway
                bit_pairs = self.iter_bitxy() # Reset iterator
                for bit_xy in bit_pairs:
                    img_xy = self.bitxy_to_imgxy(bit_xy)
                    datasub = self.img_target[img_xy.y - delta:img_xy.y + delta,
                                              img_xy.x - delta:img_xy.x + delta]
                    value = datasub.sum(dtype=int)
                    self.set_data(bit_xy, value > thresh)
            return

        # Partial update logic (iterative)
        maxval = (self.config.radius ** 2) * 255
        thresh = (maxval / self.config.bit_thresh_div)
        delta = int(self.config.radius // 2)

        print('read_data: computing (partial)')
        for bit_xy in bit_pairs:
            img_xy = self.bitxy_to_imgxy(bit_xy)
            datasub = self.img_target[img_xy.y - delta:img_xy.y + delta,
                                      img_xy.x - delta:img_xy.x + delta]
            value = datasub.sum(dtype=int)
            self.set_data(bit_xy, value > thresh)

    def get_pixel(self, img_xy):
        img_x, img_y = img_xy
        return self.img_target[img_y, img_x].sum()

    def write_data_as_txt(self, f):
        for bit_y in range(self.bit_height):
            if bit_y and bit_y % self.group_rows == 0:
                f.write('\n') # Put a space between row gaps
            for bit_x in range(self.bit_width):
                if bit_x and bit_x % self.group_cols == 0:
                    f.write(' ')
                bit = self.get_data(BitXY(bit_x, bit_y)) ^ self.config.inverted
                f.write("1" if bit else "0")
            f.write('\n') # Newline afer every row

    def load_txt_data(self, f):
        def gen_bits():
            while True:
                c = f.read(1)
                if not c:
                    return
                if c in "01":
                    yield c

        bits = ''.join([c for c in gen_bits()])
        assert len(bits) == self.bit_n, "Wanted %u bits (%uw x %uh) but got %u bits" % (
                self.bit_n, self.bit_width, self.bit_height, len(bits))

        biti = 0
        for bit_y in range(self.bit_height):
            for bit_x in range(self.bit_width):
                self.set_data(BitXY(bit_x, bit_y), bits[biti] == "1")
                biti += 1

    def dump_grid_configuration(self, grid_dir_path):
        config = dict(self.config.__dict__)
        config['view'] = config['view'].__dict__

        # Store the img path relative to the grid file
        img_fn_rel = os.path.relpath(str(self.img_fn), str(grid_dir_path))

        # XXX: this first cut is partly due to ease of converting old DB
        # Try to move everything non-volatile into config object
        j = {
            # Increment major when a fundamentally breaking change occurs
            # minor reserved for now, but could be used for non-breaking
            'version': (1, 1),
            #'grid_intersections': list(self.iter_grid_intersections()),
            'data': ["1" if self.get_data(BitXY(bit_x, bit_y)) else "0"
                     for bit_x in range(self.bit_width)
                     for bit_y in range(self.bit_height)],
            'grid_points_x': [l.to_json() for l in self._grid_lines_v],
            'grid_points_y': [l.to_json() for l in self._grid_lines_h],
            'fn': config,
            'group_cols': self.group_cols,
            'group_rows': self.group_rows,
            'config': config,
            'img_fn': img_fn_rel,
            }
        return j

    def __process_target_image(self):
        new_cache_value = (self.config.pix_thresh_min,
                           self.config.dilate, self.config.erode)
        if self.__process_cache == new_cache_value:
            return False
        self.__process_cache = new_cache_value

        t = time.time()
        cv.dilate(self.img_target, (3,3))
        cv.threshold(self.img_original, self.config.pix_thresh_min,
                     0xff, cv.THRESH_BINARY, self.img_target)
        cv.bitwise_and(self.img_target, (0, 0, 255), self.img_target)
        if self.config.dilate:
            cv.dilate(self.img_target, (3,3))
        if self.config.erode:
            cv.erode(self.img_target, (3,3))
        print("process_image time", time.time()-t)
        return True

    def _get_intersection(self, v_idx, h_idx):
        if not (0 <= v_idx < len(self._grid_lines_v)) or \
           not (0 <= h_idx < len(self._grid_lines_h)):
             return (0, 0)
             
        v = self._grid_lines_v[v_idx]
        h = self._grid_lines_h[h_idx]
        H, W = self.img_height, self.img_width
        
        dv = (v.end - v.start) / H if H else 0
        dh = (h.end - h.start) / W if W else 0
        
        denom = 1.0 - dv * dh
        if abs(denom) < 1e-9: denom = 1.0
        
        x = (v.start + dv * h.start) / denom
        y = h.start + dh * x
        return int(x), int(y)

    def bitx_to_imgx(self, bit_x):
        if (0 > bit_x >= self.bit_width):
            raise IndexError("Bit x-coodrinate (%d) out of range"%bit_x)
        # Return start point as approximation or top-edge x
        return int(self._grid_lines_v[bit_x].start)

    def bity_to_imgy(self, bit_y):
        if (0 > bit_y >= self.bit_width):
            raise IndexError("Bit y-coodrinate (%d) out of range"%bit_y)
        return int(self._grid_lines_h[bit_y].start)

    def bitxy_to_imgxy(self, bit_xy):
        bit_x, bit_y = bit_xy
        if (0 > bit_x >= self.bit_width) or \
           (0 > bit_y >= self.bit_height):
            raise IndexError("Bit coodrinate (%d, %d) out of range"%bit_xy)
        x, y = self._get_intersection(bit_x, bit_y)
        return ImgXY(x, y)

    def imgxy_to_bitxy(self, img_xy, autocenter=True):
        img_x, img_y = img_xy
        if (0 > img_x >= self.img_width) or (0 > img_y >= self.img_height):
            raise IndexError("Image coodrinate (%d, %d) out of range"%img_xy)

        if autocenter:
            delta = self.config.radius / 2
            # Minimum tolerance of 5 pixels to assist clicking small radii
            delta = max(delta, 5)
            H, W = self.img_height, self.img_width
            
            # Find closest V-line
            found_x = -1
            for bit_x, l in enumerate(self._grid_lines_v):
                # Calculate x at this y
                lx = l.get_at(img_y, H)
                if (lx - delta) <= img_x <= (lx + delta):
                    found_x = bit_x
                    break
            
            if found_x == -1:
                 raise IndexError("No bit near image coordinate (%d, %d)"%img_xy)

            # Find closest H-line
            found_y = -1
            for bit_y, l in enumerate(self._grid_lines_h):
                # Calculate y at this x
                ly = l.get_at(img_x, W)
                if (ly - delta) <= img_y <= (ly + delta):
                    found_y = bit_y
                    break
            
            if found_y == -1:
                 raise IndexError("No bit near image coordinate (%d, %d)"%img_xy)
                 
            return BitXY(found_x, found_y)
        else:
             # Find closest index based on start points (approx)
             # Or better: find interval? 
             # For Edit Mode, we need closest line to click.
             # This function is used for Toggling Data too.
             # If exact match required, this is hard.
             # The 'else' block was using .index(img_x).
             # It implies exact match on grid line coordinate?
             # Probably never reached or useful only if clicking EXACTLY on line?
             # Let's approximate using starts.
             # Or better, iterate and find closest line.
             
             best_x = -1
             min_dist_x = 9999
             for i, l in enumerate(self._grid_lines_v):
                  dist = abs(l.get_at(img_y, self.img_height) - img_x)
                  if dist < min_dist_x:
                       min_dist_x = dist
                       best_x = i
             
             # Metric for 'exact' match? < 1 pixel?
             if min_dist_x > 2: # Tolerance
                  pass 
             
             best_y = -1
             min_dist_y = 9999
             for i, l in enumerate(self._grid_lines_h):
                  dist = abs(l.get_at(img_x, self.img_width) - img_y)
                  if dist < min_dist_y:
                       min_dist_y = dist
                       best_y = i

             return BitXY(best_x, best_y)

    def get_data(self, bit_xy, inv=False):
        bit_x, bit_y = bit_xy
        ret = self.__data[bit_y, bit_x]
        return (not ret) if inv else (ret)

    def set_data(self, bit_xy, val):
        bit_x, bit_y = bit_xy
        self.__data[bit_y, bit_x] = bool(val)
        self.data_dirty = True
        self.grid_dirty = True
        return bool(val)

    def toggle_data(self, bit_xy):
        old_val = self.get_data(bit_xy)
        self.history.push(ToggleBitCommand(self, bit_xy, old_val))
        return self.set_data(bit_xy, not old_val)

    def update_radius(self):
        if self.config.radius:
            return

        if self.config.default_radius:
            self.config.radius = self.config.default_radius
            self.grid_dirty = True
        else:
            if self.step_x:
                self.config.radius = int(self.step_x / 3)
                self.grid_dirty = True
            elif self.step_y:
                self.config.radius = int(self.step_y / 3)
                self.grid_dirty = True

    def auto_center(self, img_xy):
        '''
        Auto center image global x/y coordinate on contiguous pixel x/y runs
        '''
        img_x, img_y = img_xy
        x_min = img_x
        while self.get_pixel((img_y, x_min)) != 0.0:
            x_min -= 1
        x_max = img_x
        while self.get_pixel((img_y, x_max)) != 0.0:
            x_max += 1
        img_x = x_min + ((x_max - x_min) // 2)
        y_min = img_y
        while self.get_pixel((y_min, img_x)) != 0.0:
            y_min -= 1
        y_max = img_y
        while self.get_pixel((y_max, img_x)) != 0.0:
            y_max += 1
        img_y = y_min + ((y_max - y_min) // 2)
        return ImgXY(img_x, img_y)

    #def draw_Hline(self, img_y, intersections):
    #    cv.line(self.img_grid, (0, img_y), (self.img_width, img_y), BLUE, 1)
    #    for gridx in self._grid_points_x:
    #        self.grid_draw_circle((gridx, img_y), BLUE)
    #
    #def draw_Vline(self, img_x, intersections):
    #    cv.line(self.img_grid, (img_x, 0), (img_x, self.img_height), BLUE, 1)
    #    for gridy in self._grid_points_y:
    #        self.grid_draw_circle((img_x, gridy), BLUE)

    def grid_draw_circle(self, img_xy, color, thick=1):
        cv.circle(self.img_grid, img_xy, int(self.config.radius), BLACK, -1)
        cv.circle(self.img_grid, img_xy, int(self.config.radius), color, thick)

    def render_data_layer(self, img):
        if self.data_dirty:
            self.img_data_cache.fill(0)
            for bit_y in range(self.bit_height):
                for bit_column in range(self.bit_width // self.group_cols):
                    for column_byte in range(self.group_cols // 8):
                        byte = ''
                        bit_group_x = bit_column*self.group_cols + column_byte*8
                        for bit_x_offset in range(8):
                            bit = self.get_data(BitXY(bit_group_x+bit_x_offset, bit_y),
                                                inv=self.config.inverted)
                            byte += "1" if bit else "0"
                        if self.config.LSB_Mode:
                            byte = byte[::-1]
                        num = int(byte, 2)

                        if self.config.img_display_binary:
                            disp_data = format(num, '08b')
                        else:
                            disp_data = format(num, "02X")

                        textcolor = WHITE
                        if self.Search_HEX and self.Search_HEX.count(num):
                            textcolor = YELLOW

                        cv.putText(
                            self.img_data_cache,
                            disp_data,
                            self.bitxy_to_imgxy(BitXY(bit_group_x, bit_y)),
                            cv.FONT_HERSHEY_SIMPLEX,
                            self.config.font_size,
                            textcolor,
                            thickness=2)
            self.data_dirty = False

        if img is None:
            # If no image provided, return copy of cache (or just cache?)
            # Returning copy to be safe, though usage in render_image implies drawing ONTO img.
            # But here we are asked to return img.
            # If img was None, we create new.
            img = numpy.zeros(self.img_shape, numpy.uint8)
        
        # Composite the cached data layer onto the target image
        # Since data layer is text on black (0) background, we can use add or bitwise_or if no overlap.
        # Or copyto with mask? Simplest is add assuming no background.
        # But wait, render_image calls it with img_display which has content.
        # So we need to overlay.
        # cv.putText draws with anti-aliasing usually, but on black background 
        # it's just pixels.
        # Let's use bitwise_or or add. Text is usually bright.
        cv.bitwise_or(img, self.img_data_cache, img)

        return img

    def render_annotate(self, img):
        for (col, row), annotation in self.annotate.items():
            img_xy = self.bitxy_to_imgxy((col, row))
            x, y = img_xy
            r, g, b = annotation.get("color", (255, 80, 0))
            color = (b, g, r)
            thickness = annotation.get("thickness", 2)
            radius = annotation.get("radius", self.config.radius + 1)
            # covers up bit definition
            # cv.circle(img, img_xy, int(self.config.radius), color, thickness)
            cv.rectangle(img, (x - radius, y - radius), (x + radius, y + radius), color, thickness)

    def grid_hit_test(self, img_xy, threshold=10):
        img_x, img_y = img_xy
        H, W = self.img_height, self.img_width
        
        # Check Verticals
        for i, l in enumerate(self._grid_lines_v):
             # Endpoints (prioritize handles)
             if abs(l.start - img_x) < threshold and img_y < threshold:
                  return (i, 'start', True)
             if abs(l.end - img_x) < threshold and abs(img_y - H) < threshold:
                  return (i, 'end', True)
             # Midpoint Handle (approx center of view or image?? Image for now)
             # Center of image handle for 'both' selection
             mid_y = H / 2
             mid_x = l.get_at(mid_y, H)
             if abs(mid_x - img_x) < threshold and abs(img_y - mid_y) < threshold:
                  return (i, 'both', True)

        for i, l in enumerate(self._grid_lines_h):
             if abs(l.start - img_y) < threshold and img_x < threshold:
                  return (i, 'start', False)
             if abs(l.end - img_y) < threshold and abs(img_x - W) < threshold:
                  return (i, 'end', False)
             mid_x = W / 2
             mid_y = l.get_at(mid_x, W)
             if abs(mid_y - img_y) < threshold and abs(img_x - mid_x) < threshold:
                  return (i, 'both', False)

        # Check Bodies
        for i, l in enumerate(self._grid_lines_v):
             line_x = l.get_at(img_y, H)
             if abs(line_x - img_x) < threshold:
                  return (i, 'both', True)

        for i, l in enumerate(self._grid_lines_h):
             line_y = l.get_at(img_x, W)
             if abs(line_y - img_y) < threshold:
                  return (i, 'both', False)
        
        return None

    def add_bit_column(self, img_x):
        idx = self._add_bit_column_internal(img_x)
        if idx is not None:
             cmd = AddColumnCommand(self, img_x)
             cmd.added_idx = idx
             self.history.push(cmd)
             return True
        return False

    def _add_bit_column_internal(self, img_x):
        # Check for duplicates based on start position approx?
        for l in self._grid_lines_v:
             if int(l.start) == int(img_x) and int(l.end) == int(img_x):
                  return None

        new_line = GridLine(img_x, img_x)
        
        # Insert sorted by start
        for i, l in enumerate(self._grid_lines_v):
            if img_x < l.start:
                self._grid_lines_v.insert(i, new_line)
                break
        else:
            i = len(self._grid_lines_v)
            self._grid_lines_v.append(new_line)

        self.grid_dirty = True
        self.__data = numpy.insert(self.__data, i, False, axis = 1)
        self.read_data(((i, tmp_y) for tmp_y in range(self.bit_height)))
        return i

    def add_bit_row(self, img_y):
        idx = self._add_bit_row_internal(img_y)
        if idx is not None:
             cmd = AddRowCommand(self, img_y)
             cmd.added_idx = idx
             self.history.push(cmd)
             return True
        return False

    def _add_bit_row_internal(self, img_y):
        for l in self._grid_lines_h:
             if int(l.start) == int(img_y) and int(l.end) == int(img_y):
                  return None

        new_line = GridLine(img_y, img_y)

        for i, l in enumerate(self._grid_lines_h):
            if img_y < l.start:
                self._grid_lines_h.insert(i, new_line)
                break
        else:
            i = len(self._grid_lines_h)
            self._grid_lines_h.append(new_line)

        self.grid_dirty = True
        self.__data = numpy.insert(self.__data, i, False, axis = 0)
        self.read_data(((tmp_x, i) for tmp_x in range(self.bit_width)))
        return i

    def del_bit_column(self, bit_x):
        res = self._del_bit_column_internal(bit_x)
        if res:
             cmd = DeleteColumnCommand(self, bit_x)
             cmd.saved_line, cmd.saved_data = res
             self.history.push(cmd)
             return True
        return False

    def _del_bit_column_internal(self, bit_x):
        if bit_x >= self.bit_width:
            return None
        
        line = self._grid_lines_v[bit_x]
        col_data = self.__data[:, bit_x].copy() # Copy essential? yes, deleting from array

        del self._grid_lines_v[bit_x]
        self.__data = numpy.delete(self.__data, bit_x, axis = 1)
        self.grid_dirty = True
        self.data_dirty = True
        return (line, col_data)

    def _restore_bit_column_internal(self, idx, line, col_data):
        self._grid_lines_v.insert(idx, line)
        self.__data = numpy.insert(self.__data, idx, col_data, axis=1)
        self.grid_dirty = True
        self.data_dirty = True

    def del_bit_row(self, bit_y):
        res = self._del_bit_row_internal(bit_y)
        if res:
             cmd = DeleteRowCommand(self, bit_y)
             cmd.saved_line, cmd.saved_data = res
             self.history.push(cmd)
             return True
        return False

    def _del_bit_row_internal(self, bit_y):
        if bit_y >= self.bit_height:
            return None

        line = self._grid_lines_h[bit_y]
        row_data = self.__data[bit_y, :].copy()
        
        del self._grid_lines_h[bit_y]
        self.__data = numpy.delete(self.__data, bit_y, axis=0)
        self.grid_dirty = True
        self.data_dirty = True
        return (line, row_data)

    def _restore_bit_row_internal(self, idx, line, row_data):
        self._grid_lines_h.insert(idx, line)
        self.__data = numpy.insert(self.__data, idx, row_data, axis=0)
        self.grid_dirty = True
        self.data_dirty = True

    def sort_and_rebuild_grid(self, axis):
        """
        Sort grid lines and corresponding data columns/rows.
        Updates selected_indices to match new positions.
        axis: 1 for columns (vertical lines), 0 for rows (horizontal lines)
        Returns: map of old_index -> new_index
        """
        if axis == 1:
             lines = self._grid_lines_v
             sel_indices = self.selected_indices_v
             sel_line_attr = 'selected_line_v'
        else:
             lines = self._grid_lines_h
             sel_indices = self.selected_indices_h
             sel_line_attr = 'selected_line_h'

        # Pack (line, data_slice, original_index)
        n = len(lines)
        if n == 0: return {}

        packed = []
        for i in range(n):
             if axis == 1:
                  d = self.__data[:, i].copy()
             else:
                  d = self.__data[i, :].copy()
             packed.append((lines[i], d, i))
        
        # Sort by line start
        packed.sort(key=lambda x: x[0].start)
        
        # Unpack
        new_lines = []
        new_data_list = []
        idx_map = {}
        
        for new_i, (l, d, old_i) in enumerate(packed):
             new_lines.append(l)
             new_data_list.append(d)
             idx_map[old_i] = new_i
        
        # Update State
        if axis == 1:
             self._grid_lines_v = new_lines
             self.__data = numpy.stack(new_data_list, axis=1) if new_data_list else self.__data
        else:
             self._grid_lines_h = new_lines
             self.__data = numpy.stack(new_data_list, axis=0) if new_data_list else self.__data
             
        # Remap selection
        new_sel = set()
        for old_i in sel_indices:
             if old_i in idx_map:
                  new_sel.add(idx_map[old_i])
        
        if axis == 1:
             self.selected_indices_v = new_sel
             old_v = getattr(self, sel_line_attr)
             if old_v in idx_map:
                  setattr(self, sel_line_attr, idx_map[old_v])
        else:
             self.selected_indices_h = new_sel
             old_h = getattr(self, sel_line_attr)
             if old_h in idx_map:
                  setattr(self, sel_line_attr, idx_map[old_h])
                  
        return idx_map

    def move_bit_column(self, bit_x, new_img_x, relative=False, push_history=True):
        # Determine targets
        indices = [bit_x]
        if bit_x in self.selected_indices_v:
             indices = list(self.selected_indices_v)
        
        handle_type = self.selected_handle
        res_map = self._move_bit_column_internal(bit_x, new_img_x, relative, indices=indices, handle_type=handle_type)
        if res_map:
             if push_history:
                 cmd = MoveColumnCommand(self, bit_x, new_img_x, relative, indices=indices, handle_type=handle_type)
                 target_indices = indices
                 cmd.final_indices = [res_map[i] for i in target_indices if i in res_map]
                 if bit_x in res_map: cmd.final_idx = res_map[bit_x]
                 self.history.push(cmd)
             return True
        return False

    def _move_bit_column_internal(self, bit_x, new_img_x, relative=False, indices=None, handle_type=None):
        indices = indices if indices else [bit_x]
        valid_indices = [i for i in indices if 0 <= i < len(self._grid_lines_v)]
        if not valid_indices: return None

        # Calculate dx based on bit_x (the leader)
        primary_line = self._grid_lines_v[bit_x]
        if relative:
             dx = new_img_x 
        else:
             dx = new_img_x - primary_line.start
        
        if dx == 0: return None

        use_handle = handle_type if handle_type is not None else self.selected_handle

        # Apply move based on selected handle
        for i in valid_indices:
             line = self._grid_lines_v[i]
             if use_handle in ('start', 'both'):
                  line.start += dx
             if use_handle in ('end', 'both'):
                  line.end += dx
        
        # Sort and Rebuild
        idx_map = self.sort_and_rebuild_grid(axis=1)
        
        self.grid_dirty = True
        self.data_dirty = True
        return idx_map

    def move_bit_row(self, bit_y, new_img_y, relative=False, push_history=True):
        # Determine targets
        indices = [bit_y]
        if bit_y in self.selected_indices_h:
             indices = list(self.selected_indices_h)
             
        handle_type = self.selected_handle
        res_map = self._move_bit_row_internal(bit_y, new_img_y, relative, indices=indices, handle_type=handle_type)
        if res_map:
             if push_history:
                 cmd = MoveRowCommand(self, bit_y, new_img_y, relative, indices=indices, handle_type=handle_type)
                 target_indices = indices
                 cmd.final_indices = [res_map[i] for i in target_indices if i in res_map]
                 if bit_y in res_map: cmd.final_idx = res_map[bit_y]
                 self.history.push(cmd)
             return True
        return False

    def _move_bit_row_internal(self, bit_y, new_img_y, relative=False, indices=None, handle_type=None):
        indices = indices if indices else [bit_y]
        valid_indices = [i for i in indices if 0 <= i < len(self._grid_lines_h)]
        if not valid_indices: return None
             
        primary_line = self._grid_lines_h[bit_y]
        
        # Calculate delta
        if relative:
             dy = new_img_y 
        else:
             dy = new_img_y - primary_line.start
        
        if dy == 0: return None

        use_handle = handle_type if handle_type is not None else self.selected_handle

        # Apply move based on selected handle
        for i in valid_indices:
             line = self._grid_lines_h[i]
             if use_handle in ('start', 'both'):
                  line.start += dy
             if use_handle in ('end', 'both'):
                  line.end += dy
        
        # Sort and Rebuild
        idx_map = self.sort_and_rebuild_grid(axis=0)

        self.grid_dirty = True
        self.data_dirty = True
        return idx_map



    def grid_add_vertical_line(self, img_xy, do_autocenter=True):
        if do_autocenter:
            img_xy = self.auto_center(img_xy)

        img_x, img_y = img_xy
        # Check by proximity to line start
        for l in self._grid_lines_v:
             if abs(l.start - img_x) < 2:
                  return

        # only draw a single line if this is the first one
        if self.bit_width == 0 or self.group_cols == 1:
            self.add_bit_column(img_x)
        else:
            # set up auto draw
            start_i = 0
            if self.bit_width == 1:
                # use a float to reduce rounding errors
                self.step_x = (img_x - self._grid_lines_v[0].start) / \
                              (self.group_cols - 1)
                img_x = self._grid_lines_v[0].start
                start_i = 1
                self.update_radius()
            # draw a full set of self.group_cols
            for x in range(start_i, self.group_cols):
                draw_x = int(img_x + x * self.step_x)
                if draw_x > self.img_width:
                    break
                self.add_bit_column(draw_x)

    def grid_add_horizontal_line(self, img_xy, do_autocenter=True):
        if do_autocenter and not self.get_pixel(img_xy):
            print ('autocenter: miss!')
            return

        if do_autocenter:
            img_xy = self.auto_center(img_xy)

        img_x, img_y = img_xy
        for l in self._grid_lines_h:
             if abs(l.start - img_y) < 2:
                  return

        # only draw a single line if this is the first one
        if self.bit_height == 0 or self.group_rows == 1:
            self.add_bit_row(img_y)
        else:
            # set up auto draw
            start_i = 0
            if self.bit_height == 1:
                # use a float to reduce rounding errors
                self.step_y = (img_y - self._grid_lines_h[0].start) / \
                              (self.group_rows - 1)
                img_y = self._grid_lines_h[0].start
                start_i = 1
                self.update_radius()
            # draw a full set of self.group_rows
            for y in range(start_i, self.group_rows):
                draw_y = int(img_y + y * self.step_y)
                # only draw up to the edge of the image
                if draw_y > self.img_height:
                    break
                self.add_bit_row(draw_y)

    @property
    def img_width(self):
        return self.img_shape[1]
    @property
    def img_height(self):
        return self.img_shape[0]
    @property
    def img_channels(self):
        return self.img_shape[2]
    @property
    def img_shape(self):
        return self.img_original.shape

    @property
    def bit_width(self):
        return len(self._grid_lines_v)
    @property
    def bit_height(self):
        return len(self._grid_lines_h)

    @property
    def bit_n(self):
        return len(self._grid_lines_v) * len(self._grid_lines_h)

    def iter_grid_intersections(self):
         mg_x, mg_y = self._calculate_grid_intersections()
         if mg_x.size == 0: return
         # Yield in order
         for i in range(len(self._grid_lines_v)):
              for j in range(len(self._grid_lines_h)):
                   yield ImgXY(mg_x[j, i], mg_y[j, i])

    def iter_bitxy(self):
        for bit_x in range(self.bit_width):
            for bit_y in range(self.bit_height):
                yield BitXY(bit_x, bit_y)
