
import time
import numpy as np
import cv2 as cv
import threading

def benchmark_threading_viewport():
    W, H = 8000, 8000
    n_points = 260000 # 512x512
    radius = 6
    img = np.zeros((H, W), dtype=np.uint8)
    
    # Grid points
    xs = np.random.randint(0, W, n_points).astype(int)
    ys = np.random.randint(0, H, n_points).astype(int)
    points = list(zip(xs, ys))
    
    print(f"Benchmarking 512x512 ({n_points}) points on {W}x{H} image")
    
    # 1. Baseline Serial
    img.fill(0)
    start = time.time()
    for x, y in points:
        cv.circle(img, (x, y), radius, 255, -1)
    print(f"Baseline Serial: {time.time() - start:.4f}s")
    
    # 2. Threaded (4 threads)
    img.fill(0)
    start = time.time()
    def draw_chunk(pts):
        for x, y in pts:
            cv.circle(img, (x, y), radius, 255, -1)
            
    chunks = np.array_split(points, 4)
    threads = []
    for chunk in chunks:
        # Array split returns numpy array of arrays?
        # points is list of tuples.
        # np.array_split will convert to numpy array?
        # Let's simple split list
        pass
        
    chunk_size = n_points // 4
    pts_chunks = [points[i:i+chunk_size] for i in range(0, n_points, chunk_size)]
    
    for chunk in pts_chunks:
        t = threading.Thread(target=draw_chunk, args=(chunk,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    print(f"Threaded (4 threads): {time.time() - start:.4f}s")
    
    # 3. Viewport Culling
    # Simulate zoomed in view (e.g. 1920x1080 visible)
    view_w, view_h = 1920, 1080
    view_x, view_y = 2000, 2000 # some offset
    
    img.fill(0)
    start = time.time()
    
    # Filter points
    # Vectorized filter assumption
    # Convert to array
    arr_x = np.array(xs)
    arr_y = np.array(ys)
    
    mask = (arr_x >= view_x) & (arr_x < view_x + view_w) & \
           (arr_y >= view_y) & (arr_y < view_y + view_h)
           
    visible_x = arr_x[mask].tolist()
    visible_y = arr_y[mask].tolist()
    
    # Draw visible
    for x, y in zip(visible_x, visible_y):
        cv.circle(img, (x, y), radius, 255, -1)
        
    print(f"Viewport Culling (Filter+Draw): {time.time() - start:.4f}s")

if __name__ == "__main__":
    benchmark_threading_viewport()
