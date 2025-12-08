
import time
import numpy as np
import cv2 as cv

def benchmark_scaling():
    # Simulate user's probable scenario: Large image, small radius
    W, H = 8000, 8000 # 64MP image
    n_points = 60000
    radius = 4
    
    print(f"Benchmarking Dilation vs Iteration on {W}x{H} image with {n_points} points, radius {radius}")
    
    # Setup data
    img = np.zeros((H, W), dtype=np.uint8)
    
    # Random points
    xs = np.random.randint(0, W, n_points)
    ys = np.random.randint(0, H, n_points)
    points = list(zip(xs, ys))
    
    # 1. Iterative
    start = time.time()
    for x, y in points:
        cv.circle(img, (x, y), radius, 255, -1)
    print(f"Iterative cv.circle: {time.time() - start:.4f}s")
    
    # 2. Dilation
    img.fill(0)
    # Create mask of points
    start_setup = time.time()
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[ys, xs] = 1 # Advanced indexing is fast
    print(f"Mask setup: {time.time() - start_setup:.4f}s")
    
    start_dilate = time.time()
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    dilated = cv.dilate(mask, kernel)
    print(f"Dilation only: {time.time() - start_dilate:.4f}s")
    print(f"Total Vectorized: {time.time() - start_setup:.4f}s")

if __name__ == "__main__":
    benchmark_scaling()
