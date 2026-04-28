import torch
import random
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
resolution = 512
grid = torch.zeros(1, 1, resolution, resolution, device=device)

kernel = torch.tensor([[0.0, 0.25, 0.0], 
                       [0.25, 0.0, 0.25], 
                       [0.0, 0.25, 0.0]], device=device) [None, None, :, :]

def drop_water(grid, x, y):
    grid[0, 0, x, y] += 50
    return grid

def spread (prev_grid, grid, kernel):
    neighbors_sum = F.conv2d(grid, kernel, padding=1)
    next_grid = neighbors_sum * 2.0 - prev_grid
    next_grid *= 0.95
    return next_grid

def simulate(grid):
    drops = [random.sample(range(resolution), 2) for _ in range(5)]
    buffer = []
    random_drop_time = [ random.randint(0, 1000) for _ in range(5) ] 
    print(random_drop_time)
    random_drop_time.sort()
    t = time.time()
    prev_grid = torch.zeros_like(grid)
    for i in range(1000):
        if i in random_drop_time:
            x, y = drops[random_drop_time.index(i)]
            grid = drop_water(grid, x, y)
        
        next_grid = spread(prev_grid, grid, kernel)
        prev_grid = grid.clone()
        grid = next_grid.clone()
        buffer.append(grid.clone())
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return buffer, time.time() - t

def render(grid):
    data = grid[0, 0, :, :] # resolution x resolution
    dy, dx = torch.gradient(data)
    slope_strength = 200.0
    dx *= slope_strength
    dy *= slope_strength
    mag = torch.sqrt(dx**2 + dy**2 + 1.0)
    nx, ny, nz = -dx/mag, -dy/mag, 1.0/mag
    highlight = torch.clamp((1.0 - nz) * 5.0, 0, 1)
    h, w = highlight.shape
    rgb = torch.zeros(h, w, 3, device=grid.device)
    rgb[..., 0] = torch.clamp(0.0 + highlight * 0.2, 0, 1) 
    rgb[..., 1] = torch.clamp(0.3 + highlight * 0.3, 0, 1) 
    rgb[..., 2] = torch.clamp(0.5 + highlight * 0.5, 0, 1) 
    return  (rgb * 255).cpu().numpy().astype(np.uint8)
    
def convert_to_rgb(grid):
    data = grid[0, 0, :, :]
    log_data = torch.log(data + 1)

    h,w = data.shape
    rgb = torch.zeros((h, w, 3), device=device)
    rgb[..., 0] = torch.clamp(1.5 * log_data - 0.5, 0, 1) # Red
    rgb[..., 1] = torch.clamp(1 - torch.abs(2 * log_data - 1), 0, 1) # Green
    rgb[..., 2] = torch.clamp(1.5 * (1 - log_data) - 0.5, 0, 1) # Blue
    return (rgb * 255).cpu().numpy().astype(np.uint8)
    
def visualize(buffer):
    i = 0
    for grid in buffer:
        rgb = render(grid)
        with open(f'grid/water_{i}.ppm', 'wb') as f:
            header = f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n"
            f.write(header.encode('ascii'))
            f.write(rgb.tobytes())
        i += 1

def show(buffer):
    window_name = 'Water Simulation'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 1000)
    buffer_imgs = [cv2.imread(f'grid/water_{i}.ppm') for i in range(len(buffer))]
    iteration = 0
    running = True
    while running:
        iteration += 1
        if iteration >= len(buffer_imgs):
            iteration = 0
        
        img = buffer_imgs[iteration]
        text = f"{iteration}"
        if int(text) % 10 == 0:
            print (text)
        display_img = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_CUBIC)
        cv2.imshow(window_name, display_img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            running = False
            break
        
    cv2.destroyAllWindows()

buffer, time_taken = simulate(grid)
print(f"Time taken: {time_taken} seconds")
visualize(buffer)
show(buffer)



