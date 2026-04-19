import torch
import random
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
grid = torch.zeros(1, 1, 100, 100, device=device)



kernel = torch.tensor([[0.0, 0.125, 0.0], 
                       [0.125, 0.5, 0.125], 
                       [0.0, 0.125, 0.0]], device=device) [None, None, :, :]

def drop_water(grid, x, y):
    grid[0, 0, x, y] = 10
    return grid

def spread (grid, kernel):
    new_grid = F.conv2d(grid, kernel, padding=1)
    return new_grid

def simulate(grid):
    x,y = random.randint(0, len(grid) - 1), random.randint(0, len(grid[0]) - 1)
    grid = drop_water(grid, x, y)
    t = time.time()
    for i in range(1000):
        grid = spread(grid, kernel)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return grid, time.time() - t

def convert_to_rgb(grid):
    data = grid[0, 0, :, :]
    grid_max = data.max()
    normalized = (data / (grid_max + 1e-9)).clamp(0, 1)

    h,w = data.shape
    rgb = torch.zeros((h, w, 3), device=device)
    rgb[..., 0] = torch.clamp(1.5 * normalized - 0.5, 0, 1) # Red
    rgb[..., 1] = torch.clamp(1 - torch.abs(2 * normalized - 1), 0, 1) # Green
    rgb[..., 2] = torch.clamp(1.5 * (1 - normalized) - 0.5, 0, 1) # Blue
    return (rgb * 255).cpu().numpy().astype(np.uint8)
    
def visualize(grid):
    rgb = convert_to_rgb(grid)
    with open('water.ppm', 'wb') as f:
        header = f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n"
        f.write(header.encode('ascii'))
        f.write(rgb.tobytes())
    print(f"Saved to water.ppm")

def show():
    img = Image.open('water.ppm')
    img.show()

grid, time = simulate(grid)
print(f"Time taken: {time} seconds")
gird_mean = grid[0, 0, :, :].mean()
print(f"Grid mean: {gird_mean}")
visualize(grid)
show()



