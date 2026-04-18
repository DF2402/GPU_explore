import torch
import random
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
import numpy as np
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

def visualize(grid):
    data = grid.detach().cpu().numpy() [0, 0, :, :]
    grid_max = data.max()
    display = data / grid_max
    plt.imshow(display, cmap='Blues', vmin=0, vmax=1)
    plt.show()

grid, time = simulate(grid)
print(f"Time taken: {time} seconds")
gird_mean = grid[0, 0, :, :].mean()
print(f"Grid mean: {gird_mean}")
visualize(grid)




