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
grid = torch.zeros(1, 1, 100, 100, device=device)

kernel = torch.tensor([[0.0, 0.125, 0.0], 
                       [0.125, 0.5, 0.125], 
                       [0.0, 0.125, 0.0]], device=device) [None, None, :, :]

def drop_water(grid, x, y):
    grid[0, 0, x, y] = 10
    return grid

def spread (grid, kernel):
    return F.conv2d(grid, kernel, padding=1)

def simulate(grid):
    drops = [random.sample(range(100), 2) for _ in range(5)]
    buffer = []
    random_drop_time = [ random.randint(0, 1000) for _ in range(5) ] 
    print(random_drop_time)
    random_drop_time.sort()
    t = time.time()
    for i in range(1000):
        if i in random_drop_time:
            x, y = drops[random_drop_time.index(i)]
            grid = drop_water(grid, x, y)
        grid = spread(grid, kernel)
        buffer.append(grid.clone())
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return buffer, time.time() - t
def convert_to_rgb(grid):
    data = grid[0, 0, :, :]
    log_data = torch.log(data + 1)
    current_max = log_data.max().item()
    normalized = (log_data / current_max + 1e-9).clamp(0, 1)

    h,w = data.shape
    rgb = torch.zeros((h, w, 3), device=device)
    rgb[..., 0] = torch.clamp(1.5 * normalized - 0.5, 0, 1) # Red
    rgb[..., 1] = torch.clamp(1 - torch.abs(2 * normalized - 1), 0, 1) # Green
    rgb[..., 2] = torch.clamp(1.5 * (1 - normalized) - 0.5, 0, 1) # Blue
    return (rgb * 255).cpu().numpy().astype(np.uint8)
    
def visualize(buffer):
    i = 0
    for grid in buffer:
        rgb = convert_to_rgb(grid)
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
            break
        
        img = buffer_imgs[iteration]
        text = f"{iteration}"
        if int(text) % 10 == 0:
            print (text)
        cv2.imshow(window_name, img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            running = False
            break
        
    cv2.destroyAllWindows()

buffer, time_taken = simulate(grid)
print(f"Time taken: {time_taken} seconds")
visualize(buffer)
show(buffer)



