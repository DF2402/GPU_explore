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

kernel = torch.tensor([[0.05, 0.20, 0.05], 
                       [0.20, 0.00, 0.20], 
                       [0.05, 0.20, 0.05]], device=device) [None, None, :, :]

def drop_water(grid, cx, cy):
    h, w = grid.shape[2], grid.shape[3]
    y, x = torch.meshgrid(torch.arange(h, device=grid.device), 
                          torch.arange(w, device=grid.device), indexing='ij')
    radius = 3.0
    dist_sq = (x - cx)**2 + (y - cy)**2
    blob = torch.exp(-dist_sq / (2 * radius**2))
    grid[0, 0, :, :] += blob * 5.0
    return grid
blur_kernel = torch.tensor([[0.05, 0.10, 0.05], 
                            [0.10, 0.40, 0.10], 
                            [0.05, 0.10, 0.05]], device=device)[None, None, :, :]
def spread (prev_grid, grid, kernel):
    neighbors_sum = F.conv2d(grid, kernel, padding=1)
    next_grid = neighbors_sum * 2.0 - prev_grid
    next_grid *= 0.98
    blurred_grid = F.conv2d(next_grid, blur_kernel, padding=1)
    viscosity = 0.15
    next_grid = next_grid * (1.0 - viscosity) + blurred_grid * viscosity
    return next_grid

def simulate(grid):
    drops = [random.sample(range(resolution), 2) for _ in range(10)]
    buffer = []
    random_drop_time = [ random.randint(0, 1000) for _ in range(10) ] 
    print(random_drop_time)
    random_drop_time.sort()
    t = time.time()
    prev_grid = torch.zeros_like(grid)
    for i in range(1200):
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
        rgb = render_modern(grid)
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
            pass
        display_img = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_CUBIC)
        cv2.imshow(window_name, display_img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            running = False
            break
        
    cv2.destroyAllWindows()

def render_modern(grid):
    smooth_kernel = torch.ones(1, 1, 5, 5, device=grid.device) / 25.0
    smooth_grid = F.conv2d(grid, smooth_kernel, padding=2)
    data = smooth_grid[0, 0, :, :]
    dy, dx = torch.gradient(data)
    slope_strength = 25.0
    dx *= slope_strength
    dy *= slope_strength
    mag = torch.sqrt(dx**2 + dy**2 + 1.0)
    nx, ny, nz = -dx/mag, -dy/mag, 1.0/mag
    
    h, w = nx.shape
    y_coords, x_coords = torch.meshgrid(torch.linspace(0, 1, h, device=grid.device),torch.linspace(0, 1, w, device=grid.device), indexing='ij')
    
    #Refraction
    refract_scale = 0.15 
    rx = torch.clamp(x_coords + nx * refract_scale, 0, 1)
    ry = torch.clamp(y_coords + ny * refract_scale, 0, 1)


    base_r = rx * 0.2 + 0.7     
    base_g = torch.full_like(rx, 0.75) 
    base_b = ry * 0.2 + 0.8   

    light_dir = torch.tensor([-0.5, -0.5, 0.7], device=grid.device) 
    light_dir = light_dir / torch.norm(light_dir)
    view_dir = torch.tensor([0.0, 0.0, 1.0], device=grid.device) 

    half_vec = (light_dir + view_dir)
    half_vec = half_vec / torch.norm(half_vec)

    spec_dot = torch.clamp(nx*half_vec[0] + ny*half_vec[1] + nz*half_vec[2], 0.0, 1.0)
    specular = spec_dot ** 64.0

    ao = torch.clamp(nz, 0.8, 1.0)

    rgb = torch.zeros(h, w, 3, device=grid.device)
    rgb = torch.zeros(h, w, 3, device=grid.device)
    
    raw_r = base_r * ao + specular * 0.8
    raw_g = base_g * ao + specular * 0.8
    raw_b = base_b * ao + specular * 0.9

    rgb[..., 0] = torch.tanh(raw_r)
    rgb[..., 1] = torch.tanh(raw_g)
    rgb[..., 2] = torch.tanh(raw_b)
    return (rgb * 255).cpu().numpy().astype(np.uint8)

def save_to_gif(buffer, filename="water_simulation.gif", skip_frames=2):
    frames = []

    for i in range(0, len(buffer), skip_frames):
        grid = buffer[i]
        
        rgb_array = render_modern(grid)

        img = Image.fromarray(rgb_array)
        frames.append(img)

    frames[0].save(filename, 
                   save_all=True, 
                   append_images=frames[1:], 
                   optimize=True, 
                   duration=30, 
                   loop=0) 
    
buffer, time_taken = simulate(grid)
print(f"Time taken: {time_taken} seconds")
save_to_gif(buffer)



