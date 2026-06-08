import numpy as np
import torch

def pointcloud_to_voxel_grid(points, grid_size = 32):
    #origin centerins -> normalize at -1/1 -> map to indices -> binary grid
    
    centeroid = np.mean(points, axis = 0)
    centered = points - centeroid

    max_coord = np.max(np.abs(centered))
    if max_coord < 1e-6:
        max_coord = 1.0
    
    normalized = centered / max_coord

    indices = ((normalized + 1) / 2 * (grid_size - 1)).astype(int)
    indices = np.clip(indices, 0, grid_size - 1)

    grid = np.zeros((grid_size, grid_size, grid_size), dtype = np.float32)
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0

    return torch.tensor(grid)

if __name__ == "__main__":
    dummy_point = np.array([
        [-1, -1, -1], [ 1, -1, -1], [-1,  1, -1], [ 1,  1, -1],
        [-1, -1,  1], [ 1, -1,  1], [-1,  1,  1], [ 1,  1,  1]       
    ])
    grid = pointcloud_to_voxel_grid(dummy_point, grid_size = 8)
    print(f"voxel grid shape: {grid.shape}")
    print(f"occupied voxels: {(grid > 0).sum().item()}")

