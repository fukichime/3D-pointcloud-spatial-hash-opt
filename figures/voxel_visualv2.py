import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(__file__))
from src.data.voxelization import pointcloud_to_voxel_grid

def save_voxel_visualization(voxel_grid, save_path="voxel_sample.png", show_grid=True):
    grid_np = voxel_grid.cpu().numpy() if torch.is_tensor(voxel_grid) else voxel_grid
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    x, y, z = np.where(grid_np > 0.5)
    
    ax.scatter(x, y, z, c='blue', marker='s', s=30, alpha=0.7, edgecolors='navy', linewidth=0.5)
    
    if show_grid:
        grid_range = range(0, grid_np.shape[0] + 1, 4)
        for i in grid_range:
            ax.plot([i, i], [0, grid_np.shape[1]], [0, 0], color='gray', alpha=0.2, linewidth=0.5)
            ax.plot([i, i], [0, grid_np.shape[1]], [grid_np.shape[2], grid_np.shape[2]], color='gray', alpha=0.2, linewidth=0.5)
        for j in grid_range:
            ax.plot([0, grid_np.shape[0]], [j, j], [0, 0], color='gray', alpha=0.2, linewidth=0.5)
            ax.plot([0, grid_np.shape[0]], [j, j], [grid_np.shape[2], grid_np.shape[2]], color='gray', alpha=0.2, linewidth=0.5)
    
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title(f'Voxelized Point Cloud (32×32×32)\nOccupied: {len(x)} voxels')
    
    ax.set_xlim([0, grid_np.shape[0]])
    ax.set_ylim([0, grid_np.shape[1]])
    ax.set_zlim([0, grid_np.shape[2]])
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Voxel saved to {save_path}")

if __name__ == "__main__":
    test_txt_file = "data/processed/bookshelf/train/bookshelf_0001.txt"
    output_image_path = "bookshelf_voxel_presentation.png"
    
    if os.path.exists(test_txt_file):
        print(f"Loading point cloud from: {test_txt_file}")
        points = np.loadtxt(test_txt_file)
        
        voxel_grid = pointcloud_to_voxel_grid(points, grid_size=32)
        
        total_voxels = 32 * 32 * 32
        occupied_voxels = (voxel_grid > 0).sum().item()
        print(f"Grid Shape: {voxel_grid.shape}")
        print(f"Occupied Voxels: {occupied_voxels} / {total_voxels} ({(occupied_voxels/total_voxels)*100:.2f}% dense)")
        
        save_voxel_visualization(voxel_grid, output_image_path, show_grid=True)
    else:
        print(f"Error: {test_txt_file} not found.")