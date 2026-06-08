import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(__file__))

from src.data.voxelization import pointcloud_to_voxel_grid

def save_voxel_visualization(voxel_grid, save_path = "voxel_sample.png"):
    grid_np = voxel_grid.cpu().numpy() if torch.is_tensor(voxel_grid) else voxel_grid
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    x, y, z = np.where(grid_np > 0.5)
    
    ax.scatter(x, y, z, c='blue', marker='s', s=40, alpha=0.6, edgecolors='navy')
    
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title(f'Voxelized Point Cloud ({grid_np.shape[0]}x{grid_np.shape[1]}x{grid_np.shape[2]})')
    
    ax.set_xlim([0, grid_np.shape[0]])
    ax.set_ylim([0, grid_np.shape[1]])
    ax.set_zlim([0, grid_np.shape[2]])
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Voxel saved to {save_path}")

if __name__ == "__main__":
    test_txt_file = "data/processed/airplane/train/airplane_0001.txt"
    output_image_path = "airplane_voxel_presentation.png"
    
    if os.path.exists(test_txt_file):
        print(f"Loading point cloud from: {test_txt_file}")
        
        points = np.loadtxt(test_txt_file)
        
        voxel_grid = pointcloud_to_voxel_grid(points, grid_size=32)
        
        total_voxels = 32 * 32 * 32
        occupied_voxels = (voxel_grid > 0).sum().item()
        print(f"Grid Shape: {voxel_grid.shape}")
        print(f"Occupied Voxels: {occupied_voxels} / {total_voxels} ({(occupied_voxels/total_voxels)*100:.2f}% dense)")
        
        save_voxel_visualization(voxel_grid, output_image_path)
    else:
        print(f"Error: {test_txt_file} not found.")