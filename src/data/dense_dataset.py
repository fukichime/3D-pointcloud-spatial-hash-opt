import os
import torch
from torch.utils.data import Dataset
import numpy as np
from .voxelization import pointcloud_to_voxel_grid

class dense_voxel_dataset(Dataset):
    def __init__(
            self,
            data_root,
            categories,
            split = 'train',
            grid_size = 32):
        
        self.grid_size = grid_size
        self.samples = []

        for label, cat in enumerate(categories):
            folder = os.path.join(data_root, cat, split)
            if not os.path.isdir(folder):
                print(f"!!! {folder} not found!")
                continue

            txt_files = [f for f in os.listdir(folder) if f.endswith('.txt')]
            for txt_file in txt_files:
                txt_path = os.path.join(folder, txt_file)

                points = np.loadtxt(txt_path)
                if points.ndim == 1:
                    points = points.reshape(1, -1)
                
                voxel_grid = pointcloud_to_voxel_grid(points, grid_size)
                voxel_grid = voxel_grid.unsqueeze(0)

                self.samples.append((voxel_grid, label))
        print(f"{len(self.samples)} samples for split '{split}'")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
