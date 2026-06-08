import os
import sys
import torch
from torch.utils.data import Dataset
import numpy as np

from src.data.voxelization import pointcloud_to_voxel_grid
from src.utils.hash_function import PRIMES, PRIMES2

class PrecomputedHashDataset(Dataset):
    def __init__(self, data_root, categories, split = 'train',
                 grid_size = 32, T = 2**14, feature_dim = 8,
                 use_double_hashing = False, max_probes = 5):
        
        self.T = T
        self.use_double_hashing = use_double_hashing
        self.max_probes = max_probes
        self.samples = []

        p1, p2, p3 = PRIMES['p1'], PRIMES['p2'], PRIMES['p3']
        q1, q2, q3 = PRIMES2['q1'], PRIMES2['q2'], PRIMES2['q3']

        for label, category in enumerate(categories):
            folder = os.path.join(data_root, category, split)
            if not os.path.isdir(folder):
                continue

            for txt_file in [f for f in os.listdir(folder) if f.endswith('.txt')]:
                points = np.loadtxt(os.path.join(folder, txt_file))
                if points.ndim == 1: points = points.reshape(1, -1)

                voxel_grid = pointcloud_to_voxel_grid(points, grid_size)
                occ = torch.nonzero(voxel_grid)
                
                k = len(occ)

                if k == 0:
                    continue

                occ_np = occ.cpu().numpy().astype(np.int64)
                x, y, z = occ_np[:, 0], occ_np[:, 1], occ_np[:, 2]

                h1 = (x * p1) ^ (y * p2) ^ (z * p3)
                
                if not use_double_hashing:
                    indices_tensor = torch.tensor(h1 % T, dtype=torch.long)
                else:
                    h2 = (x * q1) ^ (y * q2) ^ (z * q3)
                    indices_list = []
                    used = set()
                    for i in range(len(h1)):
                        slot = h1[i] % T
                        step = 0
                        while slot in used and step < max_probes:
                            slot = (slot + h2[i]) % T
                            step += 1
                        used.add(slot)
                        indices_list.append(slot)
                    indices_tensor = torch.tensor(indices_list, dtype=torch.long)
                
                self.samples.append((indices_tensor, label, k))
        
        print(f"Loaded {len(self.samples)} samples for split '{split}'")
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        indices, label, k = self.samples[idx]
        return indices, label, k

def variable_length_collate(batch):
    indices_list = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype = torch.long)
    k_list = [item[2] for item in batch]
    return indices_list, labels, k_list

def compute_collision_rate(indices):
    if len(indices) == 0: return 0.0
    unique = torch.unique(indices)
    return 1.0 - len(unique) / len(indices)
