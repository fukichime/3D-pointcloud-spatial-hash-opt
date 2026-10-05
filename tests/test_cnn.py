import sys
import os
import torch
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.data.dense_dataset import dense_voxel_dataset
from src.models.cnn_3d_baseline import cnn_3d_baseline

categories = ["airplane", "car", "chair", "table", "sofa"]

dataset = dense_voxel_dataset(
    data_root =  "data/processed",
    categories = categories,
    split = "train",
    grid_size = 32
)

voxel_grid, label = dataset[0]

print(f" voxel grid shape --> {voxel_grid.shape}")
print(f" label --> {categories[label]}")
print(f" occupied voxels --> {(voxel_grid > 0).sum().item()} / {32*32*32}")

model = cnn_3d_baseline(
    num_classes = 5,
    grid_size = 32,
    in_channel = 1,
    dropout = 0.3
)

start_time = time.time()
logits = model(voxel_grid.unsqueeze(0))
forward_time = (time.time() - start_time) * 1000

print(f"  logits shape --> {logits.shape}")
print(f"  predicted class --> {torch.argmax(logits, dim=1).item()}")
print(f"  actual label --> {label}")
print(f"  forward pass time --> {forward_time:.2f} ms")

hash_table_memory = (2**14 * 8 * 4) / (1024 * 1024)  
cnn_grid_memory = (32 * 32 * 32 * 4) / (1024 * 1024)  
cnn_model_memory = model.get_model_memory_mb()
cnn_total_memory = cnn_grid_memory + cnn_model_memory

print(f" hash method (T=16384)--> {hash_table_memory:.2f} MB")
print(f" 3D CNN (dense grid) --> {cnn_grid_memory:.2f} MB")
print(f" 3D CNN model parameters --> {cnn_model_memory:.2f} MB")
print(f" 3D CNN TOTAL --> {cnn_total_memory:.2f} MB")
print(f" ratio (CNN total / Hash table) --> {cnn_total_memory / hash_table_memory:.1f}x")

