import sys
import os
import torch
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.data.dataset import PrecomputedHashDataset
from src.models.hash_model import hash_mlp_model

categories = ["airplane", "car", "chair", "table", "sofa"]

dataset = PrecomputedHashDataset(
    data_root = "data/processed",
    categories = categories,
    split = "train",
    grid_size = 32,
    T = 2**14,
    use_double_hashing = False
)

indices, label, k = dataset[0]
print(f" sample --> {len(indices)} occupied voxel and label = {categories[label]}")

model = hash_mlp_model(
    table_size = 2 ** 14,
    feature_dim = 8,
    num_classes = 5,
    hidden_dims = [64, 32],
    dropout = 0.3
)

start_time = time.time()

logits = model(indices)
forward_time = (time.time() - start_time) * 1000

print(f"logit shape --> {logits.shape}")
print(f"predicted class ->> {torch.argmax(logits).item()}")
print(f"actual label --> {label}")
print(f"forward pass time --> {forward_time:.2f} ms")

memory_mb = model.get_hash_table_memory()
print(f"hash table memory used --> {memory_mb:.2f} MB")