import sys
import os
import torch

sys.path.insert(0, os.path.dirname(__file__))
from src.data.dataset import PrecomputedHashDataset

categories = ["airplane", "car", "chair", "table", "sofa"]

dataset = PrecomputedHashDataset(
    data_root="data/processed",
    categories=categories,
    split="train",
    grid_size=32,
    T=2**14,
    use_double_hashing=False
)

print(f"\n train dataset loaded: {len(dataset)} samples")

if len(dataset) > 0:
    indices, label = dataset[0]
    print(f"Label: {categories[label]}")
    print(f"Hash indices count: {len(indices)}")
    print(f"Unique hash indices: {len(torch.unique(indices))}")
    print(f"Collision rate: {(1 - len(torch.unique(indices))/len(indices)):.3f}")

try:
    test_dataset = PrecomputedHashDataset(
        data_root="data/processed",
        categories=categories,
        split="test",
        grid_size=32,
        T=2**14,
        use_double_hashing=False
    )
    print(f"\n test dataset loaded: {len(test_dataset)} samples")
    
    if len(test_dataset) > 0:
        indices, label = test_dataset[0]
        print(f"First test label: {categories[label]}")
        print(f"Test hash indices: {len(indices)}")
        
except Exception as e:
    print(f"Error loading test dataset: {e}")

for cat in categories:
    test_folder = os.path.join("data/processed", cat, "test")
    if os.path.exists(test_folder):
        txt_files = [f for f in os.listdir(test_folder) if f.endswith('.txt')]
        print(f"{cat}/test: {len(txt_files)} .txt files")
    else:
        print(f"{cat}/test: folder not found")
