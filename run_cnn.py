import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.data.dense_dataset import dense_voxel_dataset
from src.training.train_cnn import train_cnn_model
from src.utils.config import CATEGORIES, GRID_SIZE, NUM_CLASSES, EPOCHS

train_ds = dense_voxel_dataset("data/processed", CATEGORIES, "train", GRID_SIZE)
val_ds = dense_voxel_dataset("data/processed", CATEGORIES, "test", GRID_SIZE)

print(f"Train samples: {len(train_ds)}, Test samples: {len(val_ds)}")

best_acc, model, logger = train_cnn_model(
    train_dataset=train_ds,
    val_dataset=val_ds,
    num_classes=NUM_CLASSES,
    grid_size=GRID_SIZE,
    epochs=EPOCHS
)

print(f"\nCNN Baseline complete! Best accuracy: {best_acc:.4f}")