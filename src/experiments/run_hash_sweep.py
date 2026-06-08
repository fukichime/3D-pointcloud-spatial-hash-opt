import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.dataset import PrecomputedHashDataset
from src.training.train_hash import train_hash_model
from src.utils.config import CATEGORIES, GRID_SIZE, FEATURE_DIM, NUM_CLASSES, HIDDEN_DIMS, DROPOUT, EPOCHS, BATCH_SIZE, HASH_TABLE_SIZES, SAVE_MODELS

def run_hash_sweep():
    
    results = {}
    
    for T in HASH_TABLE_SIZES:
        print(f"EXPERIMENT --> T = {T}")
        
        train_dataset = PrecomputedHashDataset(
            data_root = "data/processed",
            categories = CATEGORIES,
            split = "train",
            grid_size = GRID_SIZE,
            T = T,
            use_double_hashing = False
        )
        
        val_dataset = PrecomputedHashDataset(
            data_root = "data/processed",
            categories = CATEGORIES,
            split = "test",
            grid_size = GRID_SIZE,
            T = T,
            use_double_hashing = False
        )
        
        best_acc, model, logger = train_hash_model(
            train_dataset = train_dataset,
            val_dataset = val_dataset,
            table_size = T,
            feature_dim = FEATURE_DIM,
            num_classes = NUM_CLASSES,
            hidden_dims = HIDDEN_DIMS,
            dropout = DROPOUT,
            epochs = EPOCHS,
            batch_size = BATCH_SIZE,
            experiment_name = f"hash_T{T}"
        )
        
        results[T] = {
            'best_accuracy': best_acc,
            'memory_mb': (T * FEATURE_DIM * 4) / (1024 * 1024)
        }
        
        print(f"\nT = {T} completed and best accuracy --> {best_acc:.4f}")
    
    print("SUMMARY")
    print(f"{'T':<12} {'Memory (MB)':<15} {'Best Accuracy':<15}")
    for T, res in results.items():
        print(f"{T:<12} {res['memory_mb']:<15.2f} {res['best_accuracy']:<15.4f}")
    
    return results

if __name__ == "__main__":
    results = run_hash_sweep()