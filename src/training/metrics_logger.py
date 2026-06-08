import csv
import os
import time
from datetime import datetime

class metrics_logger:
    def __init__(self, experiment_name, output_dir = "experiment_results"):

        self.experiment_name = experiment_name
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok = True)

        self.filepath = os.path.join(output_dir, f"{experiment_name}.csv")
        self.start_time = time.time()

        with open(self.filepath, 'w', newline = '') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'epoch', 'train_loss', 'train_acc', 
                'val_loss', 'val_acc', 'collision_rate', 
                'hash_table_memory_mb', 'epoch_time_seconds'
                ])
    
    def log_epoch(
            self, epoch, train_loss, train_acc,
            val_loss, val_acc,
            collision_rate = 0.0,
            hash_table_memory_mb = 0.0
    ):
        epoch_time = time.time() - self.start_time
        self.start_time = time.time()

        with open(self.filepath, 'a', newline = '') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                epoch,
                f"{train_loss:.4f}",
                f"{train_acc:.4f}",
                f"{val_loss:.4f}",
                f"{val_acc:.4f}",
                f"{collision_rate:.4f}",
                f"{hash_table_memory_mb:.2f}",
                f"{epoch_time:.2f}"
            ])
        
        print(f"Epoch {epoch:3d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Coll: {collision_rate:.3f}, Time: {epoch_time:.1f}s")
    
    def save_final_summary(self, best_val_acc, best_epoch, total_time_minutes):
        summary_path = os.path.join(self.output_dir, f"{self.experiment_name}_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write(f"Experiment --> {self.experiment_name}\n")
            f.write(f"Best validation accuracy --> {best_val_acc:.4f} at epoch {best_epoch}\n")
            f.write(f"Total training time --> {total_time_minutes:.2f} minutes\n")
            f.write(f"CSV saved to --> {self.filepath}\n")
