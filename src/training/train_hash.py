import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from torch.utils.data import DataLoader

from src.utils.hash_function import expected_collision_probability
from src.models.hash_model import hash_mlp_model
from src.training.metrics_logger import metrics_logger
from src.utils.config import EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, SEED, SAVE_MODELS

def set_seed(seed = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_collision_rate(indices):
    if len(indices) == 0:
        return 0.0
    unique = torch.unique(indices)
    return 1.0 - len(unique) / len(indices)

def custom_collate(batch):
    indices_list = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    k_list = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    return indices_list, labels, k_list

def train_hash_model(
        train_dataset, val_dataset, table_size, feature_dim = 8, 
        num_classes = 5, hidden_dims = [64, 32], dropout = 0.1,
        epochs = EPOCHS, batch_size = BATCH_SIZE, 
        lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY,
        experiment_name = None):

    set_seed()
    
    if experiment_name is None:
        experiment_name = f"hash_T{table_size}"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\ntraining --> {experiment_name}")
    print(f"  device --> {device}")
    print(f"  table size --> {table_size}, feature dim: {feature_dim}")
    
    model = hash_mlp_model(
        table_size = table_size,
        feature_dim = feature_dim,
        num_classes = num_classes,
        hidden_dims = hidden_dims,
        dropout = dropout
    )
    model.to(device)
    
    hash_memory_mb = (table_size * feature_dim * 4) / (1024 * 1024)
    print(f" hash table memory --> {hash_memory_mb:.2f} MB")
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = custom_collate)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, collate_fn = custom_collate)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    
    logger = metrics_logger(experiment_name)
    
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        epoch_start = time.time()
        
        for indices, labels, k_list in train_loader:
            indices_device = [idx.to(device) for idx in indices]
            labels = labels.to(device)

            batch_loss = 0.0
            batch_correct = 0
            
            for idx_tensor, label in zip(indices_device, labels):
                optimizer.zero_grad()
                logits = model(idx_tensor)
                loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
                loss.backward()
                optimizer.step()
                
                batch_loss += loss.item()
                _, predicted = torch.max(logits, 0)
                if predicted == label:
                    batch_correct += 1
                train_total += 1
            
            train_loss += batch_loss
            train_correct += batch_correct
        
        train_acc = train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        expected_collisions = []
        actual_collisions = []
        
        with torch.no_grad():
            for indices, labels, k_list in val_loader:
                for idx_tensor, label, k in zip(indices, labels, k_list):
                    idx_tensor = idx_tensor.to(device)
                    label = label.to(device)
                    
                    logits = model(idx_tensor)
                    loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(logits, 0)
                    val_total += 1
                    if predicted == label:
                        val_correct += 1
                    
                    actual_collision = compute_collision_rate(idx_tensor.cpu())
                    actual_collisions.append(actual_collision)
                    
                    expected_collision = expected_collision_probability(k, table_size)
                    expected_collisions.append(expected_collision)
        
        val_acc = val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        avg_expected_collision = sum(expected_collisions) / len(expected_collisions)
        avg_actual_collision = sum(actual_collisions) / len(actual_collisions)
        
        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, 
                        avg_actual_collision, hash_memory_mb)
        
        print(f"collision --> Expected={avg_expected_collision:.4f} vs Actual={avg_actual_collision:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            if SAVE_MODELS:
                torch.save(model.state_dict(), f"{experiment_name}_best.pt")
    
    total_time = (time.time() - epoch_start) / 60
    logger.save_final_summary(best_val_acc, best_epoch, total_time)
    
    print(f"\nbest validation accuracy --> {best_val_acc:.4f} at epoch {best_epoch}")
    
    return best_val_acc, model, logger

