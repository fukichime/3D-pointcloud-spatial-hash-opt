import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from torch.utils.data import DataLoader

from src.models.cnn_3d_baseline import cnn_3d_baseline
from src.training.metrics_logger import metrics_logger
from src.utils.config import EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, SEED, GRID_SIZE, NUM_CLASSES

def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_cnn_model(
        train_dataset, val_dataset, num_classes = NUM_CLASSES, 
        grid_size = GRID_SIZE, dropout = 0.1,
        epochs = EPOCHS, batch_size = BATCH_SIZE, 
        lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY,
        experiment_name = None):

    set_seed()
    
    if experiment_name is None:
        experiment_name = "cnn_baseline"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\ntraining --> {experiment_name}")
    print(f"  device --> {device}")
    
    model = cnn_3d_baseline(
        num_classes = num_classes,
        grid_size = grid_size,
        dropout = dropout
    )
    model.to(device)
    
    model_memory = model.get_model_memory_mb()
    grid_memory = model.get_dense_grid_memory_mb()
    print(f"model parameters memory --> {model_memory:.2f} MB")
    print(f"input grid memory --> {grid_memory:.2f} MB")
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    
    logger = metrics_logger(experiment_name)
    logger.hash_table_memory_mb = model_memory + grid_memory
    
    best_val_acc = 0.0
    best_epoch = 0
    epoch_start = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for voxel_grids, labels in train_loader:
            voxel_grids = voxel_grids.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(voxel_grids)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for voxel_grids, labels in val_loader:
                voxel_grids = voxel_grids.to(device)
                labels = labels.to(device)
                
                logits = model(voxel_grids)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc,
                         collision_rate=0.0, hash_table_memory_mb=(model_memory + grid_memory))
        
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
    
    total_time = (time.time() - epoch_start) / 60
    logger.save_final_summary(best_val_acc, best_epoch, total_time)
    
    print(f"\nbest validation accuracy --> {best_val_acc:.4f} at epoch {best_epoch}")
    
    return best_val_acc, model, logger

