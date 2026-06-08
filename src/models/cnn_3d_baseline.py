import torch
import torch.nn as nn
import time

class cnn_3d_baseline(nn.Module):
    def __init__(
            self,
            num_classes = 10,
            grid_size = 32,
            in_channel = 1,
            dropout = 0.1
    ):
        super().__init__()

        self.num_classes = num_classes
        self.grid_size = grid_size

        self.conv1 = nn.Conv3d(in_channel, 16, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 64, kernel_size = 3, padding = 1)
        self.bn3 = nn.BatchNorm3d(64)

        self.pool = nn.MaxPool3d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout3d(dropout)

        #after 3 pooling 32 > 16 > 8 > 4
        #feat map size 64 channel x 4x4x4 --> 4096
        conv_output_size = 64 * 4 * 4 * 4

        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.dropout_fc = nn.Dropout(dropout)

        memory_mb = self.get_dense_grid_memory_mb(batch_size = 1)
        print(f" cnn 3d baseline --> grid = {grid_size}^3, classes = {num_classes}")
        print(f" memory on dense grid --> {memory_mb:.2f} MB per batch")

        total_params = sum(p.numel() for p in self.parameters())
        print(f"params --> {total_params:,}")
    
    def forward(self, x):
        start_time = time.time()

        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.relu(self.fc2(x))
        x = self.dropout_fc(x)
        logits = self.fc3(x)
        
        forward_time = (time.time() - start_time) * 1000
        #print(f" cnn forward time --> {forward_time:.2f} ms")

        return logits
    
    def get_dense_grid_memory_mb(self, batch_size = 1):
        per_sample_mb = (self.grid_size ** 3 * 4) / (1024 * 1024)
        return per_sample_mb * batch_size
    
    def get_model_memory_mb(self):
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
    
    def get_total_memory_mb(self, batch_size = 1):
        return self.get_dense_grid_memory_mb(batch_size) + self.get_model_memory_mb()