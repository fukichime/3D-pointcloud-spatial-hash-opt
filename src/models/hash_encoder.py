import torch
import torch.nn as nn
import time

class hash_encoder(nn.Module):
    def __init__(self, table_size, feature_dim = 8):

        super().__init__()
        self.table_size = table_size
        self.feature_dim = feature_dim

        #initialized
        self.hash_table = nn.Parameter(torch.randn(table_size, feature_dim) * 0.01)

        #memory calculation
        memory_bytes = table_size * feature_dim * 4
        memory_mb = memory_bytes / (1024 * 1024)
        print(f"hash encoder --> T = {table_size}, feature dimension = {feature_dim}")
        print(f"memory --> {memory_mb:.2f} MB")

    def forward(self, indices):
        start_time = time.time()

        features = self.hash_table[indices.to(self.hash_table.device)]

        #aggregation by mean
        global_feature = features.mean(dim=0)
        forward_time = (time.time() - start_time) * 1000
        #print(f"hash encoder forward time --> {forward_time:.2f} ms")

        return global_feature
    
    def get_memory_mb(self):
        return (self.table_size * self.feature_dim * 4) / (1024 * 1024) 
