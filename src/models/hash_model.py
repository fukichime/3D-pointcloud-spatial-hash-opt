import torch
import torch.nn as nn
import time 
from .hash_encoder import hash_encoder
from .mlp_classifier import mlp_classifier

class hash_mlp_model(nn.Module):
    def __init__(
            self, 
            table_size, 
            feature_dim = 8,
            num_classes = 10,
            hidden_dims = [64, 32],
            dropout = 0.1):
        
        super().__init__()

        self.table_size = table_size
        self.feature_dim = feature_dim

        self.encoder = hash_encoder(table_size, feature_dim)
        self.classifier = mlp_classifier(feature_dim, num_classes, hidden_dims, dropout)

        encoder_memory = self.encoder.get_memory_mb()
        print(f"total model memory --> {encoder_memory:.2f} MB but hash table only")

    def forward(self, indices):
            start_time = time.time()

            features = self.encoder(indices)
            logits = self.classifier(features)

            total_time = (time.time() - start_time) * 1000

            return logits
        
    def get_hash_table_memory(self):
            return self.encoder.get_memory_mb()
        