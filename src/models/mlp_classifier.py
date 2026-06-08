import torch
import torch.nn as nn
import time

class mlp_classifier(nn.Module):
    def __init__(
            self, 
            input_dim = 8, 
            num_classes = 10, 
            hidden_dim = [64, 32], 
            dropout = 0.1):
        
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dim:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

        total_params =  sum(p.numel() for p in self.parameters())
        print(f" mlp classifier --> input dimensions = {input_dim}, hidden = {hidden_dim}, classes = {num_classes}")
        print(f"params = {total_params:,}")

    def forward(self, x):
        start_time = time.time()

        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        logits = self.mlp(x)

        forward_time = (time.time() - start_time) * 1000
        #print(f"mlp forward time --> {forward_time:.2f} ms")

        return logits.squeeze(0)