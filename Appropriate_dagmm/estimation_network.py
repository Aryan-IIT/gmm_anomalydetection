import torch
from torch import nn

class EstimationNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 32], dropout=0.5, activation=nn.Tanh):
         # D + 1 features (latent_dim plus recon_error)
         #output_dim same as number of gaussians
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation())
            layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim
        # Final projection to K logits, then Softmax to gamma (responsibilities)
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Softmax(dim=-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x) # shape: [B, K], where B is batch size and K is number of gaussians. Hence one responsibility per gaussian per sample.
