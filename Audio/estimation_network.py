import torch.nn as nn

class EstimationNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[10, 4], dropout=0.3, activation=nn.ReLU):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation())
            layers.append(nn.Dropout(p=dropout))
            prev_dim = h_dim
        # Final projection to K logits (number of Gaussians), followed by Softmax for probabilities
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Softmax(dim=-1))  # To output probabilities
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # Shape: [B, K], where B is the batch size, and K is the number of Gaussians.
