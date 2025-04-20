import torch
from torch import nn

class CompressionNetwork(nn.Module):
    def __init__(self, input_dim, latent_dim=2, hidden_dims=[60, 30, 10], activation=nn.Tanh):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, latent_dim))  # final encoder layer
        self.encoder = nn.Sequential(*layers)

        # decoder (reverse hidden_dims)
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(activation())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))  # reconstruct to original
        self.decoder = nn.Sequential(*decoder_layers)

        self._reconstruction_loss = nn.MSELoss()

    def forward(self, x):
        # Input: waveform of size (batch_size, num_samples)
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def reconstruction_loss(self, x, x_reconstructed):
        return self._reconstruction_loss(x, x_reconstructed)
