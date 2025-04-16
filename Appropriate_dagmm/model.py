import torch
import torch.nn.functional as F
from torch import nn

from compression_network import CompressionNetwork
from estimation_network import EstimationNetwork
from gmm import GMM

class DAGMM(nn.Module):
    """
    Deep Autoencoding Gaussian Mixture Model using separate Compression and Estimation modules.

    Args:
        input_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent embedding.
        n_gmm_components (int): Number of Gaussian mixture components.
        comp_kwargs (dict, optional): kwargs for CompressionNetwork (__init__).
        est_kwargs (dict, optional): kwargs for EstimationNetwork (__init__).
        device (str | torch.device, optional): Device for GMM parameters.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_gmm_components: int,
        comp_kwargs: dict | None = None,
        est_kwargs: dict | None = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        # Compression autoencoder (provides encode/decode)
        ck = comp_kwargs or {}
        ck.setdefault('input_dim', input_dim)
        ck.setdefault('latent_dim', latent_dim)
        self.compression = CompressionNetwork(**ck)

        # Estimation network
        ek = est_kwargs or {}
        ek.setdefault('input_dim', latent_dim + 1)
        ek.setdefault('output_dim', n_gmm_components)
        self.estimation = EstimationNetwork(**ek)

        # GMM on features [z, recon_error, gamma]
        feat_dim = latent_dim + 1 + n_gmm_components
        self.gmm = GMM(n_gmm_components, feat_dim, device=device)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # 1) Encode
        z = self.compression.encode(x)
        # 2) Decode / reconstruct
        x_hat = self.compression.decode(z)
        # 3) Reconstruction error (per sample)
        recon_error = torch.norm(x - x_hat, p=2, dim=1, keepdim=True)
        # 4) Estimate mixture responsibilities
        gamma = self.estimation(torch.cat([z, recon_error], dim=1))
        # 5) Compute sample energy via GMM
        features = torch.cat([z, recon_error, gamma], dim=1)
        energy = -self.gmm(features)

        return {
            'x_hat': x_hat,
            'z': z,
            'recon_error': recon_error,
            'gamma': gamma,
            'energy': energy,
        }

    def loss_function(
        self,
        x: torch.Tensor,
        outputs: dict[str, torch.Tensor],
        lambda_energy: float = 0.1,
        lambda_cov: float = 0.005,
    ) -> torch.Tensor:
        """
        Loss = MSE reconstruction + lambda_energy * mean(energy) + lambda_cov * sum(cov_diag).
        """
        # MSE loss
        mse = F.mse_loss(outputs['x_hat'], x, reduction='mean')
        # Energy term
        e_mean = outputs['energy'].mean()
        # Covariance diag penalty
        cov_penalty = torch.sum(self.gmm.cov_raw) # regularize cholesky factors
        return mse + lambda_energy * e_mean + lambda_cov * cov_penalty


if __name__ == '__main__':
    # Minimal smoke test
    model = DAGMM(
        input_dim=30,
        latent_dim=10,
        n_gmm_components=5,
        comp_kwargs={'hidden_dims': [128, 64], 'activation': nn.ReLU},
        est_kwargs={'hidden_dims': [64, 32], 'activation': nn.Tanh, 'dropout': 0.5},
        device='cpu',
    )
    x = torch.randn(4, 30)
    out = model(x)
    print('z.shape=', out['z'].shape)
    print('x_hat.shape=', out['x_hat'].shape)
    loss = model.loss_function(x, out)
    print('Loss:', loss.item())
