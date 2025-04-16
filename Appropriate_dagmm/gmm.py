import torch, torch.nn as nn, torch.nn.functional as F

class GMM(nn.Module):
    """
    A vectorised, differentiable—or EM—Gaussian Mixture.
    Use `requires_grad=False` if you want pure EM updates.
    """
    def __init__(self, n_components: int, embed_dim: int, device=None):
        super().__init__()
        self.K, self.D = n_components, embed_dim
        device = device or "cpu"
        self.register_buffer("eps", torch.tensor(1e-8))
        # Init
        self.pi      = nn.Parameter(torch.ones(self.K, device=device) / self.K)
        self.mu      = nn.Parameter(torch.randn(self.K, self.D, device=device))
        self.cov_raw = nn.Parameter(
            torch.eye(self.D, device=device).repeat(self.K, 1, 1)
        )  # will be forced PSD

    @property
    def cov(self):
        # guaranteed PSD (diagonal plus small jitter)
        eye = torch.eye(self.D, device=self.cov_raw.device).unsqueeze(0)
        return self.cov_raw @ self.cov_raw.transpose(-1, -2) + 1e-6 * eye

    def forward(self, z):
        """
        z: [B, D]
        returns log‑likelihood per sample  [B]
        """
        mvn   = torch.distributions.MultivariateNormal(self.mu, self.cov)
        log_p = mvn.log_prob(z.unsqueeze(1))           # [B, K]
        log_p += torch.log_softmax(self.pi, 0)          # mixing weights
        return torch.logsumexp(log_p, dim=1)            # [B]

    # --- EM update (call inside `torch.no_grad()` if you want manual updates)
    def em_step(self, z, gamma):
        """
        z:     [B, D] latent vectors
        gamma: [B, K] responsibilities from EstimationNet (softmax output)
        """
        Nk = gamma.sum(dim=0) + self.eps                # [K]
        self.pi.data = (Nk / Nk.sum()).detach()

        mu_new = (gamma.T @ z) / Nk.unsqueeze(1)        # [K,D]
        self.mu.data = mu_new.detach()

        z_centered = z.unsqueeze(1) - mu_new            # [B,K,D]
        cov_new = torch.einsum("bk,bkd,bke->kde", gamma, z_centered, z_centered)
        cov_new /= Nk.view(-1,1,1)
        self.cov_raw.data = torch.linalg.cholesky(
            cov_new + 1e-6 * torch.eye(self.D, device=z.device)
        ).detach()
