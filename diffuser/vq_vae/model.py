import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, dim_in: int, dim_latent: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim_latent),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, dim_latent: int, dim_out: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim_out),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class VectorQuantizer(nn.Module):
    """
    Standard VQ layer with straight-through estimator.

    Codebook is an nn.Embedding of size (K, dim_latent).
    """
    def __init__(self, num_codes: int, dim_latent: int, beta: float = 0.25):
        super().__init__()
        self.num_codes = num_codes
        self.dim_latent = dim_latent
        self.beta = beta
        self.codebook = nn.Embedding(num_codes, dim_latent)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z_e: torch.Tensor):
        """
        z_e: [B, d]
        Returns:
          z_q_st: [B, d] (quantized with straight-through)
          indices: [B]
          vq_loss: scalar tensor (codebook + commitment)
          cb_loss, commit_loss: scalar tensors
        """
        # Compute squared L2 distance to each codebook vector
        # dist(b,k) = ||z_e||^2 + ||e_k||^2 - 2 z_e · e_k
        z2 = (z_e ** 2).sum(dim=1, keepdim=True)               # [B, 1]
        e = self.codebook.weight                                # [K, d]
        e2 = (e ** 2).sum(dim=1).unsqueeze(0)                   # [1, K]
        ze = z_e @ e.t()                                        # [B, K]
        dist = z2 + e2 - 2 * ze                                 # [B, K]

        indices = torch.argmin(dist, dim=1)                     # [B]
        z_q = self.codebook(indices)                            # [B, d]

        # Losses
        cb_loss = F.mse_loss(z_q, z_e.detach())                 # codebook learns toward encoder outputs
        commit_loss = F.mse_loss(z_e, z_q.detach())             # encoder commits to chosen codes
        vq_loss = cb_loss + self.beta * commit_loss

        # Straight-through
        z_q_st = z_e + (z_q - z_e).detach()

        return z_q_st, indices, vq_loss, cb_loss, commit_loss

class VQVAE(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_latent: int,
        num_codes: int,
        hidden: int = 256,
        beta: float = 0.25,
    ):
        super().__init__()
        self.encoder = Encoder(dim_in, dim_latent, hidden=hidden)
        self.vq = VectorQuantizer(num_codes, dim_latent, beta=beta)
        self.decoder = Decoder(dim_latent, dim_in, hidden=hidden)

    def forward(self, x: torch.Tensor):
        """
        x: [B, D]
        returns dict with:
          x_hat, indices, vq_loss, cb_loss, commit_loss
        """
        z_e = self.encoder(x)                                   # [B, d]
        z_q, indices, vq_loss, cb_loss, commit_loss = self.vq(z_e)
        x_hat = self.decoder(z_q)                               # [B, D]
        return {
            "x_hat": x_hat,
            "indices": indices,
            "vq_loss": vq_loss,
            "cb_loss": cb_loss,
            "commit_loss": commit_loss,
        }
