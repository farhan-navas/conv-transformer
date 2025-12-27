import torch
import torch.nn.functional as F

def vqvae_loss(x: torch.Tensor, out: dict):
    """
    out is the dict returned from VQVAE.forward
    """
    x_hat = out["x_hat"]
    vq_loss = out["vq_loss"]
    cb_loss = out["cb_loss"]
    commit_loss = out["commit_loss"]

    recon = F.mse_loss(x_hat, x)
    total = recon + vq_loss

    return total, {
        "total": float(total.item()),
        "recon": float(recon.item()),
        "vq": float(vq_loss.item()),
        "cb": float(cb_loss.item()),
        "commit": float(commit_loss.item()),
    }
