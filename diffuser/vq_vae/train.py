from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from diffuser.vq_vae.utils import set_seed, ensure_embeddings_npy, codebook_perplexity
from diffuser.vq_vae.dataset import EmbeddingDataset
from diffuser.vq_vae.model import VQVAE
from diffuser.vq_vae.losses import vqvae_loss

@dataclass
class VQVAEConfig:
    jsonl_path: str = "sentence_embeddings.jsonl"
    npy_path: str = "embeddings.npy"
    checkpoint_path: str = "checkpoints/vqvae.pt"
    batch_size: int = 256
    epochs: int = 10
    learning_rate: float = 1e-3
    log_every: int = 100
    num_codes: int = 128
    latent_dim: int = 128
    hidden: int = 256
    beta: float = 0.25
    l2_norm_embs: bool = True
    seed: int = 42


def train_vqvae(config: VQVAEConfig) -> None:
    set_seed(config.seed)

    ckpt_path = Path(config.checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    embs = ensure_embeddings_npy(config.npy_path, jsonl_path=config.jsonl_path, l2_norm=config.l2_norm_embs)
    n, dim_in = embs.shape
    print(f"[data] embeddings: {n} x {dim_in} (saved at {config.npy_path})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    ds = EmbeddingDataset(config.npy_path)
    dl = DataLoader(ds, batch_size=config.batch_size, shuffle=True, drop_last=False)

    model = VQVAE(
        dim_in=dim_in,
        dim_latent=config.latent_dim,
        num_codes=config.num_codes,
        hidden=config.hidden,
        beta=config.beta,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(
        f"[model] {model.__class__.__name__} params={num_params:,} dim_in={dim_in} "
        f"latent={config.latent_dim} codes={config.num_codes} hidden={config.hidden} beta={config.beta}"
    )

    opt = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        steps = 0

        for x in dl:
            x = x.to(device)

            out = model(x)
            loss, logs = vqvae_loss(x, out)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += loss.item()
            steps += 1

            if steps % config.log_every == 0:
                with torch.no_grad():
                    perp_batch = codebook_perplexity(out["indices"].detach().cpu(), config.num_codes)
                print(
                    f"  [batch {steps}] loss={logs['total']:.6f} recon={logs['recon']:.6f} "
                    f"vq={logs['vq']:.6f} cb={logs['cb']:.6f} commit={logs['commit']:.6f} "
                    f"perplexity={perp_batch:.2f}"
                )

        avg = running / max(steps, 1)

        with torch.no_grad():
            perp = codebook_perplexity(out["indices"].detach().cpu(), config.num_codes)

        print(f"[epoch {epoch}] loss={avg:.6f} recon={logs['recon']:.6f} vq={logs['vq']:.6f} perplexity={perp:.2f}")

        if avg < best_loss:
            best_loss = avg
            torch.save(
                {
                    "model": model.state_dict(),
                    "dim_in": dim_in,
                    "latent_dim": config.latent_dim,
                    "num_codes": config.num_codes,
                    "hidden": config.hidden,
                    "beta": config.beta,
                    "l2_norm_embs": config.l2_norm_embs,
                },
                ckpt_path,
            )
            print(f"  [save] best -> {ckpt_path}")

