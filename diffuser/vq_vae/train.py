import os
import torch
from torch.utils.data import DataLoader

from utils import set_seed, ensure_embeddings_npy, codebook_perplexity
from dataset import EmbeddingDataset
from model import VQVAE
from losses import vqvae_loss

# put all in yaml file later
SEED = 42

JSONL_PATH = "sentence_embeddings.jsonl"
NPY_PATH = "embeddings.npy"

CKPT_DIR = "checkpoints"
CKPT_PATH = os.path.join(CKPT_DIR, "vqvae.pt")

BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3
LOG_EVERY = 100  # batches

# Model
NUM_CODES = 128
LATENT_DIM = 128
HIDDEN = 256
BETA = 0.25

L2_NORM_EMBS = True

def main():
    set_seed(SEED)

    # Build embeddings.npy if missing
    embs = ensure_embeddings_npy(NPY_PATH, l2_norm=L2_NORM_EMBS)
    n, dim_in = embs.shape
    print(f"[data] embeddings: {n} x {dim_in} (saved at {NPY_PATH})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    ds = EmbeddingDataset(NPY_PATH)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model = VQVAE(
        dim_in=dim_in,
        dim_latent=LATENT_DIM,
        num_codes=NUM_CODES,
        hidden=HIDDEN,
        beta=BETA,
    ).to(device)

    # Minimal model summary
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {model.__class__.__name__} params={num_params:,} dim_in={dim_in} latent={LATENT_DIM} codes={NUM_CODES} hidden={HIDDEN} beta={BETA}")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    os.makedirs(CKPT_DIR, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
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

            if steps % LOG_EVERY == 0:
                with torch.no_grad():
                    perp_batch = codebook_perplexity(out["indices"].detach().cpu(), NUM_CODES)
                print(
                    f"  [batch {steps}] loss={logs['total']:.6f} recon={logs['recon']:.6f} "
                    f"vq={logs['vq']:.6f} cb={logs['cb']:.6f} commit={logs['commit']:.6f} "
                    f"perplexity={perp_batch:.2f}"
                )

        avg = running / max(steps, 1)

        # quick utilization metric: perplexity on last batch indices
        with torch.no_grad():
            perp = codebook_perplexity(out["indices"].detach().cpu(), NUM_CODES)

        print(f"[epoch {epoch}] loss={avg:.6f} recon={logs['recon']:.6f} vq={logs['vq']:.6f} perplexity={perp:.2f}")

        # save best
        if avg < best_loss:
            best_loss = avg
            torch.save(
                {
                    "model": model.state_dict(),
                    "dim_in": dim_in,
                    "latent_dim": LATENT_DIM,
                    "num_codes": NUM_CODES,
                    "hidden": HIDDEN,
                    "beta": BETA,
                    "l2_norm_embs": L2_NORM_EMBS,
                },
                CKPT_PATH,
            )
            print(f"  [save] best -> {CKPT_PATH}")

if __name__ == "__main__":
    main()
