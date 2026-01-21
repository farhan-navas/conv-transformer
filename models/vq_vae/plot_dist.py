import json
from collections import Counter
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml

CONFIG_PATH = Path("config.yaml")
RUN_VERSION_DEFAULT = "v0.1"
OUTPUT_ROOT_DEFAULT = Path("runs")
JSONL_REL_DEFAULT = Path("vqvae") / "sentence_cluster_ids.jsonl"
OUT_PNG = "code_usage_hist.png"
NUM_CODES = 128


def resolve_jsonl_path(cfg_path: Path = CONFIG_PATH) -> Path:
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    core = cfg.get("core", {})
    run_version = core.get("run_version", RUN_VERSION_DEFAULT)
    output_root = Path(core.get("output_root", OUTPUT_ROOT_DEFAULT))
    export_cfg = cfg.get("export_codes", {})

    jsonl_override = export_cfg.get("jsonl_out")
    if jsonl_override:
        jsonl_path = Path(jsonl_override)
        if not jsonl_path.is_absolute():
            jsonl_path = output_root / run_version / "vqvae" / jsonl_path.name
    else:
        jsonl_path = output_root / run_version / JSONL_REL_DEFAULT

    return jsonl_path

def resolve_num_codes(cfg_path: Path = CONFIG_PATH) -> int:
    cfg = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    vqvae = cfg.get("vqvae", {})
    return int(vqvae.get("num_codes", NUM_CODES))

def load_counts(jsonl_path: Path) -> Tuple[Counter, int]:
    cnt: Counter[int] = Counter()
    total = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for speaker in obj.get("speakers", []):
                ids = speaker.get("cluster_ids", []) or []
                for cid in ids:
                    if isinstance(cid, int):
                        cnt[cid] += 1
                        total += 1

    return cnt, total

def main():
    jsonl_path = resolve_jsonl_path()
    K = resolve_num_codes()
    cnt, total = load_counts(jsonl_path)

    counts = np.zeros(K, dtype=np.int64)
    for k, v in cnt.items():
        if 0 <= k < K:
            counts[k] = v

    used = int((counts > 0).sum())

    print(f"Total cluster_id tokens: {total}")
    print(f"Num codes (K): {K}")
    print(f"Used codes: {used} ({used / K:.2%})")

    # Top / bottom
    top10 = counts.argsort()[::-1][:10]
    print("\nTop 10 most common codes:")
    for k in top10:
        print(f"  code {k:4d}: {counts[k]}")

    nonzero = np.where(counts > 0)[0]
    if len(nonzero) > 0:
        rare10 = nonzero[np.argsort(counts[nonzero])[:10]]
        print("\nTop 10 rarest (non-zero) codes:")
        for k in rare10:
            print(f"  code {k:4d}: {counts[k]}")

    # Plot histogram (bar chart across code indices)
    plt.figure()
    plt.bar(np.arange(K), counts)
    plt.title("Code usage histogram")
    plt.xlabel("code id")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    print(f"\nSaved plot to: {OUT_PNG}")

if __name__ == "__main__":
    main()
