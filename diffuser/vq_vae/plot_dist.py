from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

JSONL_PATH = "sentence_cluster_ids.jsonl"
OUT_PNG = "code_usage_hist.png"
NUM_CODES = 128

def main():
    cnt = Counter()
    total = 0

    K = NUM_CODES
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
