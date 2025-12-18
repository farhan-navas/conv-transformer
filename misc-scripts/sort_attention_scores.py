import ast

INFILE = "attention_scores.jsonl"
OUTFILE = "attention_scores_sorted.jsonl"

PREFIX = "Tokens+attn:"

with open(INFILE, "r", encoding="utf-8") as f, open(OUTFILE, "w", encoding="utf-8") as out:
    for line in f:
        if line.startswith(PREFIX):
            # everything after "Tokens+attn:" is a Python literal list of tuples
            payload = line[len(PREFIX):].strip()
            token_scores = ast.literal_eval(payload)
            token_scores = sorted(token_scores, key=lambda x: x[1], reverse=True)
            out.write(f"{PREFIX} {token_scores}\n")
        else:
            out.write(line)

print(f"wrote: {OUTFILE}")
