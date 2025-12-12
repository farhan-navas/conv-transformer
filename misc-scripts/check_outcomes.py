import csv
from collections import Counter

def count_values(csv_path: str, field_name: str):
    counts = Counter()

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for r in reader:
            val = r.get(field_name)
            if val is not None:
                val = val.strip()
                counts[val] += 1

    return dict(counts)

if __name__ == "__main__":
    PATH = "data.csv"
    FIELD = "Disposition"
    print(count_values(PATH, FIELD))

# Our outcomes:
# {
#     'promise to pay': 5995, 
#     'callback': 20491, 
#     'wrong number': 3514
# }, which corresponds to ~~\
# {'Promise to Pay': 5995, 'Callback': 20491, 'Wrong Number': 3514}
# 19.983% positive, 68.303% intermediate/neutral, 11.713% negative
