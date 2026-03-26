from pathlib import Path
import csv
from collections import Counter, defaultdict

TSV_PATH = Path("assets/raw_30s.tsv")
OUTPUT_PATH = Path("assets/raw_30s_filtered.tsv")

WANTED_GENRES = {"folk", "rock", "electronic", "classical", "hiphop"}
MAX_PER_CLASS = 700


def first_genre(tags: list[str]):
    for tag in tags:
        if tag.startswith("genre---"):
            return tag.removeprefix("genre---")
    return None


counter = Counter()
per_class = defaultdict(int)

with TSV_PATH.open("r", encoding="utf-8", newline="") as src, OUTPUT_PATH.open(
    "w", encoding="utf-8", newline=""
) as dst:
    reader = csv.reader(src, delimiter="\t")
    writer = csv.writer(dst, delimiter="\t")

    header = next(reader)
    writer.writerow(header)

    kept = 0

    for row in reader:
        if len(row) < 6:
            continue

        genre = first_genre(row[5:])
        if genre in WANTED_GENRES and per_class[genre] < MAX_PER_CLASS:
            writer.writerow(row)
            counter.update([genre])
            per_class[genre] += 1
            kept += 1

        if all(per_class[g] >= MAX_PER_CLASS for g in WANTED_GENRES):
            break

print(f"Saved {kept} tracks to {OUTPUT_PATH}")
print(counter)