from pathlib import Path
import csv
from collections import Counter

TSV_PATH = Path("assets/raw_30s.tsv")
OUTPUT_PATH = Path("assets/raw_30s_filtered.tsv")

WANTED_GENRES = {"folk", "rock", "electro", "classical", "hiphop"}


def extract_genres(tags: list[str]) -> set[str]:
    return {
        tag.removeprefix("genre---")
        for tag in tags
        if tag.startswith("genre---")
    }


counter = Counter()

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

        genres = extract_genres(row[5:])
        matched = genres & WANTED_GENRES

        if matched:
            writer.writerow(row)
            counter.update(matched)
            kept += 1

print(f"Saved {kept} tracks to {OUTPUT_PATH}")
print(counter)