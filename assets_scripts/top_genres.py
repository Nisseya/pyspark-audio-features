from collections import Counter
from pathlib import Path
import csv

TSV_PATH = Path("assets/raw_30s.tsv")


def iter_rows(tsv_path: Path):
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) < 6:
                continue
            track_id, artist_id, album_id, path, duration, *tags = row
            yield {
                "track_id": track_id,
                "artist_id": artist_id,
                "album_id": album_id,
                "path": path,
                "duration": float(duration),
                "tags": tags,
            }


def extract_genres(tags: list[str]) -> list[str]:
    genres = []
    for tag in tags:
        if tag.startswith("genre---"):
            genres.append(tag.removeprefix("genre---"))
    return genres


def top_genres(tsv_path: Path, top_n: int = 30):
    counter = Counter()

    for row in iter_rows(tsv_path):
        counter.update(extract_genres(row["tags"]))

    for genre, count in counter.most_common(top_n):
        print(f"{genre}\t{count}")


if __name__ == "__main__":
    top_genres(TSV_PATH, top_n=50)