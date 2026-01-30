import csv
import hashlib
import os
import random
import shutil
import tarfile
import tempfile
from pathlib import Path
import sys
import requests
from tqdm import tqdm

# -------------------------
# CONSTANTS
# -------------------------

SEED = 42

DATASET = "raw_30s"
DATA_TYPE = "audio-low"

BASE_URL = "https://cdn.freesound.org/mtg-jamendo"

BASE_PATH = Path(__file__).parent
ASSETS_DIR = BASE_PATH / "../assets"
OUTPUT_DIR = BASE_PATH / "../data/audio"

SHA256_TARS_FILE = ASSETS_DIR / "raw_30s_audio-low_sha256_tars.txt"
SHA256_TRACKS_FILE = ASSETS_DIR / "raw_30s_audio-low_sha256_tracks.txt"

CHUNK_SIZE = 512 * 1024  # 512 KB


# -------------------------
# HELPERS
# -------------------------
def ensure_audio_dir():
    """
    Ensure data/audio directory exists.
    Safe to call multiple times.
    """
    path = Path(__file__).parent / "../data/audio"
    path.mkdir(parents=True, exist_ok=True)
    return path

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))

        with tempfile.NamedTemporaryFile(
            dir=dest.parent,
            delete=False
        ) as tmp:
            with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
                for chunk in r.iter_content(CHUNK_SIZE):
                    if chunk:
                        tmp.write(chunk)
                        bar.update(len(chunk))

        shutil.move(tmp.name, dest)


def load_tar_checksums() -> dict[str, str]:
    with SHA256_TARS_FILE.open() as f:
        return {name: sha for sha, name in csv.reader(f, delimiter=" ")}


def load_track_checksums() -> dict[str, str]:
    with SHA256_TRACKS_FILE.open() as f:
        return {name: sha for sha, name in csv.reader(f, delimiter=" ")}


def unpack_and_verify(tar_path: Path, track_checksums: dict[str, str]):
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        tar.extractall(OUTPUT_DIR)

    for m in members:
        if not m.isfile():
            continue
        rel = m.name
        if rel not in track_checksums:
            continue

        p = OUTPUT_DIR / rel
        if compute_sha256(p) != track_checksums[rel]:
            raise RuntimeError(f"Checksum mismatch: {p}")


# -------------------------
# CORE LOGIC
# -------------------------

def _download_tars(tar_names: list[str]):
    tar_checksums = load_tar_checksums()
    track_checksums = load_track_checksums()

    for tar_name in tar_names:
        tar_path = OUTPUT_DIR / tar_name
        if tar_path.exists():
            continue

        url = f"{BASE_URL}/{DATASET}/{DATA_TYPE}/{tar_name}"
        print("↓", tar_name)
        download_file(url, tar_path)

        if compute_sha256(tar_path) != tar_checksums[tar_name]:
            tar_path.unlink(missing_ok=True)
            raise RuntimeError(f"Bad checksum for {tar_name}")

        unpack_and_verify(tar_path, track_checksums)
        tar_path.unlink()


# -------------------------
# PUBLIC API
# -------------------------

def download():
    tar_checksums = load_tar_checksums()
    tar_names = list(tar_checksums.keys())

    print(f"Downloading FULL dataset ({len(tar_names)} tar files)")
    _download_tars(tar_names)


def download_sample():
    """
    Download a 5% SAMPLE of raw_30s / audio-low.
    """
    tar_checksums = load_tar_checksums()
    tar_names = list(tar_checksums.keys())

    random.Random(SEED).shuffle(tar_names)
    k = max(1, int(len(tar_names) * 0.05))
    sample = tar_names[:k]

    print(f"Downloading SAMPLE dataset ({k}/{len(tar_names)} tar files)")
    _download_tars(sample)


def main():
    ensure_audio_dir()

    if "--full" in sys.argv:
        download()
    else:
        download_sample()


if __name__ == "__main__":
    main()