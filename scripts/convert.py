from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


SRC_DIR = Path("data/audio/mp3")
DST_DIR = Path("data/audio/wav")

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_FMT = "pcm_s16le"

MAX_WORKERS = 6


def ensure_dirs():
    DST_DIR.mkdir(parents=True, exist_ok=True)


def convert_one(src: Path):
    dst = DST_DIR / f"{src.stem}.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", str(src),
        "-ac", str(CHANNELS),
        "-ar", str(SAMPLE_RATE),
        "-sample_fmt", "s16",
        "-c:a", SAMPLE_FMT,
        str(dst),
    ]

    subprocess.run(cmd, check=True)
    src.unlink()
    return src.name


def convert_all():
    ensure_dirs()
    files = list(SRC_DIR.rglob("*.mp3"))

    print(f"Found {len(files)} mp3 files")
    if not files:
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(convert_one, f) for f in files]

        for _ in tqdm(as_completed(futures), total=len(futures), desc="Converting audio"):
            pass


if __name__ == "__main__":
    import os
    convert_all()
