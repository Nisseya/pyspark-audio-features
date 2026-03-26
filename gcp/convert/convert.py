from pathlib import Path
import subprocess
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.cloud import storage
from tqdm import tqdm

BUCKET = "spark-audio-bucket"
MP3_PREFIX = "audio/mp3"
WAV_PREFIX = "audio/wav"
SAMPLE_RATE = 16000
CHANNELS = 1
MAX_WORKERS = 16
MAX_RETRIES = 3

# Injected by Cloud Run
TASK_INDEX = int(os.environ.get("CLOUD_RUN_TASK_INDEX", 0))
TASK_COUNT = int(os.environ.get("CLOUD_RUN_TASK_COUNT", 1))


def list_mp3_blobs(client):
    return sorted(
        b.name
        for b in client.list_blobs(BUCKET, prefix=MP3_PREFIX)
        if b.name.endswith(".mp3")
    )


def list_wav_blobs(client):
    return {
        b.name
        for b in client.list_blobs(BUCKET, prefix=WAV_PREFIX)
        if b.name.endswith(".wav")
    }


def mp3_to_wav_blob_name(blob_name: str) -> str:
    rel = blob_name.removeprefix(f"{MP3_PREFIX}/")
    return f"{WAV_PREFIX}/{Path(rel).with_suffix('.wav').as_posix()}"


def convert_one(blob_name: str):
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    wav_blob_name = mp3_to_wav_blob_name(blob_name)

    for attempt in range(MAX_RETRIES):
        try:
            mp3_bytes = io.BytesIO()
            bucket.blob(blob_name).download_to_file(mp3_bytes)
            mp3_bytes.seek(0)

            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", "pipe:0",
                    "-ac", str(CHANNELS),
                    "-ar", str(SAMPLE_RATE),
                    "-sample_fmt", "s16",
                    "-c:a", "pcm_s16le",
                    "-f", "wav",
                    "pipe:1",
                ],
                input=mp3_bytes.read(),
                capture_output=True,
                check=True,
            )

            wav_bytes = io.BytesIO(result.stdout)
            bucket.blob(wav_blob_name).upload_from_file(wav_bytes, content_type="audio/wav")
            return blob_name

        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)


def main():
    client = storage.Client()

    print("Listing blobs...")
    all_mp3s = list_mp3_blobs(client)
    existing_wavs = list_wav_blobs(client)

    pending = [
        b for b in all_mp3s
        if mp3_to_wav_blob_name(b) not in existing_wavs
    ]

    # Each task takes its own slice
    my_slice = pending[TASK_INDEX::TASK_COUNT]
    print(f"Task {TASK_INDEX}/{TASK_COUNT}: {len(my_slice)} files to convert")

    failed = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(convert_one, b): b for b in my_slice}
        with tqdm(total=len(my_slice), desc=f"Task {TASK_INDEX}") as bar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    failed.append(futures[future])
                    tqdm.write(f"Failed: {futures[future]} — {e}")
                finally:
                    bar.update(1)

    if failed:
        print(f"\n{len(failed)} files failed:")
        for f in failed:
            print(f"  {f}")
        exit(1)  # non-zero exit so Cloud Run marks the task as failed


if __name__ == "__main__":
    main()