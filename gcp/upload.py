from pathlib import Path
import csv
import tempfile
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import storage
from tqdm import tqdm

BASE_URL = "https://cdn.freesound.org/mtg-jamendo/raw_30s/audio-low"
TSV_PATH = Path("assets/raw_30s_filtered.tsv")
BUCKET = "spark-audio-bucket"
PREFIX = "audio/mp3"
CHUNK_SIZE = 512 * 1024
MAX_WORKERS = 32
MAX_RETRIES = 3


def iter_paths(tsv_path: Path):
    with tsv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) >= 4:
                yield Path(row[3]).with_suffix(".low.mp3").as_posix()


def existing_blobs(client):
    return {
        blob.name
        for blob in client.list_blobs(BUCKET, prefix=PREFIX)
        if blob.name.endswith(".mp3")
    }


def worker(rel_path: str):
    url = f"{BASE_URL}/{rel_path}"
    blob_name = f"{PREFIX}/{rel_path}"

    for attempt in range(MAX_RETRIES):
        try:
            with requests.Session() as session:
                with session.get(url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                        for chunk in r.iter_content(CHUNK_SIZE):
                            if chunk:
                                tmp.write(chunk)
                        tmp_path = Path(tmp.name)

            storage.Client().bucket(BUCKET).blob(blob_name).upload_from_filename(str(tmp_path))
            tmp_path.unlink()
            return rel_path

        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, 4s


def main():
    root_client = storage.Client()
    paths = sorted(set(iter_paths(TSV_PATH)))
    already_uploaded = existing_blobs(root_client)
    pending = [p for p in paths if f"{PREFIX}/{p}" not in already_uploaded]

    print(f"{len(pending)} files to upload ({len(already_uploaded)} already done)")

    failed = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(worker, p): p for p in pending}
        with tqdm(total=len(pending), desc="Uploading mp3") as bar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    failed.append(futures[future])
                    tqdm.write(f"Failed: {futures[future]} — {e}")
                finally:
                    bar.update(1)

    if failed:
        print(f"\n{len(failed)} files failed permanently:")
        for f in failed:
            print(f"  {f}")


if __name__ == "__main__":
    main()