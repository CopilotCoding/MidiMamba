import os
import hashlib
from pathlib import Path
from collections import defaultdict
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


def sha256_file(path: Path) -> tuple[str, str]:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest(), str(path)


def find_midis(root: Path):
    return list(root.rglob("*.mid")) + list(root.rglob("*.midi"))


def get_timestamp(path: Path) -> float:
    # Windows: creation time exists, fallback to modified time
    try:
        return path.stat().st_ctime
    except Exception:
        return path.stat().st_mtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log", default="dedupe_log.txt")
    args = parser.parse_args()

    root = Path(args.path)
    files = find_midis(root)

    print(f"Found {len(files)} MIDI files")

    hash_map = defaultdict(list)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(sha256_file, f) for f in files]

        for i, fut in enumerate(as_completed(futures), 1):
            try:
                h, path = fut.result()
                hash_map[h].append(Path(path))
            except Exception as e:
                print(f"Failed: {e}")

            if i % 1000 == 0:
                print(f"Processed {i}/{len(files)}")

    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}

    total_deleted = 0

    with open(args.log, "w", encoding="utf-8") as log:
        for h, paths in duplicates.items():

            # sort by age (oldest first)
            paths_sorted = sorted(paths, key=get_timestamp)
            keeper = paths_sorted[0]
            to_delete = paths_sorted[1:]

            log.write(f"\nHASH: {h}\n")
            log.write(f"KEEP: {keeper}\n")
            for p in to_delete:
                log.write(f"DELETE: {p}\n")

            if not args.dry_run:
                for p in to_delete:
                    try:
                        os.remove(p)
                        total_deleted += 1
                    except Exception as e:
                        log.write(f"FAILED DELETE: {p} ({e})\n")

    print(f"\nDone.")
    print(f"Duplicate groups: {len(duplicates)}")
    print(f"Deleted files: {total_deleted if not args.dry_run else 0}")
    print(f"Log saved to: {args.log}")


if __name__ == "__main__":
    main()