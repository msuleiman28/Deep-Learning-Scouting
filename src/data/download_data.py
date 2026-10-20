"""Download the raw Wyscout and Transfermarkt data into data/raw/.

Run from anywhere inside the repo (Windows, macOS, or Linux):

    python src/data/download_data.py           # fetch whatever is missing
    python src/data/download_data.py --force   # re-download everything

Uses only the Python standard library. Files that already exist are skipped,
and downloads are written to a temporary ``.part`` file first so an interrupted
run never leaves a truncated file that looks complete.

Notes on the sources:
- Wyscout (figshare) is a fixed 2017/18 release, so it is reproducible.
- Transfermarkt (dcaribou/transfermarkt-datasets) URLs always serve the latest
  snapshot. To keep everyone on identical data we pin the snapshot published
  2026-09-05 (data current to 2026-07-06) by MD5 in ``TM_MD5``. Downloads and
  existing files are checked against it, so if upstream ever changes a file
  the script fails loudly instead of silently handing out different data.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

LEAGUES = ["England", "France", "Germany", "Italy", "Spain"]

FIGSHARE = "https://ndownloader.figshare.com/files"
WYSCOUT_FILES = {"players.json": f"{FIGSHARE}/15073721", "teams.json": f"{FIGSHARE}/15073697"}
WYSCOUT_ZIPS = {  # zip URL -> member file prefix; only the five league files are extracted
    f"{FIGSHARE}/14464622": "matches",
    f"{FIGSHARE}/14464685": "events",
}

TM_BASE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
TM_TABLES = ["players", "transfers", "appearances", "games", "clubs", "player_valuations"]
TM_MD5 = {  # pinned snapshot: published 2026-09-05, data current to 2026-07-06
    "players": "bf46ce79e2aec6db1decce7713bb618c",
    "transfers": "21cdd99be19205ed521890c45da44235",
    "appearances": "2b39bcd2f5116aa020cdf69cbb8fe3d6",
    "games": "9d56854283f5d47601688ffbd3baaedd",
    "clubs": "f30713f2d0022ca4ecbb382cd695e389",
    "player_valuations": "087cb1d21a2469ff4a597b6ca86cced4",
}


def find_repo_root() -> Path:
    """Walk up from this file to the folder that contains data/raw."""
    for p in Path(__file__).resolve().parents:
        if (p / "data" / "raw").is_dir():
            return p
    sys.exit("Could not find data/raw above this script; run it from inside the repo.")


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, md5: str | None = None) -> None:
    """Stream url to dest via a .part file, printing progress.

    If md5 is given, the file is only moved into place when its hash matches.
    """
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    h = hashlib.md5()
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {dest.name}: {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print(f"\r  {dest.name}: {done / 1e6:.1f} MB" + " " * 20)
    if md5 and h.hexdigest() != md5:
        tmp.unlink()
        raise ValueError("does not match the pinned snapshot (upstream changed the file); "
                         "ask the team before updating TM_MD5")
    tmp.replace(dest)


def fetch(url: str, dest: Path, force: bool, failures: list[str], md5: str | None = None) -> None:
    if dest.exists() and not force:
        if md5 and file_md5(dest) != md5:
            print(f"  FAILED {dest.name}: your copy differs from the pinned snapshot; "
                  "re-run with --force to replace it")
            failures.append(dest.name)
        else:
            print(f"  {dest.name}: already present" + (", checksum OK" if md5 else "") + ", skipping")
        return
    try:
        download(url, dest, md5)
    except Exception as exc:  # keep going so one bad link doesn't block the rest
        print(f"\n  FAILED {dest.name}: {exc}")
        failures.append(dest.name)


def fetch_league_zip(url: str, prefix: str, out_dir: Path, force: bool, failures: list[str]) -> None:
    """Download a Wyscout zip and extract only the five league files."""
    wanted = [f"{prefix}_{lg}.json" for lg in LEAGUES]
    if all((out_dir / w).exists() for w in wanted) and not force:
        print(f"  {prefix}_*.json: already present, skipping")
        return
    zip_path = out_dir / f"{prefix}.zip"
    try:
        download(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            members = {Path(n).name: n for n in zf.namelist()}
            for w in wanted:
                if w not in members:
                    raise FileNotFoundError(f"{w} missing from {zip_path.name}")
                with zf.open(members[w]) as src, open(out_dir / w, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                print(f"  extracted {w}")
    except Exception as exc:
        print(f"\n  FAILED {prefix}.zip: {exc}")
        failures.append(f"{prefix}.zip")
    finally:
        zip_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = parser.parse_args()

    raw = find_repo_root() / "data" / "raw"
    wy_dir, tm_dir = raw / "wyscout", raw / "transfermarkt"
    wy_dir.mkdir(parents=True, exist_ok=True)
    tm_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    print(f"Wyscout -> {wy_dir}")
    for name, url in WYSCOUT_FILES.items():
        fetch(url, wy_dir / name, args.force, failures)
    for url, prefix in WYSCOUT_ZIPS.items():
        fetch_league_zip(url, prefix, wy_dir, args.force, failures)

    print(f"Transfermarkt -> {tm_dir}")
    for t in TM_TABLES:
        fetch(f"{TM_BASE}/{t}.csv.gz", tm_dir / f"{t}.csv.gz", args.force, failures, TM_MD5[t])

    expected = ([wy_dir / n for n in WYSCOUT_FILES]
                + [wy_dir / f"{p}_{lg}.json" for p in WYSCOUT_ZIPS.values() for lg in LEAGUES]
                + [tm_dir / f"{t}.csv.gz" for t in TM_TABLES])
    missing = [p.name for p in expected if not p.exists()]
    if failures or missing:
        print("\nIncomplete. Failed:", ", ".join(failures) or "-", "| Missing:", ", ".join(missing) or "-")
        print("Re-run the script to retry; files already downloaded are kept.")
        sys.exit(1)
    print(f"\nDone: all {len(expected)} files are in {raw}")


if __name__ == "__main__":
    main()
