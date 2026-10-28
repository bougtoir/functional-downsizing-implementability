from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.common import ROOT

ORCID = "0000-0001-7261-9062"
URL = f"https://pub.orcid.org/v3.0/{ORCID}/record"
DESTINATION = ROOT / "data" / "raw" / "author"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    path = DESTINATION / "orcid_0000-0001-7261-9062.json"
    ledger = DESTINATION / "acquisition_ledger.csv"
    snapshot_exists = path.exists()
    retrieved = datetime.now(timezone.utc).isoformat()
    if snapshot_exists and ledger.exists():
        with ledger.open(encoding="utf-8", newline="") as stream:
            previous = list(csv.DictReader(stream))
        if previous:
            retrieved = previous[0]["retrieved_utc"]
    if not snapshot_exists:
        response = requests.get(
            URL,
            headers={"Accept": "application/json"},
            timeout=60,
        )
        response.raise_for_status()
        path.write_bytes(response.content)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["orcid-identifier"]["path"] != ORCID:
        raise RuntimeError("ORCID record identifier mismatch")
    record = {
        "dataset_id": f"orcid:{ORCID}",
        "source_url": URL,
        "retrieved_utc": retrieved,
        "retrieval_conditions": "ORCID Public API v3.0 record; public fields only",
        "storage_path": str(path.relative_to(ROOT)),
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "license": "ORCID public record; retained for author-metadata verification",
    }
    with ledger.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(record), lineterminator="\n")
        writer.writeheader()
        writer.writerow(record)


if __name__ == "__main__":
    main()
