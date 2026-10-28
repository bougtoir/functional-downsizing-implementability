from __future__ import annotations

import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.common import ROOT

SOURCES = [
    (
        "regional-studies-scope",
        "https://www.tandfonline.com/journals/cres20/about-this-journal",
    ),
    (
        "taylor-francis-data-sharing",
        "https://authorservices.taylorandfrancis.com/data-sharing/",
    ),
    (
        "taylor-francis-generative-ai",
        "https://authorservices.taylorandfrancis.com/editorial-policies/"
        "using-ai-in-your-research-and-manuscript-preparations/",
    ),
    (
        "taylor-francis-images-figures",
        "https://authorservices.taylorandfrancis.com/editorial-policies/"
        "images-and-figures/",
    ),
    (
        "regional-studies-author-instructions",
        "https://www.tandfonline.com/action/authorSubmission?"
        "show=instructions&journalCode=cres20",
    ),
]

DESTINATION = ROOT / "data" / "raw" / "journal"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    ledger = DESTINATION / "acquisition_ledger.csv"
    existing_records = {}
    if ledger.exists():
        with ledger.open(encoding="utf-8", newline="") as stream:
            existing_records = {
                row["dataset_id"]: row for row in csv.DictReader(stream)
            }
    records = []
    for dataset_id, url in SOURCES:
        path = DESTINATION / f"{dataset_id}.html"
        status_path = DESTINATION / f"{dataset_id}.status.txt"
        if path.exists() != status_path.exists():
            raise RuntimeError(
                f"Incomplete retained journal snapshot for {dataset_id}: "
                f"{path.exists()=}, {status_path.exists()=}"
            )
        snapshot_exists = path.exists() and status_path.exists()
        if not snapshot_exists:
            response = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 regional-research-audit"},
                timeout=90,
            )
            path.write_bytes(response.content)
            status_path.write_text(
                f"HTTP {response.status_code}\nfinal_url={response.url}\n",
                encoding="utf-8",
            )
        status = status_path.read_text(encoding="utf-8").splitlines()[0]
        retrieved = (
            existing_records.get(dataset_id, {}).get(
                "retrieved_utc",
                datetime.now(timezone.utc).isoformat(),
            )
            if snapshot_exists
            else datetime.now(timezone.utc).isoformat()
        )
        condition = f"Single-page publisher snapshot; {status}"
        if dataset_id == "regional-studies-author-instructions":
            condition += "; Cloudflare verification response, not complete instructions"
        records.append(
            {
                "dataset_id": dataset_id,
                "source_url": url,
                "retrieved_utc": retrieved,
                "retrieval_conditions": condition,
                "storage_path": str(path.relative_to(ROOT)),
                "file_size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "license": "Publisher web content; retained for research audit only",
            }
        )
    with ledger.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(records[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
