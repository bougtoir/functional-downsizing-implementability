from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.common import ROOT

DOIS = [
    "10.1080/10511482.2011.585164",
    "10.1080/01944360802354956",
    "10.1111/j.1468-2427.2011.01095.x",
    "10.1111/juaf.12150",
    "10.3390/land13050660",
    "10.1787/07970966-en",
    "10.1111/padm.12104",
    "10.1177/1078087419839492",
    "10.3390/su132313107",
    "10.1080/13563475.2021.1971951",
    "10.1186/1476-072X-3-3",
    "10.1016/j.seps.2023.101666",
    "10.22605/RRH4521",
    "10.3390/su152115272",
    "10.1093/ssjj/jyae022",
    "10.1186/s12939-017-0671-2",
    "10.1016/j.tranpol.2022.06.015",
    "10.3390/su12072876",
    "10.1016/j.ijdrr.2020.101937",
    "10.1016/j.oneear.2019.09.008",
]

DESTINATION = ROOT / "data" / "raw" / "literature"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def filename(doi: str) -> str:
    safe = doi.lower().replace("/", "_").replace(".", "-")
    return f"crossref_{safe}.json"


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    ledger = DESTINATION / "acquisition_ledger.csv"
    existing_records = {}
    if ledger.exists():
        with ledger.open(encoding="utf-8", newline="") as stream:
            existing_records = {
                row["dataset_id"]: row for row in csv.DictReader(stream)
            }
    session = requests.Session()
    session.headers["User-Agent"] = (
        "regional-functional-downsizing/0.1 (mailto:bougtoir@gmail.com)"
    )
    records = []
    for doi in DOIS:
        url = f"https://api.crossref.org/works/{doi}"
        path = DESTINATION / filename(doi)
        dataset_id = f"crossref:{doi}"
        snapshot_exists = path.exists()
        if not snapshot_exists:
            response = session.get(url, timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
        retrieved = (
            existing_records.get(dataset_id, {}).get(
                "retrieved_utc",
                datetime.now(timezone.utc).isoformat(),
            )
            if snapshot_exists
            else datetime.now(timezone.utc).isoformat()
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["message"]["DOI"].lower() != doi.lower():
            raise RuntimeError(f"Crossref DOI mismatch for {doi}")
        records.append(
            {
                "dataset_id": dataset_id,
                "source_url": url,
                "retrieved_utc": retrieved,
                "retrieval_conditions": "Crossref REST API single-work response",
                "storage_path": str(path.relative_to(ROOT)),
                "file_size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "license": "Crossref metadata; see https://www.crossref.org/",
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
