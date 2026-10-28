from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
LEDGER_CSV = RAW_DIR / "acquisition_ledger.csv"
LEDGER_JSON = RAW_DIR / "acquisition_ledger.json"
USER_AGENT = "regional-functional-downsizing/0.1 (academic reproducibility)"


@dataclass(frozen=True)
class Source:
    dataset_id: str
    version: str
    url: str
    relative_path: str
    license: str
    coverage: str


SOURCES = [
    Source(
        dataset_id="worldpop_jpn_population",
        version="2020_1km_aggregated",
        url=(
            "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/"
            "2020/JPN/jpn_ppp_2020_1km_Aggregated.tif"
        ),
        relative_path="worldpop/jpn_ppp_2020_1km_Aggregated.tif",
        license="WorldPop open-data terms / CC BY 4.0; verify provider metadata",
        coverage="Japan, 2020, approximately 1-km population counts",
    ),
    Source(
        dataset_id="geofabrik_chugoku_osm",
        version="chugoku-260901",
        url="https://download.geofabrik.de/asia/japan/chugoku-260901.osm.pbf",
        relative_path="osm/chugoku-260901.osm.pbf",
        license="OpenStreetMap data © contributors; Open Database License 1.0",
        coverage=(
            "Chūgoku road and ferry network snapshot published 2026-09-02 "
            "for targeted travel-time validation"
        ),
    ),
    *[
        Source(
            dataset_id=f"nlni_n03_{code}",
            version="N03-2025-20250101",
            url=(
                "https://nlftp.mlit.go.jp/ksj/gml/data/N03/N03-2025/"
                f"N03-20250101_{code}_GML.zip"
            ),
            relative_path=f"nlni/N03-20250101_{code}_GML.zip",
            license="CC BY 4.0; Survey Act notice applies",
            coverage=f"Administrative areas, prefecture code {code}, 2025-01-01",
        )
        for code in ("31", "32", "33")
    ],
    *[
        Source(
            dataset_id=f"nlni_p04_{code}",
            version="P04-2020",
            url=(
                "https://nlftp.mlit.go.jp/ksj/gml/data/P04/P04-20/"
                f"P04-20_{code}_GML.zip"
            ),
            relative_path=f"nlni/P04-20_{code}_GML.zip",
            license="CC BY 4.0",
            coverage=f"Medical institutions, prefecture code {code}, FY2020",
        )
        for code in ("31", "32", "33")
    ],
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def remote_metadata(url: str) -> dict[str, str]:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return {
            "status": str(response.status),
            "content_length": response.headers.get("Content-Length", ""),
            "last_modified": response.headers.get("Last-Modified", ""),
            "etag": response.headers.get("ETag", ""),
            "content_type": response.headers.get("Content-Type", ""),
        }


def download(
    source: Source,
    destination: Path,
    existing: dict[str, str] | None,
) -> tuple[str, dict[str, str]]:
    if destination.exists():
        if existing is not None:
            if destination.stat().st_size != int(existing["file_size_bytes"]):
                raise RuntimeError(
                    f"Existing immutable raw file has unexpected size: {destination}"
                )
            if sha256(destination) != existing["sha256"]:
                raise RuntimeError(
                    f"Existing immutable raw file has unexpected checksum: {destination}"
                )
            metadata = {
                "status": existing["http_status"],
                "content_length": existing["file_size_bytes"],
                "last_modified": existing["remote_last_modified"],
                "etag": existing["remote_etag"],
                "content_type": existing["content_type"],
            }
        else:
            metadata = remote_metadata(source.url)
            expected = int(metadata["content_length"] or 0)
            if expected and destination.stat().st_size != expected:
                raise RuntimeError(
                    f"Existing immutable raw file has unexpected size: {destination}"
                )
        return "reused", metadata

    metadata = remote_metadata(source.url)
    if metadata["status"] != "200":
        raise RuntimeError(f"HEAD {source.url} returned {metadata['status']}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()

    request = urllib.request.Request(source.url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)

    expected = int(metadata["content_length"] or 0)
    if expected and temporary.stat().st_size != expected:
        raise RuntimeError(
            f"Partial download for {source.dataset_id}: "
            f"{temporary.stat().st_size} != {expected} bytes"
        )
    os.replace(temporary, destination)
    return "downloaded", metadata


def acquire_all() -> list[dict[str, object]]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    existing_records: dict[str, dict[str, str]] = {}
    if LEDGER_CSV.exists():
        with LEDGER_CSV.open(encoding="utf-8", newline="") as stream:
            existing_records = {
                row["dataset_id"]: row for row in csv.DictReader(stream)
            }
    records: list[dict[str, object]] = []
    for source in SOURCES:
        destination = RAW_DIR / source.relative_path
        started = utc_now()
        existing_record = existing_records.get(source.dataset_id)
        status, metadata = download(source, destination, existing_record)
        existing = existing_record if status == "reused" else None
        record: dict[str, object] = {
            **asdict(source),
            "retrieved_at_utc": (
                existing["retrieved_at_utc"] if existing is not None else started
            ),
            "retrieval_status": (
                existing["retrieval_status"] if existing is not None else status
            ),
            "http_status": (
                existing["http_status"] if existing is not None else metadata["status"]
            ),
            "content_type": (
                existing["content_type"]
                if existing is not None
                else metadata["content_type"]
            ),
            "remote_last_modified": (
                existing["remote_last_modified"]
                if existing is not None
                else metadata["last_modified"]
            ),
            "remote_etag": (
                existing["remote_etag"] if existing is not None else metadata["etag"]
            ),
            "storage_path": str(destination.relative_to(ROOT)),
            "file_size_bytes": destination.stat().st_size,
            "sha256": sha256(destination),
        }
        records.append(record)
        print(
            f"{source.dataset_id}: {status}, "
            f"{destination.stat().st_size / (1024 * 1024):.1f} MiB"
        )
        time.sleep(0.2)
    return records


def write_ledgers(records: list[dict[str, object]]) -> None:
    fieldnames = list(records[0])
    with LEDGER_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    with LEDGER_JSON.open("w", encoding="utf-8") as stream:
        json.dump(records, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> None:
    records = acquire_all()
    write_ledgers(records)
    print(f"Wrote {LEDGER_CSV.relative_to(ROOT)} and {LEDGER_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
