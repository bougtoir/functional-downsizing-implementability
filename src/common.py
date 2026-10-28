from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with (ROOT / "config" / "analysis.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def reporting_parameters(config: dict) -> dict[str, float]:
    rf_values = sorted(float(value) for value in config["experiment"]["relocation_friction"])
    decline_values = sorted(
        float(value) for value in config["experiment"]["population_decline"]
    )
    return {
        "baseline_decline": decline_values[0],
        "maximum_decline": decline_values[-1],
        "minimum_rf": rf_values[0],
        "central_rf": rf_values[len(rf_values) // 2],
        "maximum_rf": rf_values[-1],
    }


def detail_stem(scenario: str, relocation_friction: float, decline: float) -> str:
    return (
        f"{scenario}_rf{relocation_friction:g}_"
        f"decline{int(round(decline * 100))}"
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)
