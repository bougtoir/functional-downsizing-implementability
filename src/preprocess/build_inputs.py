from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
from shapely.geometry import Point

from src.common import ROOT, load_config, write_json

RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"


def geojson_from_zip(path: Path) -> gpd.GeoDataFrame:
    with zipfile.ZipFile(path) as archive:
        name = next(name for name in archive.namelist() if name.lower().endswith(".geojson"))
        data = json.loads(archive.read(name))
    return gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:6668")


def build_municipalities(prefecture_codes: list[str], projected_crs: str) -> gpd.GeoDataFrame:
    frames = []
    for code in prefecture_codes:
        frame = geojson_from_zip(RAW / "nlni" / f"N03-20250101_{code}_GML.zip")
        frame["pref_code"] = code
        frames.append(frame)
    polygons = pd.concat(frames, ignore_index=True)
    polygons = gpd.GeoDataFrame(polygons, geometry="geometry", crs="EPSG:6668")
    polygons = polygons[polygons["N03_007"].notna()].copy()
    polygons["municipality_code"] = polygons["N03_007"].astype(str)
    polygons["municipality_name"] = polygons["N03_004"].fillna("").astype(str)
    polygons = polygons.to_crs(projected_crs)
    polygons["geometry"] = polygons.geometry.make_valid()
    municipalities = polygons.dissolve(
        by="municipality_code",
        aggfunc={
            "municipality_name": "first",
            "pref_code": "first",
            "N03_001": "first",
        },
        as_index=False,
    )
    municipalities = municipalities.rename(columns={"N03_001": "prefecture_name"})
    return municipalities[
        [
            "municipality_code",
            "municipality_name",
            "pref_code",
            "prefecture_name",
            "geometry",
        ]
    ]


def population_points(
    raster_path: Path, municipalities: gpd.GeoDataFrame, minimum_population: float
) -> gpd.GeoDataFrame:
    bounds_wgs84 = municipalities.to_crs("EPSG:4326").total_bounds
    with rasterio.open(raster_path) as dataset:
        window = rasterio.windows.from_bounds(*bounds_wgs84, transform=dataset.transform)
        window = window.round_offsets().round_lengths()
        values = dataset.read(1, window=window, masked=True)
        rows, cols = np.where((~values.mask) & (values.data >= minimum_population))
        populations = values.data[rows, cols].astype(float)
        global_rows = rows + int(window.row_off)
        global_cols = cols + int(window.col_off)
        xs, ys = xy(dataset.transform, global_rows, global_cols, offset="center")
        source_crs = dataset.crs

    points = gpd.GeoDataFrame(
        {
            "cell_id": [f"cell_{r}_{c}" for r, c in zip(global_rows, global_cols, strict=True)],
            "population": populations,
        },
        geometry=gpd.points_from_xy(xs, ys),
        crs=source_crs,
    ).to_crs(municipalities.crs)
    points = gpd.sjoin(
        points,
        municipalities[
            ["municipality_code", "municipality_name", "pref_code", "geometry"]
        ],
        how="inner",
        predicate="within",
    ).drop(columns="index_right")
    return points


def allocate_cluster_counts(municipalities: pd.DataFrame, target: int) -> dict[str, int]:
    population = municipalities.set_index("municipality_code")["population"]
    allocation = pd.Series(1, index=population.index, dtype=int)
    remaining = max(0, target - len(allocation))
    if remaining:
        raw = population / population.sum() * remaining
        allocation += np.floor(raw).astype(int)
        residual = target - int(allocation.sum())
        if residual:
            for code in (raw - np.floor(raw)).sort_values(ascending=False).index[:residual]:
                allocation.loc[code] += 1
    return allocation.to_dict()


def weighted_kmeans(
    coordinates: np.ndarray, weights: np.ndarray, clusters: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if clusters >= len(coordinates):
        labels = np.arange(len(coordinates))
        return labels, coordinates.copy()

    rng = np.random.default_rng(seed)
    centers = [coordinates[rng.choice(len(coordinates), p=weights / weights.sum())]]
    for _ in range(1, clusters):
        nearest_sq = np.min(
            np.sum((coordinates[:, None, :] - np.asarray(centers)[None, :, :]) ** 2, axis=2),
            axis=1,
        )
        score = weights * nearest_sq
        if score.sum() <= 0:
            candidate = rng.integers(0, len(coordinates))
        else:
            candidate = rng.choice(len(coordinates), p=score / score.sum())
        centers.append(coordinates[candidate])
    centers_array = np.asarray(centers, dtype=float)

    labels = np.zeros(len(coordinates), dtype=int)
    for _ in range(100):
        new_labels = np.argmin(
            np.sum((coordinates[:, None, :] - centers_array[None, :, :]) ** 2, axis=2),
            axis=1,
        )
        new_centers = centers_array.copy()
        for cluster in range(clusters):
            member = new_labels == cluster
            if member.any():
                new_centers[cluster] = np.average(
                    coordinates[member], axis=0, weights=weights[member]
                )
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers_array):
            labels = new_labels
            centers_array = new_centers
            break
        labels = new_labels
        centers_array = new_centers
    return labels, centers_array


def cluster_population(
    points: gpd.GeoDataFrame,
    clusters_per_prefecture: int,
    seed: int,
) -> gpd.GeoDataFrame:
    municipal_totals = (
        points.groupby(["pref_code", "municipality_code"], as_index=False)["population"]
        .sum()
        .sort_values(["pref_code", "municipality_code"])
    )
    allocations: dict[str, int] = {}
    for _, group in municipal_totals.groupby("pref_code"):
        allocations.update(allocate_cluster_counts(group, clusters_per_prefecture))

    records = []
    for municipality_code, group in points.groupby("municipality_code", sort=True):
        coordinates = np.column_stack((group.geometry.x, group.geometry.y))
        weights = group["population"].to_numpy()
        count = min(allocations[municipality_code], len(group))
        local_seed = seed + int(municipality_code)
        labels, centers = weighted_kmeans(coordinates, weights, count, local_seed)
        for cluster in range(count):
            member = labels == cluster
            member_population = float(weights[member].sum())
            records.append(
                {
                    "demand_id": f"{municipality_code}_{cluster:02d}",
                    "municipality_code": municipality_code,
                    "municipality_name": group["municipality_name"].iloc[0],
                    "pref_code": group["pref_code"].iloc[0],
                    "population": member_population,
                    "source_cells": int(member.sum()),
                    "geometry": Point(*centers[cluster]),
                }
            )
    return gpd.GeoDataFrame(records, crs=points.crs)


def build_facilities(
    prefecture_codes: list[str],
    municipalities: gpd.GeoDataFrame,
    demand: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    frames = []
    for code in prefecture_codes:
        frame = geojson_from_zip(RAW / "nlni" / f"P04-20_{code}_GML.zip")
        frame["pref_code"] = code
        frames.append(frame)
    facilities = pd.concat(frames, ignore_index=True)
    facilities = gpd.GeoDataFrame(facilities, geometry="geometry", crs="EPSG:6668")
    facilities = facilities.to_crs(municipalities.crs)
    facilities = facilities.drop_duplicates(subset=["P04_001", "P04_002", "geometry"]).copy()
    facilities = gpd.sjoin(
        facilities,
        municipalities[
            ["municipality_code", "municipality_name", "pref_code", "geometry"]
        ].rename(columns={"pref_code": "boundary_pref_code"}),
        how="left",
        predicate="within",
    ).drop(columns="index_right")
    facilities["facility_class"] = facilities["P04_001"].map(
        {1: "hospital", 2: "clinic", 3: "dental"}
    )
    facilities["beds"] = pd.to_numeric(facilities["P04_008"], errors="coerce").fillna(0)
    facilities["facility_id"] = [f"facility_{index:05d}" for index in range(len(facilities))]
    facilities["is_existing"] = True
    facilities["is_candidate"] = (
        (facilities["facility_class"] == "hospital")
        | ((facilities["facility_class"] == "clinic") & (facilities["beds"] > 0))
    )

    demand_centres = (
        demand.dissolve(
            by="municipality_code",
            aggfunc={
                "population": "sum",
                "municipality_name": "first",
                "pref_code": "first",
            },
            as_index=False,
        )
        .drop(columns="geometry")
        .merge(
            demand.assign(
                weighted_x=demand.geometry.x * demand["population"],
                weighted_y=demand.geometry.y * demand["population"],
            )
            .groupby("municipality_code", as_index=False)
            .agg(
                weighted_x=("weighted_x", "sum"),
                weighted_y=("weighted_y", "sum"),
                population_check=("population", "sum"),
            ),
            on="municipality_code",
        )
    )
    demand_centres["x"] = demand_centres["weighted_x"] / demand_centres["population_check"]
    demand_centres["y"] = demand_centres["weighted_y"] / demand_centres["population_check"]
    existing_municipalities = set(
        facilities.loc[facilities["is_candidate"], "municipality_code"].dropna()
    )
    for row in demand_centres[
        ~demand_centres["municipality_code"].isin(existing_municipalities)
    ].itertuples():
        clinics = facilities[
            (facilities["municipality_code"] == row.municipality_code)
            & (facilities["facility_class"] == "clinic")
        ]
        if len(clinics):
            distance = np.hypot(clinics.geometry.x - row.x, clinics.geometry.y - row.y)
            facilities.loc[distance.idxmin(), "is_candidate"] = True
    planned = demand_centres.copy()
    planned_facilities = gpd.GeoDataFrame(
        {
            "P04_001": 0,
            "P04_002": "Planned consolidation and service hub",
            "P04_003": "",
            "P04_004": "",
            "P04_005": "",
            "P04_006": "",
            "P04_007": 0,
            "P04_008": 9,
            "P04_009": 9,
            "P04_010": 9,
            "pref_code": planned["pref_code"],
            "boundary_pref_code": planned["pref_code"],
            "municipality_code": planned["municipality_code"],
            "municipality_name": planned["municipality_name"],
            "facility_class": "planned_hub",
            "beds": 0.0,
            "facility_id": [f"planned_{code}" for code in planned["municipality_code"]],
            "is_existing": False,
            "is_candidate": True,
        },
        geometry=gpd.points_from_xy(planned["x"], planned["y"]),
        crs=demand.crs,
    )
    return pd.concat([facilities, planned_facilities], ignore_index=True).pipe(
        lambda frame: gpd.GeoDataFrame(frame, geometry="geometry", crs=demand.crs)
    )


def main() -> None:
    config = load_config()
    prefectures = [item["code"] for item in config["project"]["study_prefectures"]]
    projected_crs = config["geography"]["projected_crs"]
    INTERIM.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    municipalities = build_municipalities(prefectures, projected_crs)
    points = population_points(
        RAW / "worldpop" / "jpn_ppp_2020_1km_Aggregated.tif",
        municipalities,
        config["geography"]["population_minimum"],
    )
    demand = cluster_population(
        points,
        config["geography"]["demand_clusters_per_prefecture"],
        config["project"]["seed"],
    )
    facilities = build_facilities(prefectures, municipalities, demand)

    municipalities.to_file(PROCESSED / "geography.gpkg", layer="municipalities", driver="GPKG")
    demand.to_file(PROCESSED / "geography.gpkg", layer="demand", driver="GPKG")
    facilities.to_file(PROCESSED / "geography.gpkg", layer="facilities", driver="GPKG")
    points.drop(columns="geometry").to_csv(PROCESSED / "population_cells.csv", index=False)
    demand.assign(x=demand.geometry.x, y=demand.geometry.y).drop(columns="geometry").to_csv(
        PROCESSED / "demand.csv", index=False
    )
    facilities.assign(x=facilities.geometry.x, y=facilities.geometry.y).drop(
        columns="geometry"
    ).to_csv(PROCESSED / "facilities.csv", index=False)

    source_population = float(points["population"].sum())
    clustered_population = float(demand["population"].sum())
    relative_error = (
        abs(source_population - clustered_population) / source_population
        if source_population
        else math.nan
    )
    summary = {
        "prefectures": prefectures,
        "municipalities": int(len(municipalities)),
        "populated_cells": int(len(points)),
        "demand_clusters": int(len(demand)),
        "all_medical_institutions": int(facilities["is_existing"].sum()),
        "candidate_fixed_assets_and_hubs": int(facilities["is_candidate"].sum()),
        "planned_hubs": int((~facilities["is_existing"]).sum()),
        "population_from_cells": source_population,
        "population_in_clusters": clustered_population,
        "population_conservation_relative_error": relative_error,
        "crs": projected_crs,
    }
    if relative_error > 1e-12:
        raise RuntimeError(f"Population conservation failed: {relative_error}")
    write_json(PROCESSED / "preprocessing_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
