from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import osmium
import pandas as pd
from pyproj import Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from src.common import ROOT, load_config, reporting_parameters, write_json
from src.download.public_data import sha256

DETAILS = ROOT / "outputs" / "results" / "details"
PROCESSED = ROOT / "data" / "processed"
VALIDATION = ROOT / "outputs" / "validation"
DOC = ROOT / "docs" / "NETWORK_TRAVEL_VALIDATION.md"
NO_PREDECESSOR = -9999
EARTH_RADIUS_METRES = 6_371_008.8
MAXSPEED_PATTERN = re.compile(r"(\d+(?:\.\d+)?)")


@dataclass
class RoadGraph:
    matrix: csr_matrix
    x: np.ndarray
    y: np.ndarray
    ferry_edges: set[int]
    metadata: dict[str, object]


def parse_maxspeed(value: str | None) -> float | None:
    if not value:
        return None
    match = MAXSPEED_PATTERN.search(value)
    if match is None:
        return None
    speed = float(match.group(1))
    if "mph" in value.lower():
        speed *= 1.609344
    return speed if speed > 0 else None


def way_profile(tags: dict[str, str], config: dict) -> tuple[float, bool] | None:
    access_values = {
        tags.get("access", ""),
        tags.get("vehicle", ""),
        tags.get("motor_vehicle", ""),
        tags.get("motorcar", ""),
    }
    if access_values & {"no", "private"}:
        return None
    highway = tags.get("highway")
    road_speeds = config["road_speeds_kmh"]
    if highway in road_speeds:
        if highway == "service" and tags.get("service") == "parking_aisle":
            return None
        return parse_maxspeed(tags.get("maxspeed")) or float(road_speeds[highway]), False
    if tags.get("route") == "ferry":
        return float(config["ferry_speed_kmh"]), True
    return None


def directed_permissions(tags: dict[str, str]) -> tuple[bool, bool]:
    oneway = tags.get("oneway", "").lower()
    if oneway in {"-1", "reverse"}:
        return False, True
    if oneway in {"yes", "1", "true"} or tags.get("junction") == "roundabout":
        return True, False
    return True, True


def haversine_metres(
    first_lon: float,
    first_lat: float,
    second_lon: float,
    second_lat: float,
) -> float:
    first_lat_radians = math.radians(first_lat)
    second_lat_radians = math.radians(second_lat)
    latitude_delta = second_lat_radians - first_lat_radians
    longitude_delta = math.radians(second_lon - first_lon)
    a = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(first_lat_radians)
        * math.cos(second_lat_radians)
        * math.sin(longitude_delta / 2) ** 2
    )
    return 2 * EARTH_RADIUS_METRES * math.asin(min(1.0, math.sqrt(a)))


class RoadGraphHandler(osmium.SimpleHandler):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.bounds = tuple(float(value) for value in config["bounding_box_wgs84"])
        self.node_index: dict[int, int] = {}
        self.longitude: list[float] = []
        self.latitude: list[float] = []
        self.edges: dict[tuple[int, int], tuple[float, bool]] = {}
        self.included_ways = 0
        self.included_ferry_ways = 0

    def in_bounds(self, longitude: float, latitude: float) -> bool:
        minimum_lon, minimum_lat, maximum_lon, maximum_lat = self.bounds
        return (
            minimum_lon <= longitude <= maximum_lon
            and minimum_lat <= latitude <= maximum_lat
        )

    def index_for(self, node_id: int, longitude: float, latitude: float) -> int:
        existing = self.node_index.get(node_id)
        if existing is not None:
            return existing
        index = len(self.longitude)
        self.node_index[node_id] = index
        self.longitude.append(longitude)
        self.latitude.append(latitude)
        return index

    def add_edge(
        self,
        source: int,
        target: int,
        minutes: float,
        is_ferry: bool,
    ) -> None:
        key = (source, target)
        existing = self.edges.get(key)
        if existing is None or minutes < existing[0]:
            self.edges[key] = (minutes, is_ferry)

    def way(self, way: osmium.osm.Way) -> None:
        tags = {tag.k: tag.v for tag in way.tags}
        profile = way_profile(tags, self.config)
        if profile is None or len(way.nodes) < 2:
            return
        speed_kmh, is_ferry = profile
        forward, reverse = directed_permissions(tags)
        locations = []
        for node in way.nodes:
            if not node.location.valid():
                return
            locations.append((node.ref, node.location.lon, node.location.lat))
        used = False
        for first, second in zip(locations[:-1], locations[1:], strict=True):
            first_id, first_lon, first_lat = first
            second_id, second_lon, second_lat = second
            if not (
                self.in_bounds(first_lon, first_lat)
                or self.in_bounds(second_lon, second_lat)
            ):
                continue
            distance_metres = haversine_metres(
                first_lon,
                first_lat,
                second_lon,
                second_lat,
            )
            if distance_metres <= 0:
                continue
            first_index = self.index_for(first_id, first_lon, first_lat)
            second_index = self.index_for(second_id, second_lon, second_lat)
            minutes = distance_metres / 1000 / speed_kmh * 60
            if forward:
                self.add_edge(first_index, second_index, minutes, is_ferry)
            if reverse:
                self.add_edge(second_index, first_index, minutes, is_ferry)
            used = True
        if used:
            self.included_ways += 1
            self.included_ferry_ways += int(is_ferry)


def build_graph(
    pbf_path: Path,
    config: dict,
    projected_crs: str,
    profile_sha256: str,
) -> RoadGraph:
    handler = RoadGraphHandler(config)
    handler.apply_file(str(pbf_path), locations=True)
    node_count = len(handler.longitude)
    if node_count == 0 or not handler.edges:
        raise RuntimeError("No routable OpenStreetMap network was extracted")

    sources = np.fromiter(
        (edge[0] for edge in handler.edges),
        dtype=np.int64,
        count=len(handler.edges),
    )
    targets = np.fromiter(
        (edge[1] for edge in handler.edges),
        dtype=np.int64,
        count=len(handler.edges),
    )
    minutes = np.fromiter(
        (value[0] for value in handler.edges.values()),
        dtype=float,
        count=len(handler.edges),
    )
    ferry_mask = np.fromiter(
        (value[1] for value in handler.edges.values()),
        dtype=bool,
        count=len(handler.edges),
    )
    matrix = csr_matrix((minutes, (sources, targets)), shape=(node_count, node_count))
    transformer = Transformer.from_crs("EPSG:4326", projected_crs, always_xy=True)
    x, y = transformer.transform(
        np.asarray(handler.longitude),
        np.asarray(handler.latitude),
    )
    ferry_edges = {
        int(source) * node_count + int(target)
        for source, target in zip(sources[ferry_mask], targets[ferry_mask], strict=True)
    }
    metadata = {
        "nodes": node_count,
        "directed_edges": int(matrix.nnz),
        "included_ways": handler.included_ways,
        "included_ferry_ways": handler.included_ferry_ways,
        "directed_ferry_edges": int(ferry_mask.sum()),
        "bounding_box_wgs84": list(handler.bounds),
        "profile_sha256": profile_sha256,
    }
    return RoadGraph(
        matrix=matrix,
        x=np.asarray(x),
        y=np.asarray(y),
        ferry_edges=ferry_edges,
        metadata=metadata,
    )


def save_graph(graph: RoadGraph, path: Path, source_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ferry_edges = np.fromiter(graph.ferry_edges, dtype=np.uint64)
    with path.open("wb") as stream:
        np.savez(
            stream,
            data=graph.matrix.data,
            indices=graph.matrix.indices,
            indptr=graph.matrix.indptr,
            shape=np.asarray(graph.matrix.shape),
            x=graph.x,
            y=graph.y,
            ferry_edges=ferry_edges,
            metadata=json.dumps(graph.metadata),
            source_sha256=source_sha256,
            profile_sha256=graph.metadata["profile_sha256"],
        )


def load_graph(
    path: Path,
    source_sha256: str,
    profile_sha256: str,
) -> RoadGraph | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as cached:
        if (
            str(cached["source_sha256"]) != source_sha256
            or "profile_sha256" not in cached
            or str(cached["profile_sha256"]) != profile_sha256
        ):
            return None
        shape = tuple(int(value) for value in cached["shape"])
        matrix = csr_matrix(
            (cached["data"], cached["indices"], cached["indptr"]),
            shape=shape,
        )
        return RoadGraph(
            matrix=matrix,
            x=cached["x"],
            y=cached["y"],
            ferry_edges={int(value) for value in cached["ferry_edges"]},
            metadata=json.loads(str(cached["metadata"])),
        )


def graph_profile_sha256(config: dict, projected_crs: str) -> str:
    network = config["network_validation"]
    profile = {
        "bounding_box_wgs84": network["bounding_box_wgs84"],
        "road_speeds_kmh": network["road_speeds_kmh"],
        "ferry_speed_kmh": network["ferry_speed_kmh"],
        "projected_crs": projected_crs,
    }
    payload = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def snapped_nodes(
    points: pd.DataFrame,
    graph: RoadGraph,
    eligible_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(np.column_stack((graph.x[eligible_nodes], graph.y[eligible_nodes])))
    distance, position = tree.query(points[["x", "y"]].to_numpy())
    return eligible_nodes[np.asarray(position, dtype=int)], np.asarray(distance, dtype=float)


def path_uses_ferry(
    source: int,
    target: int,
    predecessors: np.ndarray,
    ferry_edges: set[int],
    node_count: int,
) -> bool:
    current = int(target)
    visited = 0
    while current != source:
        previous = int(predecessors[current])
        if previous == NO_PREDECESSOR:
            return False
        if previous * node_count + current in ferry_edges:
            return True
        current = previous
        visited += 1
        if visited > node_count:
            raise RuntimeError("Cycle encountered while reconstructing shortest path")
    return False


def weighted_mean(frame: pd.DataFrame, column: str) -> float:
    return float(np.average(frame[column], weights=frame["population"]))


def metrics(frame: pd.DataFrame, threshold_minutes: float) -> dict[str, float | int]:
    if frame.empty:
        return {"pairs": 0}
    proxy = frame["proxy_minutes"].to_numpy()
    network = frame["network_minutes"].to_numpy()
    absolute_error = np.abs(proxy - network)
    relative_error = absolute_error / np.maximum(network, 1e-9)
    correlation = spearmanr(proxy, network).statistic
    agreement = (proxy <= threshold_minutes) == (network <= threshold_minutes)
    frame = frame.copy()
    frame["absolute_error"] = absolute_error
    frame["relative_error"] = relative_error
    frame["threshold_agreement"] = agreement.astype(float)
    return {
        "pairs": int(len(frame)),
        "population": float(frame["population"].sum()),
        "spearman_rho": float(correlation),
        "mean_absolute_error_minutes": float(absolute_error.mean()),
        "median_absolute_error_minutes": float(np.median(absolute_error)),
        "population_weighted_mean_absolute_error_minutes": weighted_mean(
            frame,
            "absolute_error",
        ),
        "median_absolute_relative_error": float(np.median(relative_error)),
        "threshold_agreement": float(agreement.mean()),
        "population_weighted_threshold_agreement": weighted_mean(
            frame,
            "threshold_agreement",
        ),
        "proxy_median_minutes": float(np.median(proxy)),
        "network_median_minutes": float(np.median(network)),
    }


def validate_routes(graph: RoadGraph, config: dict) -> tuple[pd.DataFrame, dict[str, object]]:
    reporting = reporting_parameters(config)
    stem = (
        f"D_rf{reporting['central_rf']:g}_"
        f"decline{int(round(reporting['baseline_decline'] * 100))}"
    )
    demand = pd.read_csv(
        DETAILS / f"{stem}_demand.csv",
        dtype={"municipality_code": str, "pref_code": str},
    )
    demand_source = pd.read_csv(
        PROCESSED / "demand.csv",
        dtype={"municipality_code": str, "pref_code": str},
    )
    demand = demand.merge(
        demand_source[["demand_id", "source_cells"]],
        on="demand_id",
        validate="one_to_one",
    )
    demand["density_proxy"] = demand["population"] / demand["source_cells"]
    demand["density_tercile"] = pd.qcut(
        demand["density_proxy"],
        q=3,
        labels=["low", "middle", "high"],
    ).astype(str)
    facilities = pd.read_csv(
        DETAILS / f"{stem}_facilities.csv",
        dtype={"municipality_code": str},
    )
    open_facilities = facilities[facilities["open"] > 0.5].copy().reset_index(drop=True)
    component_count, component_labels = connected_components(
        graph.matrix,
        directed=True,
        connection="weak",
    )
    component_sizes = np.bincount(component_labels)
    minimum_component_nodes = int(
        config["network_validation"]["minimum_routable_component_nodes"]
    )
    eligible_nodes = np.flatnonzero(
        component_sizes[component_labels] >= minimum_component_nodes
    )
    demand_nodes, demand_snap = snapped_nodes(demand, graph, eligible_nodes)
    facility_nodes, facility_snap = snapped_nodes(
        open_facilities,
        graph,
        eligible_nodes,
    )
    connector_speed = float(config["network_validation"]["connector_speed_kmh"])
    demand_connector = demand_snap / 1000 / connector_speed * 60
    facility_connector = facility_snap / 1000 / connector_speed * 60
    facility_lookup = {
        facility_id: index
        for index, facility_id in enumerate(open_facilities["facility_id"])
    }
    projected_distance = np.hypot(
        demand["x"].to_numpy()[:, None] - open_facilities["x"].to_numpy()[None, :],
        demand["y"].to_numpy()[:, None] - open_facilities["y"].to_numpy()[None, :],
    )
    proxy_matrix = (
        projected_distance
        / 1000
        * float(config["geography"]["assumed_road_circuity"])
        / float(config["geography"]["assumed_average_speed_kmh"])
        * 60
    )
    proxy_nearest = proxy_matrix.argmin(axis=1)
    network_matrix = np.full(proxy_matrix.shape, np.inf)
    dominant_ferry = np.zeros(len(demand), dtype=bool)
    network_nearest_ferry = np.zeros(len(demand), dtype=bool)
    node_count = graph.matrix.shape[0]
    for demand_index, source in enumerate(demand_nodes):
        distances, predecessors = dijkstra(
            graph.matrix,
            directed=True,
            indices=int(source),
            return_predecessors=True,
        )
        route_minutes = distances[facility_nodes]
        network_matrix[demand_index] = (
            route_minutes + demand_connector[demand_index] + facility_connector
        )
        dominant_position = facility_lookup[demand.loc[demand_index, "dominant_facility_id"]]
        if np.isfinite(route_minutes[dominant_position]):
            dominant_ferry[demand_index] = path_uses_ferry(
                int(source),
                int(facility_nodes[dominant_position]),
                predecessors,
                graph.ferry_edges,
                node_count,
            )
        reachable = np.flatnonzero(np.isfinite(route_minutes))
        if len(reachable):
            nearest_position = reachable[
                np.argmin(network_matrix[demand_index, reachable])
            ]
            network_nearest_ferry[demand_index] = path_uses_ferry(
                int(source),
                int(facility_nodes[nearest_position]),
                predecessors,
                graph.ferry_edges,
                node_count,
            )
    network_nearest = network_matrix.argmin(axis=1)

    records = []
    for index, row in demand.iterrows():
        dominant_position = facility_lookup[row["dominant_facility_id"]]
        proxy_nearest_position = int(proxy_nearest[index])
        network_nearest_position = int(network_nearest[index])
        dominant_facility = open_facilities.loc[dominant_position]
        proxy_nearest_facility = open_facilities.loc[proxy_nearest_position]
        network_nearest_facility = open_facilities.loc[network_nearest_position]
        network_minutes = network_matrix[index, dominant_position]
        records.append(
            {
                "demand_id": row["demand_id"],
                "pref_code": row["pref_code"],
                "municipality_code": row["municipality_code"],
                "density_tercile": row["density_tercile"],
                "population": row["population"],
                "dominant_mode": row["dominant_mode"],
                "dominant_facility_id": row["dominant_facility_id"],
                "dominant_cross_boundary": (
                    str(dominant_facility["municipality_code"])
                    != str(row["municipality_code"])
                ),
                "proxy_minutes": proxy_matrix[index, dominant_position],
                "network_minutes": network_minutes,
                "reachable": bool(np.isfinite(network_minutes)),
                "uses_ferry": bool(dominant_ferry[index]),
                "demand_snap_metres": demand_snap[index],
                "facility_snap_metres": facility_snap[dominant_position],
                "well_snapped": bool(
                    max(demand_snap[index], facility_snap[dominant_position])
                    <= float(
                        config["network_validation"]["well_snapped_maximum_metres"]
                    )
                ),
                "proxy_within_45_minutes": bool(
                    proxy_matrix[index, dominant_position]
                    <= config["mfg"]["healthcare_max_minutes"]
                ),
                "network_within_45_minutes": bool(
                    network_minutes <= config["mfg"]["healthcare_max_minutes"]
                ),
                "proxy_nearest_facility_id": proxy_nearest_facility["facility_id"],
                "network_nearest_facility_id": network_nearest_facility["facility_id"],
                "same_nearest_facility": bool(
                    proxy_nearest_position == network_nearest_position
                ),
                "same_nearest_municipality": bool(
                    str(proxy_nearest_facility["municipality_code"])
                    == str(network_nearest_facility["municipality_code"])
                ),
                "proxy_nearest_cross_boundary": bool(
                    str(proxy_nearest_facility["municipality_code"])
                    != str(row["municipality_code"])
                ),
                "network_nearest_cross_boundary": bool(
                    str(network_nearest_facility["municipality_code"])
                    != str(row["municipality_code"])
                ),
                "network_nearest_uses_ferry": bool(network_nearest_ferry[index]),
            }
        )
    pairs = pd.DataFrame(records)
    fixed_reachable = pairs[
        (pairs["dominant_mode"] == "fixed")
        & pairs["reachable"]
        & ~pairs["uses_ferry"]
    ].copy()
    fixed_well_snapped = fixed_reachable[fixed_reachable["well_snapped"]].copy()
    threshold = float(config["mfg"]["healthcare_max_minutes"])
    nearest_reachable = pairs[np.isfinite(network_matrix).any(axis=1)].copy()
    nearest_reachable["same_nearest_facility_float"] = nearest_reachable[
        "same_nearest_facility"
    ].astype(float)
    nearest_reachable["same_nearest_municipality_float"] = nearest_reachable[
        "same_nearest_municipality"
    ].astype(float)
    nearest_reachable["proxy_nearest_cross_boundary_float"] = nearest_reachable[
        "proxy_nearest_cross_boundary"
    ].astype(float)
    nearest_reachable["network_nearest_cross_boundary_float"] = nearest_reachable[
        "network_nearest_cross_boundary"
    ].astype(float)
    summary: dict[str, object] = {
        "design": {
            "pair_definition": (
                "Every central Scenario-D demand cluster and its dominant open fixed "
                "destination; threshold metrics exclude relocation-mode and "
                "schedule-dependent ferry paths"
            ),
            "demand_clusters": int(len(demand)),
            "open_fixed_facilities": int(len(open_facilities)),
            "network_snapshot": config["network_validation"]["snapshot"],
            "threshold_minutes": threshold,
            "connector_speed_kmh": connector_speed,
            "well_snapped_maximum_metres": config["network_validation"][
                "well_snapped_maximum_metres"
            ],
        },
        "graph": graph.metadata,
        "all_eligible_fixed_pairs": metrics(fixed_reachable, threshold),
        "well_snapped_fixed_pairs": metrics(fixed_well_snapped, threshold),
        "route_counts": {
            "dominant_fixed_pairs": int((pairs["dominant_mode"] == "fixed").sum()),
            "dominant_relocation_pairs": int(
                (pairs["dominant_mode"] == "relocation").sum()
            ),
            "unreachable_dominant_pairs": int((~pairs["reachable"]).sum()),
            "dominant_pairs_using_ferry": int(pairs["uses_ferry"].sum()),
            "well_snapped_dominant_pairs": int(pairs["well_snapped"].sum()),
        },
        "nearest_open_facility_stability": {
            "same_facility_share": float(
                nearest_reachable["same_nearest_facility"].mean()
            ),
            "population_weighted_same_facility_share": weighted_mean(
                nearest_reachable,
                "same_nearest_facility_float",
            ),
            "same_municipality_share": float(
                nearest_reachable["same_nearest_municipality"].mean()
            ),
            "population_weighted_same_municipality_share": weighted_mean(
                nearest_reachable,
                "same_nearest_municipality_float",
            ),
            "proxy_population_weighted_cross_boundary_share": weighted_mean(
                nearest_reachable,
                "proxy_nearest_cross_boundary_float",
            ),
            "network_population_weighted_cross_boundary_share": weighted_mean(
                nearest_reachable,
                "network_nearest_cross_boundary_float",
            ),
        },
        "density": {
            density: metrics(group, threshold)
            for density, group in fixed_reachable.groupby("density_tercile")
        },
        "prefecture": {
            str(code): metrics(group, threshold)
            for code, group in fixed_reachable.groupby("pref_code")
        },
        "boundary_status": {
            (
                "cross_boundary" if bool(is_cross_boundary) else "within_boundary"
            ): metrics(group, threshold)
            for is_cross_boundary, group in fixed_reachable.groupby(
                "dominant_cross_boundary"
            )
        },
    }
    proxy_time_band = pd.cut(
        fixed_reachable["proxy_minutes"],
        bins=[-np.inf, 15.0, 30.0, 45.0, np.inf],
        labels=["0-15", "15-30", "30-45", "over-45"],
    )
    summary["proxy_time_band"] = {
        str(label): metrics(group, threshold)
        for label, group in fixed_reachable.groupby(proxy_time_band, observed=True)
    }
    summary["graph"]["weak_components"] = int(component_count)
    summary["graph"]["minimum_routable_component_nodes"] = minimum_component_nodes
    summary["graph"]["eligible_snapping_nodes"] = int(len(eligible_nodes))
    summary["unreachable_pairs"] = pairs.loc[
        ~pairs["reachable"],
        [
            "demand_id",
            "municipality_code",
            "population",
            "dominant_mode",
            "dominant_facility_id",
        ],
    ].to_dict(orient="records")
    return pairs, summary


def render_report(summary: dict[str, object], source: Path, source_sha256: str) -> str:
    eligible = summary["all_eligible_fixed_pairs"]
    snapped = summary["well_snapped_fixed_pairs"]
    counts = summary["route_counts"]
    nearest = summary["nearest_open_facility_stability"]
    graph = summary["graph"]
    return f"""# Targeted road-network validation

## Design

- Source: Geofabrik Chūgoku OpenStreetMap PBF snapshot dated
  `{summary['design']['network_snapshot']}`.
- Local immutable file: `{source.relative_to(ROOT)}`.
- SHA-256: `{source_sha256}`.
- Licence: OpenStreetMap contributors, Open Database License 1.0.
- Predefined pairs: all central Scenario-D demand clusters and their dominant
  open fixed destination ({counts['dominant_fixed_pairs']} fixed-mode pairs and
  {counts['dominant_relocation_pairs']} relocation-mode pairs).
- Graph: {graph['nodes']:,} nodes, {graph['directed_edges']:,} directed road/ferry
  edges, and {graph['included_ferry_ways']:,} included ferry ways within the
  predefined buffered bounding box.
- Routing uses posted `maxspeed` where parseable and declared road-class default
  speeds otherwise. Straight connectors from clustered demand/facility points to
  the nearest routable node use
  {summary['design']['connector_speed_kmh']:.0f} km/h.
- Snapping excludes road components with fewer than
  {graph['minimum_routable_component_nodes']} nodes to avoid isolated driveways
  and digitization fragments.

This is a reproducible network validation of the primary proxy, not observed
journey time. It omits congestion, signals, parking, appointment access, and
ferry waiting/schedules.

## Results

Among {eligible['pairs']} reachable, non-ferry, dominant fixed-mode pairs, proxy
and network time had Spearman rho {eligible['spearman_rho']:.3f}. Median absolute
error was {eligible['median_absolute_error_minutes']:.1f} minutes, median absolute
relative error was {eligible['median_absolute_relative_error']:.1%}, and
45-minute classification agreement was {eligible['threshold_agreement']:.1%}
({eligible['population_weighted_threshold_agreement']:.1%}
population-weighted). The network median was
{eligible['network_median_minutes']:.1f} minutes versus
{eligible['proxy_median_minutes']:.1f} proxy minutes.

The prespecified well-snapped subset (both endpoint connectors no more than
{summary['design']['well_snapped_maximum_metres']:.0f} m) contained
{snapped['pairs']} pairs. Its Spearman rho was
{snapped.get('spearman_rho', float('nan')):.3f}, median absolute error was
{snapped.get('median_absolute_error_minutes', float('nan')):.1f} minutes, and
45-minute agreement was
{snapped.get('threshold_agreement', float('nan')):.1%}.

The proxy-nearest and network-nearest open facility were identical for
{nearest['population_weighted_same_facility_share']:.1%} of represented
population and lay in the same municipality for
{nearest['population_weighted_same_municipality_share']:.1%}. The
population-weighted cross-boundary share for the nearest-open-facility diagnostic
was {nearest['proxy_population_weighted_cross_boundary_share']:.1%} under the
proxy and {nearest['network_population_weighted_cross_boundary_share']:.1%}
under network routing. This diagnostic does not replace the capacity-constrained
optimization assignment.

There were {counts['unreachable_dominant_pairs']} unreachable dominant pairs and
{counts['dominant_pairs_using_ferry']} dominant paths using a ferry. Ferry paths
are excluded from the primary error and threshold metrics because the PBF does
not provide schedule or waiting time. Unreachable pairs are listed in the
machine-readable summary rather than assigned invented travel times.

## Interpretation rule

The validation supports comparative use of the proxy only to the extent that
rank order, threshold classification, and the direction of the cross-boundary
diagnostic remain stable. It cannot convert the primary analysis into a
policy-calibrated road-time study. A class-D primary-model correction is required
only if network disagreement changes a substantial share of threshold
classifications or assignment geography and undermines a headline comparative
conclusion.

The observed agreement is sufficiently high to retain the primary proxy and
avoid a class-D model rewrite. The correct scope remains a comparative planning
stress test: the validation supports the direction of the functional-geography
result, not precise local journey-time calibration.
"""


def main() -> None:
    config = load_config()
    network_config = config["network_validation"]
    pbf_path = ROOT / network_config["pbf_path"]
    if not pbf_path.exists():
        raise FileNotFoundError(
            f"Missing {pbf_path}; run `make download-analysis` before validation"
        )
    source_sha256 = sha256(pbf_path)
    cache_path = ROOT / network_config["cache_path"]
    projected_crs = config["geography"]["projected_crs"]
    profile_sha256 = graph_profile_sha256(config, projected_crs)
    graph = load_graph(cache_path, source_sha256, profile_sha256)
    if graph is None:
        graph = build_graph(
            pbf_path,
            network_config,
            projected_crs,
            profile_sha256,
        )
        save_graph(graph, cache_path, source_sha256)
    pairs, summary = validate_routes(graph, config)
    summary["source"] = {
        "path": str(pbf_path.relative_to(ROOT)),
        "sha256": source_sha256,
        "file_size_bytes": pbf_path.stat().st_size,
    }
    VALIDATION.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(VALIDATION / "network_travel_pairs.csv", index=False)
    write_json(VALIDATION / "network_travel_summary.json", summary)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(
        render_report(summary, pbf_path, source_sha256),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
