from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.network_travel_validation import (
    NO_PREDECESSOR,
    directed_permissions,
    graph_profile_sha256,
    metrics,
    parse_maxspeed,
    path_uses_ferry,
    way_profile,
)


def test_parse_maxspeed_handles_metric_and_mph() -> None:
    assert parse_maxspeed("50") == 50.0
    assert np.isclose(parse_maxspeed("30 mph"), 48.28032)
    assert parse_maxspeed("signals") is None


def test_way_profile_filters_private_and_uses_defaults() -> None:
    config = {
        "road_speeds_kmh": {"primary": 50.0, "service": 20.0},
        "ferry_speed_kmh": 20.0,
    }
    assert way_profile({"highway": "primary"}, config) == (50.0, False)
    assert way_profile({"route": "ferry"}, config) == (20.0, True)
    assert way_profile({"highway": "primary", "access": "private"}, config) is None
    assert way_profile(
        {"highway": "service", "service": "parking_aisle"},
        config,
    ) is None


def test_directed_permissions_respect_oneway_direction() -> None:
    assert directed_permissions({}) == (True, True)
    assert directed_permissions({"oneway": "yes"}) == (True, False)
    assert directed_permissions({"oneway": "-1"}) == (False, True)
    assert directed_permissions({"junction": "roundabout"}) == (True, False)


def test_path_uses_ferry_reconstructs_predecessors() -> None:
    predecessors = np.array([NO_PREDECESSOR, 0, 1, 2])
    node_count = 4
    assert path_uses_ferry(0, 3, predecessors, {1 * node_count + 2}, node_count)
    assert not path_uses_ferry(0, 1, predecessors, {1 * node_count + 2}, node_count)


def test_metrics_reports_threshold_agreement() -> None:
    frame = pd.DataFrame(
        {
            "proxy_minutes": [20.0, 50.0, 40.0],
            "network_minutes": [25.0, 55.0, 60.0],
            "population": [1.0, 2.0, 1.0],
        }
    )
    result = metrics(frame, 45.0)
    assert result["pairs"] == 3
    assert np.isclose(result["threshold_agreement"], 2 / 3)
    assert np.isclose(result["population_weighted_threshold_agreement"], 0.75)


def test_graph_profile_hash_changes_with_routing_assumptions() -> None:
    config = {
        "network_validation": {
            "bounding_box_wgs84": [131.3, 33.9, 134.9, 36.7],
            "road_speeds_kmh": {"primary": 50.0},
            "ferry_speed_kmh": 20.0,
        }
    }
    baseline = graph_profile_sha256(config, "EPSG:6673")
    config["network_validation"]["road_speeds_kmh"]["primary"] = 45.0
    assert graph_profile_sha256(config, "EPSG:6673") != baseline
