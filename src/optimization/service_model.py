from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

MODEL_VERSION = "2026-09-24.4"


@dataclass
class ModelSolution:
    summary: dict[str, float | int | str | bool]
    demand: pd.DataFrame
    facilities: pd.DataFrame


class LinearModel:
    def __init__(self) -> None:
        self.cost: list[float] = []
        self.lower: list[float] = []
        self.upper: list[float] = []
        self.integrality: list[int] = []
        self.names: list[str] = []
        self.row: list[int] = []
        self.column: list[int] = []
        self.value: list[float] = []
        self.constraint_lower: list[float] = []
        self.constraint_upper: list[float] = []

    def variable(
        self,
        name: str,
        cost: float = 0.0,
        lower: float = 0.0,
        upper: float = np.inf,
        integer: bool = False,
    ) -> int:
        index = len(self.cost)
        self.names.append(name)
        self.cost.append(cost)
        self.lower.append(lower)
        self.upper.append(upper)
        self.integrality.append(1 if integer else 0)
        return index

    def constraint(
        self,
        coefficients: dict[int, float],
        lower: float = -np.inf,
        upper: float = np.inf,
    ) -> None:
        row = len(self.constraint_lower)
        for column, value in coefficients.items():
            if value:
                self.row.append(row)
                self.column.append(column)
                self.value.append(float(value))
        self.constraint_lower.append(lower)
        self.constraint_upper.append(upper)

    def solve(self, time_limit: float, mip_rel_gap: float):
        matrix = coo_matrix(
            (self.value, (self.row, self.column)),
            shape=(len(self.constraint_lower), len(self.cost)),
        ).tocsr()
        return milp(
            c=np.asarray(self.cost),
            integrality=np.asarray(self.integrality),
            bounds=Bounds(np.asarray(self.lower), np.asarray(self.upper)),
            constraints=LinearConstraint(
                matrix,
                np.asarray(self.constraint_lower),
                np.asarray(self.constraint_upper),
            ),
            options={
                "time_limit": time_limit,
                "mip_rel_gap": mip_rel_gap,
                "presolve": True,
            },
        )


def weighted_gini(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    x = values[order]
    w = weights[order]
    if not len(x) or w.sum() <= 0 or np.average(x, weights=w) <= 0:
        return 0.0
    cumulative_w = np.cumsum(w)
    cumulative_xw = np.cumsum(x * w)
    relative_w = np.insert(cumulative_w / cumulative_w[-1], 0, 0)
    relative_xw = np.insert(cumulative_xw / cumulative_xw[-1], 0, 0)
    return float(1 - np.sum((relative_xw[1:] + relative_xw[:-1]) * np.diff(relative_w)))


def unit_value(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def candidate_edges(
    demand: pd.DataFrame,
    facilities: pd.DataFrame,
    scenario: str,
    same_municipality_limit: int,
    regional_limit: int,
) -> list[tuple[int, int, float]]:
    edges: set[tuple[int, int]] = set()
    candidates = facilities[facilities["is_candidate"]].copy()
    for demand_index, node in demand.iterrows():
        same = candidates[candidates["municipality_code"] == node["municipality_code"]]
        same_distance = np.hypot(same["x"] - node["x"], same["y"] - node["y"])
        for facility_index in same_distance.nsmallest(same_municipality_limit).index:
            edges.add((demand_index, facility_index))
        if scenario in {"A", "D"}:
            distance = np.hypot(candidates["x"] - node["x"], candidates["y"] - node["y"])
            for facility_index in distance.nsmallest(regional_limit).index:
                edges.add((demand_index, facility_index))
    result = []
    for demand_index, facility_index in sorted(edges):
        distance_km = float(
            np.hypot(
                facilities.loc[facility_index, "x"] - demand.loc[demand_index, "x"],
                facilities.loc[facility_index, "y"] - demand.loc[demand_index, "y"],
            )
            / 1000
        )
        result.append((demand_index, facility_index, distance_km))
    return result


def facility_capacity(row: pd.Series, config: dict) -> float:
    capacity = config["capacity"]
    if row["facility_class"] == "hospital":
        return max(
            capacity["hospital_minimum_residents"],
            float(row["beds"]) * capacity["hospital_residents_per_bed"],
        )
    if row["facility_class"] == "clinic":
        return max(
            capacity["clinic_minimum_residents"],
            float(row["beds"]) * capacity["clinic_residents_per_bed"],
        )
    return capacity["planned_hub_residents"]


def solve_service_model(
    demand_input: pd.DataFrame,
    facilities_input: pd.DataFrame,
    config: dict,
    scenario: str,
    relocation_friction: float,
    population_decline: float,
    *,
    substitution_multiplier: float = 1.0,
    mobile_cost_multiplier: float = 1.0,
    digital_cost_multiplier: float = 1.0,
    impedance_multiplier: float = 1.0,
    capacity_multiplier: float = 1.0,
    mfg_multiplier: float = 1.0,
    time_limit: float | None = None,
    mip_rel_gap: float | None = None,
) -> ModelSolution:
    started = perf_counter()
    demand = demand_input.copy().reset_index(drop=True)
    facilities = facilities_input.copy().reset_index(drop=True)
    demand["population"] *= 1 - population_decline
    candidates = facilities.index[facilities["is_candidate"]].tolist()
    municipalities = sorted(demand["municipality_code"].astype(str).unique())
    edge_config = config["candidate_edges"]
    edges = candidate_edges(
        demand,
        facilities,
        scenario,
        edge_config["same_municipality_limit"],
        edge_config["regional_limit"],
    )
    edge_map: dict[int, list[tuple[int, float]]] = {index: [] for index in demand.index}
    for demand_index, facility_index, distance_km in edges:
        edge_map[demand_index].append((facility_index, distance_km))

    model = LinearModel()
    costs = config["costs"]
    substitution = config["substitution"]
    threshold_minutes = config["mfg"]["healthcare_max_minutes"] * mfg_multiplier
    circuity = config["geography"]["assumed_road_circuity"] * impedance_multiplier
    speed = config["geography"]["assumed_average_speed_kmh"]

    y: dict[int, int] = {}
    existing_candidates = 0
    for facility_index in candidates:
        facility = facilities.loc[facility_index]
        existing = bool(facility["is_existing"])
        if existing:
            existing_candidates += 1
        if scenario == "A":
            lower = upper = 1.0 if existing else 0.0
        else:
            lower, upper = 0.0, 1.0
        transition_adjustment = -costs["transition_per_changed_asset"] if existing else 0.0
        opening_transition = 0.0 if existing else costs["transition_per_changed_asset"]
        y[facility_index] = model.variable(
            f"open[{facility_index}]",
            cost=costs["fixed_facility_open"]
            + facility_capacity(facility, config)
            * costs["fixed_facility_capacity_unit"]
            + transition_adjustment
            + opening_transition,
            lower=lower,
            upper=upper,
            integer=True,
        )

    mobile_hub: dict[str, int] = {}
    for municipality in municipalities:
        enabled = scenario in {"C", "D"}
        mobile_hub[municipality] = model.variable(
            f"mobile_hub[{municipality}]",
            cost=costs["mobile_hub_open"],
            upper=1.0 if enabled else 0.0,
            integer=True,
        )

    direct: dict[tuple[int, int], int] = {}
    relocated: dict[tuple[int, int], int] = {}
    edge_distance: dict[tuple[int, int], float] = {}
    mobile: dict[int, int] = {}
    digital: dict[int, int] = {}
    slack: dict[int, int] = {}
    for demand_index, node in demand.iterrows():
        population = float(node["population"])
        for facility_index, distance_km in edge_map[demand_index]:
            minutes = distance_km * circuity / speed * 60
            edge_distance[(demand_index, facility_index)] = distance_km
            direct[(demand_index, facility_index)] = model.variable(
                f"direct[{demand_index},{facility_index}]",
                cost=population * minutes * costs["access_per_person_minute"],
                upper=1.0,
            )
            if scenario != "A":
                relocated[(demand_index, facility_index)] = model.variable(
                    f"relocated[{demand_index},{facility_index}]",
                    cost=(
                        population
                        * distance_km
                        * costs["relocation_per_person_km"]
                        * relocation_friction
                    ),
                    upper=1.0,
                )
        mobile[demand_index] = model.variable(
            f"mobile[{demand_index}]",
            cost=population
            * costs["mobile_service_per_resident"]
            * substitution_multiplier
            * mobile_cost_multiplier,
            upper=(
                substitution["healthcare_mobile_max_share"]
                if scenario in {"C", "D"}
                else 0.0
            ),
        )
        digital[demand_index] = model.variable(
            f"digital[{demand_index}]",
            cost=population
            * costs["digital_service_per_resident"]
            * substitution_multiplier
            * digital_cost_multiplier,
            upper=(
                substitution["healthcare_digital_max_share"]
                if scenario in {"C", "D"}
                else 0.0
            ),
        )
        slack[demand_index] = model.variable(
            f"mfg_slack[{demand_index}]",
            cost=population * config["mfg"]["permitted_diagnostic_slack_penalty"],
            upper=1.0,
        )

    for demand_index, node in demand.iterrows():
        share: dict[int, float] = {
            direct[(demand_index, facility_index)]: 1.0
            for facility_index, _ in edge_map[demand_index]
        }
        for facility_index, _ in edge_map[demand_index]:
            variable = relocated.get((demand_index, facility_index))
            if variable is not None:
                share[variable] = 1.0
        share[mobile[demand_index]] = 1.0
        share[digital[demand_index]] = 1.0
        model.constraint(share, lower=1.0, upper=1.0)

        quality: dict[int, float] = {}
        for facility_index, distance_km in edge_map[demand_index]:
            minutes = distance_km * circuity / speed * 60
            access_quality = min(1.0, threshold_minutes / max(threshold_minutes, minutes))
            quality[direct[(demand_index, facility_index)]] = access_quality
            relocation_variable = relocated.get((demand_index, facility_index))
            if relocation_variable is not None:
                quality[relocation_variable] = 1.0
        quality[mobile[demand_index]] = substitution["mobile_quality_effectiveness"]
        quality[digital[demand_index]] = substitution["digital_quality_effectiveness"]
        quality[slack[demand_index]] = 1.0
        model.constraint(
            quality,
            lower=config["mfg"]["minimum_effective_provision"],
        )

        model.constraint(
            {
                mobile[demand_index]: 1.0,
                digital[demand_index]: 1.0,
            },
            upper=substitution["healthcare_combined_max_share"],
        )
        municipality = str(node["municipality_code"])
        model.constraint(
            {
                mobile[demand_index]: 1.0,
                mobile_hub[municipality]: -substitution[
                    "healthcare_mobile_max_share"
                ],
            },
            upper=0.0,
        )

        for facility_index, _ in edge_map[demand_index]:
            model.constraint(
                {
                    direct[(demand_index, facility_index)]: 1.0,
                    y[facility_index]: -1.0,
                },
                upper=0.0,
            )
            relocation_variable = relocated.get((demand_index, facility_index))
            if relocation_variable is not None:
                model.constraint(
                    {relocation_variable: 1.0, y[facility_index]: -1.0},
                    upper=0.0,
                )

    for facility_index in candidates:
        coefficients = {}
        for demand_index, node in demand.iterrows():
            population = float(node["population"])
            direct_variable = direct.get((demand_index, facility_index))
            if direct_variable is not None:
                coefficients[direct_variable] = population
            relocation_variable = relocated.get((demand_index, facility_index))
            if relocation_variable is not None:
                coefficients[relocation_variable] = population
        coefficients[y[facility_index]] = (
            -facility_capacity(facilities.loc[facility_index], config)
            * capacity_multiplier
        )
        model.constraint(coefficients, upper=0.0)

    for municipality in municipalities:
        member = demand["municipality_code"].astype(str) == municipality
        coefficients = {
            mobile[index]: float(demand.loc[index, "population"])
            for index in demand.index[member]
        }
        municipal_population = float(demand.loc[member, "population"].sum())
        coefficients[mobile_hub[municipality]] = (
            -substitution["mobile_hub_capacity_share"] * municipal_population
        )
        model.constraint(coefficients, upper=0.0)

    declared_mip_gap = (
        config["solver"]["mip_relative_gap"]
        if mip_rel_gap is None
        else mip_rel_gap
    )
    result = model.solve(
        time_limit=(
            config["solver"]["time_limit_seconds"]
            if time_limit is None
            else time_limit
        ),
        mip_rel_gap=declared_mip_gap,
    )
    if result.x is None:
        raise RuntimeError(
            f"Optimization failed for {scenario=}, {relocation_friction=}, "
            f"{population_decline=}: status={result.status}, message={result.message}"
        )
    values = result.x
    achieved_mip_gap = float(result.get("mip_gap", np.nan))
    solver_success = bool(
        result.success
        or (
            np.isfinite(achieved_mip_gap)
            and achieved_mip_gap <= declared_mip_gap + 1e-9
        )
    )
    transition_constant = existing_candidates * costs["transition_per_changed_asset"]

    demand_records = []
    cross_boundary_population = 0.0
    direct_times = []
    direct_weights = []
    facility_direct_load = {index: 0.0 for index in candidates}
    facility_relocated_load = {index: 0.0 for index in candidates}
    facility_cross_boundary_load = {index: 0.0 for index in candidates}
    for demand_index, node in demand.iterrows():
        fixed_share = 0.0
        relocated_share = 0.0
        weighted_minutes = 0.0
        node_cross_boundary_share = 0.0
        facility_shares: dict[int, float] = {}
        for facility_index, distance_km in edge_map[demand_index]:
            direct_value = unit_value(values[direct[(demand_index, facility_index)]])
            relocated_value = (
                unit_value(values[relocated[(demand_index, facility_index)]])
                if (demand_index, facility_index) in relocated
                else 0.0
            )
            fixed_share += direct_value
            relocated_share += relocated_value
            facility_shares[facility_index] = direct_value + relocated_value
            facility_direct_load[facility_index] += float(node["population"]) * direct_value
            facility_relocated_load[facility_index] += (
                float(node["population"]) * relocated_value
            )
            minutes = distance_km * circuity / speed * 60
            weighted_minutes += direct_value * minutes
            if (
                str(node["municipality_code"])
                != str(facilities.loc[facility_index, "municipality_code"])
            ):
                cross_boundary_value = direct_value + relocated_value
                node_cross_boundary_share += cross_boundary_value
                cross_boundary_population += (
                    float(node["population"]) * cross_boundary_value
                )
                facility_cross_boundary_load[facility_index] += (
                    float(node["population"]) * cross_boundary_value
                )
        mobile_share = unit_value(values[mobile[demand_index]])
        digital_share = unit_value(values[digital[demand_index]])
        slack_value = unit_value(values[slack[demand_index]])
        mode_shares = {
            "fixed": fixed_share,
            "relocation": relocated_share,
            "mobile": mobile_share,
            "digital": digital_share,
        }
        dominant_mode = max(mode_shares, key=mode_shares.get)
        dominant_facility_index = (
            max(facility_shares, key=facility_shares.get)
            if dominant_mode in {"fixed", "relocation"} and facility_shares
            else None
        )
        direct_times.append(weighted_minutes)
        direct_weights.append(float(node["population"]))
        demand_records.append(
            {
                "demand_id": node["demand_id"],
                "municipality_code": str(node["municipality_code"]),
                "pref_code": str(node["pref_code"]),
                "population": float(node["population"]),
                "fixed_share": fixed_share,
                "relocated_share": relocated_share,
                "mobile_share": mobile_share,
                "digital_share": digital_share,
                "cross_boundary_share": node_cross_boundary_share,
                "mfg_slack": slack_value,
                "expected_access_minutes": weighted_minutes,
                "dominant_mode": dominant_mode,
                "dominant_facility_id": (
                    facilities.loc[dominant_facility_index, "facility_id"]
                    if dominant_facility_index is not None
                    else ""
                ),
                "dominant_facility_x": (
                    float(facilities.loc[dominant_facility_index, "x"])
                    if dominant_facility_index is not None
                    else np.nan
                ),
                "dominant_facility_y": (
                    float(facilities.loc[dominant_facility_index, "y"])
                    if dominant_facility_index is not None
                    else np.nan
                ),
                "x": float(node["x"]),
                "y": float(node["y"]),
            }
        )

    demand_result = pd.DataFrame(demand_records)
    facility_records = []
    for facility_index in candidates:
        capacity = (
            facility_capacity(facilities.loc[facility_index], config)
            * capacity_multiplier
        )
        direct_load = facility_direct_load[facility_index]
        relocated_load = facility_relocated_load[facility_index]
        facility_records.append(
            {
                "facility_id": facilities.loc[facility_index, "facility_id"],
                "facility_class": facilities.loc[facility_index, "facility_class"],
                "municipality_code": str(facilities.loc[facility_index, "municipality_code"]),
                "beds": float(facilities.loc[facility_index, "beds"]),
                "is_existing": bool(facilities.loc[facility_index, "is_existing"]),
                "open": unit_value(values[y[facility_index]]),
                "capacity": capacity,
                "direct_load": direct_load,
                "relocated_load": relocated_load,
                "total_load": direct_load + relocated_load,
                "capacity_utilization": (
                    (direct_load + relocated_load) / capacity if capacity > 0 else 0.0
                ),
                "cross_boundary_load": facility_cross_boundary_load[facility_index],
                "x": float(facilities.loc[facility_index, "x"]),
                "y": float(facilities.loc[facility_index, "y"]),
            }
        )
    facility_result = pd.DataFrame(facility_records)
    total_population = float(demand["population"].sum())
    direct_times_array = np.asarray(direct_times)
    direct_weights_array = np.asarray(direct_weights)
    mfg_violation_population = float(
        demand_result.loc[demand_result["mfg_slack"] > 1e-7, "population"].sum()
    )
    infrastructure_cost = float(
        sum(
            values[y[index]]
            * (
                costs["fixed_facility_open"]
                + facility_capacity(facilities.loc[index], config)
                * costs["fixed_facility_capacity_unit"]
            )
            for index in candidates
        )
    )
    service_cost = float(
        sum(values[index] * costs["mobile_hub_open"] for index in mobile_hub.values())
        + sum(
            float(demand.loc[index, "population"])
            * values[mobile[index]]
            * costs["mobile_service_per_resident"]
            * substitution_multiplier
            * mobile_cost_multiplier
            for index in demand.index
        )
        + sum(
            float(demand.loc[index, "population"])
            * values[digital[index]]
            * costs["digital_service_per_resident"]
            * substitution_multiplier
            * digital_cost_multiplier
            for index in demand.index
        )
    )
    relocation_cost = float(
        sum(
            float(demand.loc[demand_index, "population"])
            * distance_km
            * costs["relocation_per_person_km"]
            * relocation_friction
            * values[relocated[(demand_index, facility_index)]]
            for demand_index, facility_index, distance_km in edges
            if (demand_index, facility_index) in relocated
        )
    )
    access_cost = float(
        sum(
            float(demand.loc[demand_index, "population"])
            * distance_km
            * circuity
            / speed
            * 60
            * costs["access_per_person_minute"]
            * values[direct[(demand_index, facility_index)]]
            for demand_index, facility_index, distance_km in edges
        )
    )
    transition_cost = float(
        0.0
        if scenario == "A"
        else sum(
            costs["transition_per_changed_asset"]
            * (
                1 - values[y[index]]
                if bool(facilities.loc[index, "is_existing"])
                else values[y[index]]
            )
            for index in candidates
        )
    )
    mfg_penalty_cost = float(
        sum(
            float(demand.loc[index, "population"])
            * config["mfg"]["permitted_diagnostic_slack_penalty"]
            * values[slack[index]]
            for index in demand.index
        )
    )
    objective_components = (
        infrastructure_cost
        + service_cost
        + relocation_cost
        + access_cost
        + transition_cost
        + mfg_penalty_cost
    )
    summary: dict[str, float | int | str | bool] = {
        "model_version": MODEL_VERSION,
        "scenario": scenario,
        "relocation_friction": relocation_friction,
        "population_decline": population_decline,
        "population": total_population,
        "solver_success": solver_success,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_mip_gap": achieved_mip_gap,
        "runtime_seconds": perf_counter() - started,
        "objective_total": objective_components,
        "cost_per_resident": objective_components / total_population,
        "cost_infrastructure": infrastructure_cost,
        "cost_service": service_cost,
        "cost_relocation": relocation_cost,
        "cost_access": access_cost,
        "cost_transition": transition_cost,
        "cost_mfg_penalty": mfg_penalty_cost,
        "objective_solver_difference": float(
            objective_components - (result.fun + transition_constant)
        ),
        "facilities_open": float(facility_result["open"].sum()),
        "mobile_hubs_open": float(
            sum(values[index] for index in mobile_hub.values())
        ),
        "fixed_share": float(
            np.average(demand_result["fixed_share"], weights=demand_result["population"])
        ),
        "relocated_share": float(
            np.average(
                demand_result["relocated_share"], weights=demand_result["population"]
            )
        ),
        "mobile_share": float(
            np.average(demand_result["mobile_share"], weights=demand_result["population"])
        ),
        "digital_share": float(
            np.average(demand_result["digital_share"], weights=demand_result["population"])
        ),
        "cross_boundary_share": float(cross_boundary_population / total_population),
        "mean_access_minutes": float(
            np.average(direct_times_array, weights=direct_weights_array)
        ),
        "p95_access_minutes": float(
            demand_result.sort_values("expected_access_minutes")
            .assign(
                cumulative=lambda frame: frame["population"].cumsum()
                / frame["population"].sum()
            )
            .query("cumulative >= 0.95")["expected_access_minutes"]
            .iloc[0]
        ),
        "access_gini": weighted_gini(direct_times_array, direct_weights_array),
        "mfg_violation_population": mfg_violation_population,
        "mfg_violation_share": mfg_violation_population / total_population,
    }
    return ModelSolution(summary=summary, demand=demand_result, facilities=facility_result)
