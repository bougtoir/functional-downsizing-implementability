from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path

import pandas as pd
from docx import Document
from PIL import Image

from src.common import ROOT, load_config, reporting_parameters
from src.manuscript.build_submission import REFERENCES

RESULTS = ROOT / "outputs" / "results"
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"
BUILD = ROOT / "manuscript" / "build"
DIAGNOSTICS = ROOT / "outputs" / "diagnostics"
SUBMISSION = ROOT / "submission"
VALIDATION = ROOT / "outputs" / "validation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_ledger(path: Path) -> bool:
    with path.open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    return bool(records) and all(
        (ROOT / row["storage_path"]).exists()
        and (ROOT / row["storage_path"]).stat().st_size
        == int(row["file_size_bytes"])
        and sha256(ROOT / row["storage_path"]) == row["sha256"]
        for row in records
    )


def document_text(path: Path) -> str:
    document = Document(path)
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def document_core_paragraphs(path: Path) -> list[str]:
    document = Document(path)
    return [
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text
        and paragraph.text != "Figure legends"
        and not re.match(r"Figure \d+\.", paragraph.text)
    ]


def count_drawings(path: Path) -> int:
    document = Document(path)
    return sum(
        len(paragraph._element.xpath(".//w:drawing"))
        for paragraph in document.paragraphs
    )


def count_superscript_runs(path: Path) -> int:
    document = Document(path)
    return sum(
        run.font.superscript is True
        for paragraph in document.paragraphs
        for run in paragraph.runs
    )


def reference_entries(path: Path) -> list[str]:
    document = Document(path)
    entries: list[str] = []
    in_references = False
    for paragraph in document.paragraphs:
        if paragraph.text == "References":
            in_references = True
            continue
        if paragraph.text == "Figure legends":
            break
        if in_references and paragraph.text:
            entries.append(paragraph.text)
    return entries


def all_solver_gaps_valid(frame: pd.DataFrame, tolerance: float) -> bool:
    return bool(
        frame["solver_success"].astype(bool).all()
        and frame["solver_mip_gap"].notna().all()
        and frame["solver_mip_gap"].le(tolerance + 1e-9).all()
    )


def source_files_have_no_hardcoded_results(values: list[str]) -> bool:
    source_paths = [
        ROOT / "src" / "manuscript" / "build_submission.py",
        ROOT / "src" / "manuscript" / "package_submission.py",
        ROOT / "src" / "figures" / "build_figures.py",
        ROOT / "src" / "tables" / "build_tables.py",
    ]
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in source_paths)
    return not any(value in source_text for value in values)


def first_appearances_are_ordered(text: str, label: str, count: int) -> bool:
    positions = [text.find(f"{label} {number}") for number in range(1, count + 1)]
    return all(position >= 0 for position in positions) and positions == sorted(positions)


def verify_submission_manifest() -> bool:
    package = SUBMISSION / "seps_package"
    manifest_path = package / "submission_manifest.json"
    archive = SUBMISSION / "SEPS_submission_package_FINAL.zip"
    if not manifest_path.exists() or not archive.exists() or archive.stat().st_size == 0:
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    package_verified = bool(manifest["files"]) and all(
        (package / record["name"]).exists()
        and (package / record["name"]).stat().st_size == record["bytes"]
        and sha256(package / record["name"]) == record["sha256"]
        for record in manifest["files"]
    )
    with zipfile.ZipFile(archive) as zipped:
        expected_names = {
            record["name"] for record in manifest["files"]
        } | {"submission_manifest.json"}
        archive_verified = set(zipped.namelist()) == expected_names and all(
            hashlib.sha256(zipped.read(record["name"])).hexdigest()
            == record["sha256"]
            for record in manifest["files"]
        )
    return package_verified and archive_verified


def main() -> None:
    config = load_config()
    primary = pd.read_csv(RESULTS / "primary_results.csv")
    sensitivity = pd.read_csv(RESULTS / "sensitivity_results.csv")
    refinement = pd.read_csv(RESULTS / "transition_refinement.csv")
    capacity = pd.read_csv(RESULTS / "capacity_robustness.csv")
    rf_scaling = pd.read_csv(RESULTS / "rf_scaling_diagnostic.csv")
    sensitivity_stability = pd.read_csv(
        RESULTS / "sensitivity_rank_stability.csv"
    )
    density = pd.read_csv(RESULTS / "density_stratified_results.csv")
    facilities = pd.read_csv(RESULTS / "facility_diagnostics.csv")
    municipalities = pd.read_csv(RESULTS / "municipality_diagnostics.csv")
    robustness = json.loads(
        (RESULTS / "robustness_summary.json").read_text(encoding="utf-8")
    )
    network = json.loads(
        (VALIDATION / "network_travel_summary.json").read_text(encoding="utf-8")
    )
    preprocessing = json.loads(
        (ROOT / "data" / "processed" / "preprocessing_summary.json").read_text(
            encoding="utf-8"
        )
    )
    numeric_lineage = json.loads(
        (BUILD / "manuscript_numeric_lineage.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (DIAGNOSTICS / "validation_report.json").read_text(encoding="utf-8")
    )
    metrics = json.loads(
        (BUILD / "manuscript_metrics.json").read_text(encoding="utf-8")
    )
    expected_primary = (
        len(config["experiment"]["scenarios"])
        * len(config["experiment"]["relocation_friction"])
        * len(config["experiment"]["population_decline"])
    )
    objective_difference = primary["objective_solver_difference"].abs().max()
    sensitivity_objective_difference = (
        sensitivity["objective_solver_difference"].abs().max()
    )
    refinement_objective_difference = (
        refinement["objective_solver_difference"].abs().max()
    )
    manuscript_path = BUILD / "anonymized_manuscript.docx"
    inline_path = BUILD / "anonymized_manuscript_with_inline_figures.docx"
    author_details_path = BUILD / "manuscript_with_author_details.docx"
    manuscript_text = document_text(manuscript_path)
    references = reference_entries(manuscript_path)
    author_details_text = document_text(author_details_path)
    supplement_text = document_text(BUILD / "online_supplement.docx")
    cover_letter_text = document_text(BUILD / "cover_letter.docx")
    table_3 = pd.read_csv(TABLES / "T3_primary_results.csv").set_index(
        "scenario"
    )
    table_4 = pd.read_csv(TABLES / "T4_capacity_robustness.csv")
    expected_figure_names = {
        "F1_conceptual_framework.png",
        "F2_study_area.png",
        "F3_cost_rf_depopulation.png",
        "F4_strategy_substitution.png",
        "F5_functional_catchments.png",
        "F6_normalized_objective_reduction.png",
        "F7_sensitivity.png",
    }
    expected_table_names = {
        "T1_data_and_assumptions.csv",
        "T2_scenario_design.csv",
        "T3_primary_results.csv",
        "T4_capacity_robustness.csv",
        "T5_rf_transitions.csv",
        "T6_sensitivity_rankings.csv",
    }
    figure_files = sorted(FIGURES.glob("F[1-7]_*.png"))
    table_files = sorted(TABLES.glob("T[1-6]_*.csv"))
    reporting = reporting_parameters(config)
    reporting_rf = reporting["central_rf"]
    baseline_decline = reporting["baseline_decline"]
    central = primary[
        (primary["relocation_friction"] == reporting_rf)
        & (primary["population_decline"] == baseline_decline)
    ].set_index("scenario")
    central_a = central.loc["A"]
    central_d = central.loc["D"]
    objective_reduction = (
        100
        * (central_a["cost_per_resident"] - central_d["cost_per_resident"])
        / central_a["cost_per_resident"]
    )
    maximum_decline = max(config["experiment"]["population_decline"])
    maximum_decline_rows = primary[
        (primary["relocation_friction"] == reporting_rf)
        & (primary["population_decline"] == maximum_decline)
    ].set_index("scenario")
    maximum_decline_reduction = (
        100
        * (
            maximum_decline_rows.loc["A", "cost_per_resident"]
            - maximum_decline_rows.loc["D", "cost_per_resident"]
        )
        / maximum_decline_rows.loc["A", "cost_per_resident"]
    )
    rf_endpoints = primary[
        (primary["scenario"] == "D")
        & (primary["population_decline"] == baseline_decline)
        & primary["relocation_friction"].isin(
            [
                min(config["experiment"]["relocation_friction"]),
                max(config["experiment"]["relocation_friction"]),
            ]
        )
    ].set_index("relocation_friction")
    network_primary = network["all_eligible_fixed_pairs"]
    network_stability = network["nearest_open_facility_stability"]
    capacity_nonmonotonicity = robustness["capacity_nonmonotonicity"]
    package_names = {
        path.name
        for path in (SUBMISSION / "seps_package").iterdir()
        if path.is_file()
    }
    capacity_multiplier_text = ", ".join(
        f"{value:g}" for value in capacity["capacity_multiplier"]
    )
    rf_multiplier_text = ", ".join(
        f"{value:g}"
        for value in sorted(
            rf_scaling["relocation_cost_coefficient_multiplier"].unique()
        )
    )
    figure_dimensions = {}
    for path in figure_files:
        with Image.open(path) as image:
            figure_dimensions[path.name] = list(image.size)
    checks = {
        "primary_row_count": len(primary) == expected_primary,
        "primary_all_solvers_success": primary["solver_success"].astype(bool).all(),
        "primary_mip_gaps_within_tolerance": all_solver_gaps_valid(
            primary,
            config["solver"]["mip_relative_gap"],
        ),
        "primary_objectives_reconcile": objective_difference < 1e-5,
        "primary_no_material_mfg_violation": (
            primary["mfg_violation_share"].max() < 1e-6
        ),
        "sensitivity_draw_count": len(sensitivity)
        == config["experiment"]["sensitivity_draws"],
        "sensitivity_complete": not sensitivity.isna().any().any(),
        "sensitivity_all_solvers_success": sensitivity["solver_success"]
        .astype(bool)
        .all(),
        "sensitivity_mip_gaps_within_tolerance": all_solver_gaps_valid(
            sensitivity,
            config["solver"]["mip_relative_gap"],
        ),
        "sensitivity_objectives_reconcile": sensitivity_objective_difference < 1e-5,
        "transition_refinement_nonempty": len(refinement) > 0,
        "transition_refinement_all_solvers_success": refinement["solver_success"]
        .astype(bool)
        .all(),
        "refinement_mip_gaps_within_tolerance": all_solver_gaps_valid(
            refinement,
            config["solver"]["transition_mip_relative_gap"],
        ),
        "transition_refinement_objectives_reconcile": (
            refinement_objective_difference < 1e-5
        ),
        "capacity_grid_is_predeclared": capacity["capacity_multiplier"].tolist()
        == [0.6, 0.8, 1.0, 1.2, 1.4],
        "capacity_all_solvers_within_tolerance": all_solver_gaps_valid(
            capacity,
            config["solver"]["mip_relative_gap"],
        ),
        "capacity_all_objectives_reconcile": (
            capacity["objective_solver_difference"].abs().max() < 1e-5
        ),
        "capacity_all_mfg_violations_zero": (
            capacity["mfg_violation_share"].abs().max() < 1e-9
        ),
        "capacity_mobile_share_zero": capacity["mobile_share"].abs().max() < 1e-9,
        "capacity_nonmonotonicity_verified": (
            capacity_nonmonotonicity["classification"]
            == "genuine_discrete_regime_change"
            and abs(
                capacity_nonmonotonicity["canonical_relocated_share"]
                - capacity.loc[
                    capacity["capacity_multiplier"] == 1.2,
                    "relocated_share",
                ].iloc[0]
            )
            < 1e-9
            and capacity_nonmonotonicity["verification_achieved_mip_gap"]
            <= capacity_nonmonotonicity["verification_mip_relative_gap"] + 1e-9
            and capacity_nonmonotonicity[
                "leading_relocation_facility_class"
            ]
            == "planned_hub"
            and capacity_nonmonotonicity[
                "leading_relocation_facility_utilization"
            ]
            > 0.99
        ),
        "rf_scaling_coefficients_are_predeclared": set(
            rf_scaling["relocation_cost_coefficient_multiplier"]
        )
        == {0.5, 1.0, 2.0},
        "rf_scaling_all_solvers_within_tolerance": all_solver_gaps_valid(
            rf_scaling,
            config["solver"]["mip_relative_gap"],
        ),
        "rf_scaling_effective_rf_identity": (
            (
                rf_scaling["nominal_relocation_friction"]
                * rf_scaling["relocation_cost_coefficient_multiplier"]
                - rf_scaling["effective_relocation_friction"]
            )
            .abs()
            .max()
            < 1e-12
        ),
        "rf_scaling_equal_effective_rf_spread_zero": max(
            robustness["rf_equivalence_maximum_spread"].values()
        )
        < 1e-12,
        "sensitivity_rank_stability_uses_four_holdouts": set(
            sensitivity_stability["sample"]
        )
        == {
            "full",
            "holdout_fold_0",
            "holdout_fold_1",
            "holdout_fold_2",
            "holdout_fold_3",
        },
        "sensitivity_leading_drivers_stable": all(
            values["top_driver_holdout_consistency"] == 1.0
            for values in robustness["sensitivity_rank_stability"].values()
        ),
        "network_validation_pair_definition_predeclared": (
            network["design"]["demand_clusters"] == preprocessing["demand_clusters"]
            and network["design"]["threshold_minutes"]
            == config["mfg"]["healthcare_max_minutes"]
        ),
        "network_validation_reachable_pairs_reported": (
            network_primary["pairs"] == 140
            and network["route_counts"]["unreachable_dominant_pairs"] == 3
        ),
        "network_validation_no_primary_ferry_paths": (
            network["route_counts"]["dominant_pairs_using_ferry"] == 0
        ),
        "network_validation_supports_proxy_ranking": (
            network_primary["spearman_rho"] > 0.9
            and network_primary["population_weighted_threshold_agreement"] > 0.95
        ),
        "network_nearest_facility_stability_reported": (
            network_stability["population_weighted_same_facility_share"] > 0.9
            and network_stability["population_weighted_same_municipality_share"] > 0.95
        ),
        "predeclared_validation_passed": all(validation["checks"].values()),
        "public_data_ledger_verified": verify_ledger(
            ROOT / "data" / "raw" / "acquisition_ledger.csv"
        ),
        "literature_ledger_verified": verify_ledger(
            ROOT / "data" / "raw" / "literature" / "acquisition_ledger.csv"
        ),
        "journal_policy_ledger_verified": verify_ledger(
            ROOT / "data" / "raw" / "journal" / "acquisition_ledger.csv"
        ),
        "author_metadata_ledger_verified": verify_ledger(
            ROOT / "data" / "raw" / "author" / "acquisition_ledger.csv"
        ),
        "seven_figures_present": {path.name for path in figure_files}
        == expected_figure_names,
        "six_tables_present": {path.name for path in table_files}
        == expected_table_names,
        "submission_manuscript_has_no_embedded_figures": count_drawings(
            manuscript_path
        )
        == 0,
        "review_manuscript_has_seven_embedded_figures": count_drawings(inline_path)
        == 7,
        "all_figure_legends_present": all(
            f"Figure {number}." in manuscript_text for number in range(1, 8)
        ),
        "all_table_citations_present": all(
            f"Table {number}" in manuscript_text for number in range(1, 7)
        ),
        "abstract_within_working_limit": metrics["abstract_words"]
        <= metrics["journal_abstract_limit"],
        "manuscript_within_working_limit": metrics[
            "main_words_including_front_matter"
        ]
        <= metrics["journal_main_limit"],
        "keywords_within_working_limit": metrics["keyword_count"]
        <= config["manuscript"]["keyword_limit"],
        "author_date_citations_use_plain_text": count_superscript_runs(
            manuscript_path
        )
        == 0,
        "no_unicode_superscript_digits": not any(
            character in manuscript_text for character in "⁰¹²³⁴⁵⁶⁷⁸⁹"
        ),
        "anonymized_manuscript_has_no_author_identifiers": not any(
            identifier in manuscript_text.lower()
            for identifier in ("tatsuki onishi", "bougtoir", "orcid", "shiga university")
        ),
        "full_manuscript_contains_author_details": all(
            identifier in author_details_text
            for identifier in (
                "Journal: Socio-Economic Planning Sciences",
                "Original research article",
                "Tatsuki Onishi",
                "bougtoir@gmail.com",
                "https://orcid.org/0000-0001-7261-9062",
                "Acknowledgements",
            )
        ),
        "figures_first_appear_in_number_order": first_appearances_are_ordered(
            manuscript_text,
            "Figure",
            7,
        ),
        "tables_first_appear_in_number_order": first_appearances_are_ordered(
            manuscript_text,
            "Table",
            7,
        ),
        "reference_list_has_39_entries": len(references) == 39,
        "reference_list_matches_registry_in_citation_order": [
            re.sub(r"^\[\d+\]\s*", "", entry) for entry in references
        ]
        == [
            REFERENCES[int(ref_id)]
            for ref_id, _ in sorted(
                json.loads(
                    (BUILD / "manuscript_numeric_lineage.json").read_text(
                        encoding="utf-8"
                    )
                )["citation_order"].items(),
                key=lambda item: item[1],
            )
        ],
        "reference_numbers_are_sequential": [
            int(m.group(1))
            for entry in references
            if (m := re.match(r"\[(\d+)\]", entry))
        ]
        == list(range(1, len(references) + 1)),
        "all_registry_references_cited": len(references) == len(REFERENCES),
        "references_include_doi_url": all(
            "https://doi.org/10." in reference for reference in references
        ),
        "manuscript_numeric_lineage_present": (
            BUILD / "manuscript_numeric_lineage.json"
        ).exists(),
        "references_include_39_dois": len(
            re.findall(r"https://doi\.org/10\.", manuscript_text, flags=re.IGNORECASE)
        )
        == 39,
        "citation_ecology_maps_all_dois": {
            doi.rstrip(".)")
            for doi in re.findall(
                r"(10\.\d{4,9}/[^\s|]+)",
                (ROOT / "SEPS_CITATION_ECOLOGY.md").read_text(encoding="utf-8"),
            )
        }
        >= {
            doi.rstrip(".")
            for doi in re.findall(
                r"https://doi\.org/(10\.[^\s]+)",
                manuscript_text,
            )
        },
        "central_objective_reduction_reconciles": abs(objective_reduction - 70.0)
        < 0.05,
        "maximum_decline_reduction_reconciles": abs(
            maximum_decline_reduction - 75.6
        )
        < 0.05,
        "central_sites_reconcile": int(round(central_d["facilities_open"])) == 46,
        "central_mode_shares_reconcile": (
            abs(central_d["cross_boundary_share"] - 0.18137) < 5e-4
            and abs(central_d["digital_share"] - 0.03998) < 5e-4
            and abs(central_d["mobile_share"]) < 1e-9
            and abs(central_d["relocated_share"] - 0.00137) < 5e-4
        ),
        "rf_endpoint_relocation_reconciles": (
            rf_endpoints.loc[
                min(config["experiment"]["relocation_friction"]),
                "relocated_share",
            ]
            > 0.98
            and rf_endpoints.loc[
                max(config["experiment"]["relocation_friction"]),
                "relocated_share",
            ]
            < 0.001
        ),
        "study_area_counts_reconcile": (
            preprocessing["municipalities"] == 68
            and preprocessing["populated_cells"] == 23699
            and preprocessing["demand_clusters"] == 144
            and preprocessing["all_medical_institutions"] == 4591
            and preprocessing["candidate_fixed_assets_and_hubs"] == 496
            and preprocessing["planned_hubs"] == 68
            and preprocessing["population_conservation_relative_error"] == 0
        ),
        "density_diagnostics_reconcile": (
            density["density_tercile"].tolist() == ["low", "middle", "high"]
            and density["demand_clusters"].sum() == 144
            and density["mfg_violation_share"].max() == 0
        ),
        "facility_diagnostics_reconcile": (
            facilities.set_index("facility_class").loc["clinic", "open_sites"] == 2
            and facilities.set_index("facility_class").loc["hospital", "open_sites"]
            == 44
            and facilities.set_index("facility_class").loc[
                "planned_hub",
                "open_sites",
            ]
            == 0
        ),
        "municipality_cross_boundary_diagnostics_reconcile": (
            (municipalities["cross_boundary_share"] > 0.5).sum() == 38
            and municipalities["cross_boundary_share"].max() > 0.999
        ),
        "primary_table_reconciles_with_canonical_results": (
            abs(
                table_3.loc["D", "normalized_objective_per_resident"]
                - central_d["cost_per_resident"]
            )
            < 1e-12
            and abs(
                table_3.loc["D", "cross_boundary_share"]
                - central_d["cross_boundary_share"]
            )
            < 1e-12
            and abs(
                table_3.loc["D", "digital_share"]
                - central_d["digital_share"]
            )
            < 1e-12
            and abs(
                table_3.loc["D", "relocated_share"]
                - central_d["relocated_share"]
            )
            < 1e-12
        ),
        "capacity_table_reconciles_with_canonical_results": (
            table_4["capacity_multiplier"].tolist()
            == capacity["capacity_multiplier"].tolist()
            and (
                table_4[
                    [
                        "normalized_objective_per_resident",
                        "facilities_open",
                        "relocated_share",
                        "mobile_share",
                        "digital_share",
                        "cross_boundary_share",
                        "mfg_violation_share",
                        "maximum_capacity_utilization_open",
                        "solver_mip_gap",
                    ]
                ]
                - capacity[
                    [
                        "cost_per_resident",
                        "facilities_open",
                        "relocated_share",
                        "mobile_share",
                        "digital_share",
                        "cross_boundary_share",
                        "mfg_violation_share",
                        "maximum_capacity_utilization_open",
                        "solver_mip_gap",
                    ]
                ].rename(
                    columns={
                        "cost_per_resident": (
                            "normalized_objective_per_resident"
                        )
                    }
                )
            )
            .abs()
            .to_numpy()
            .max()
            < 1e-12
        ),
        "cover_letter_headline_values_reconcile": all(
            phrase in cover_letter_text
            for phrase in (
                "70.0% lower normalized model objective",
                "46 fixed sites",
                "18.1% cross-boundary provision",
                "4.0% digital provision",
                "0% mobile provision",
                "0.14% relocation",
                "not a fiscal estimate",
            )
        ),
        "supplement_network_and_capacity_values_reconcile": all(
            phrase in supplement_text
            for phrase in (
                "Spearman rho was 0.962",
                "99.5%",
                "capacity multipliers 0.6, 0.8, 1.0, 1.2, and 1.4",
                "zero MFG violation",
                "42 to 49",
                "mobile provision remained zero",
            )
        ),
        "manuscript_reports_normalized_objective_not_savings": (
            "70.0% lower normalized model objective" in manuscript_text
            and "rather than an estimate of fiscal savings" in manuscript_text
            and not re.search(
                r"\b(?:fiscal|budget|public-expenditure) savings?\b",
                manuscript_text.replace("estimate of fiscal savings", ""),
                re.IGNORECASE,
            )
        ),
        "manuscript_reports_zero_mobile_mechanism": (
            "mobile provision 0.0%" in manuscript_text
            and "does not mechanically select a technology" in manuscript_text
        ),
        "manuscript_reports_capacity_conditionality": (
            "do not demonstrate that exactly 46 current real facilities"
            in manuscript_text
            and "reflects a discrete site-opening regime" in manuscript_text
        ),
        "manuscript_reports_fairness_limits": (
            "not a complete distributive-justice theory" in manuscript_text
            and "modeled resident travel burden, not total service effort"
            in manuscript_text
            and "alternative fairness specifications can change the optimum"
            in manuscript_text
        ),
        "manuscript_reports_healthcare_only_scope": (
            "the empirical application models healthcare only" in manuscript_text
        ),
        "inline_and_separate_manuscripts_match_text": document_core_paragraphs(
            inline_path
        )
        == document_core_paragraphs(manuscript_path),
        "numeric_lineage_sources_complete": set(
            numeric_lineage["sources"]
        )
        == {
            "central_scenarios",
            "rf_endpoints",
            "maximum_decline",
            "sensitivity_driver",
            "density_diagnostics",
            "facility_diagnostics",
            "municipality_diagnostics",
            "robustness",
            "network_validation",
        },
        "analytical_results_not_hardcoded_in_python": source_files_have_no_hardcoded_results(
            ["70.0%", "75.6%", "99.5%", "0.962"]
        ),
        "no_stale_savings_figure_name": not any(
            "savings" in name or "scenario_d_savings" in name
            for name in package_names | expected_figure_names
        ),
        "submission_package_manifest_verified": verify_submission_manifest(),
        "separate_data_statement_has_public_repository_url": (
            "https://github.com/bougtoir/regional-functional-downsizing"
            in (
                BUILD / "data_availability_statement.txt"
            ).read_text(encoding="utf-8")
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    report = {
        "checks": checks,
        "diagnostics": {
            "expected_primary_rows": expected_primary,
            "observed_primary_rows": len(primary),
            "sensitivity_rows": len(sensitivity),
            "transition_refinement_rows": len(refinement),
            "maximum_objective_reconciliation_error": objective_difference,
            "maximum_sensitivity_objective_reconciliation_error": (
                sensitivity_objective_difference
            ),
            "maximum_refinement_objective_reconciliation_error": (
                refinement_objective_difference
            ),
            "maximum_primary_mip_gap": primary["solver_mip_gap"].max(),
            "maximum_sensitivity_mip_gap": sensitivity["solver_mip_gap"].max(),
            "maximum_refinement_mip_gap": refinement["solver_mip_gap"].max(),
            "maximum_capacity_mip_gap": capacity["solver_mip_gap"].max(),
            "capacity_nonmonotonicity": capacity_nonmonotonicity,
            "maximum_rf_scaling_mip_gap": rf_scaling["solver_mip_gap"].max(),
            "maximum_mfg_violation_share": primary["mfg_violation_share"].max(),
            "central_objective_reduction_percent": objective_reduction,
            "maximum_decline_objective_reduction_percent": (
                maximum_decline_reduction
            ),
            "central_scenario_d_sites": central_d["facilities_open"],
            "central_scenario_d_cross_boundary_share": central_d[
                "cross_boundary_share"
            ],
            "central_scenario_d_digital_share": central_d["digital_share"],
            "central_scenario_d_mobile_share": central_d["mobile_share"],
            "central_scenario_d_relocated_share": central_d["relocated_share"],
            "network_spearman_rho": network_primary["spearman_rho"],
            "network_population_weighted_threshold_agreement": network_primary[
                "population_weighted_threshold_agreement"
            ],
            "abstract_words": metrics["abstract_words"],
            "main_words_including_front_matter": metrics[
                "main_words_including_front_matter"
            ],
            "figure_dimensions_pixels": figure_dimensions,
        },
    }
    DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
    (DIAGNOSTICS / "final_audit.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    numeric_audit = f"""# Numeric audit

Generated programmatically from final machine-readable outputs.

- Primary scenario rows: {len(primary)} of {expected_primary} expected.
- Sensitivity draws: {len(sensitivity)}.
- Transition-refinement rows: {len(refinement)}.
- Maximum absolute solver/objective-component difference: {objective_difference:.3e}.
- Maximum sensitivity objective difference: {sensitivity_objective_difference:.3e}.
- Maximum refinement objective difference: {refinement_objective_difference:.3e}.
- Maximum primary MIP gap: {primary["solver_mip_gap"].max():.3e}.
- Maximum sensitivity MIP gap: {sensitivity["solver_mip_gap"].max():.3e}.
- Maximum transition-refinement MIP gap: {refinement["solver_mip_gap"].max():.3e}.
- Maximum capacity-robustness MIP gap: {capacity["solver_mip_gap"].max():.3e}.
- Maximum RF-scaling MIP gap: {rf_scaling["solver_mip_gap"].max():.3e}.
- Predeclared primary/sensitivity MIP-gap tolerance: {config["solver"]["mip_relative_gap"]:.3e}.
- Predeclared transition-refinement MIP-gap tolerance:
  {config["solver"]["transition_mip_relative_gap"]:.3e}.
- Maximum population-weighted MFG violation share: {primary["mfg_violation_share"].max():.3e}.
- Study-area inputs: {preprocessing["municipalities"]} municipalities,
  {preprocessing["populated_cells"]} populated cells,
  {preprocessing["demand_clusters"]} demand clusters,
  {preprocessing["all_medical_institutions"]} medical institutions,
  and {preprocessing["population_from_cells"]:,.3f} represented residents.
- Central Scenario A/D normalized objectives per resident:
  {central_a["cost_per_resident"]:.8f} / {central_d["cost_per_resident"]:.8f};
  Scenario D is {objective_reduction:.1f}% lower.
- Central Scenario D: {central_d["facilities_open"]:.0f} open sites,
  {100 * central_d["cross_boundary_share"]:.1f}% cross-boundary,
  {100 * central_d["digital_share"]:.1f}% digital,
  {100 * central_d["mobile_share"]:.1f}% mobile, and
  {100 * central_d["relocated_share"]:.2f}% relocation.
- Under {100 * maximum_decline:.0f}% synthetic decline, Scenario D is
  {maximum_decline_reduction:.1f}% below Scenario A at central RF.
- Network validation: {network_primary["pairs"]} reachable non-ferry fixed-mode pairs,
  Spearman rho {network_primary["spearman_rho"]:.4f}, and
  {100 * network_primary["population_weighted_threshold_agreement"]:.3f}%
  population-weighted 45-minute agreement.
- Capacity multipliers: {capacity_multiplier_text}.
- RF coefficient multipliers: {rf_multiplier_text};
  maximum equal-effective-RF spread is
  {max(robustness["rf_equivalence_maximum_spread"].values()):.3e}.
- All four sensitivity outcomes retained the same leading driver in every
  deterministic holdout.
- All manuscript analytical values are loaded by `src/manuscript/build_submission.py`
  from generated CSV and JSON files; no result value is manually entered into a DOCX.
- Costs are normalized planning units, not currency.
- Relocation friction is dimensionless.
- Decline levels are synthetic stress tests, not official projections.
"""
    (ROOT / "docs").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "NUMERIC_AUDIT.md").write_text(
        numeric_audit,
        encoding="utf-8",
    )
    passed = sum(checks.values())
    failed = [name for name, value in checks.items() if not value]
    final_qc = f"""# Final quality-control report

Programmatic audit: **{passed}/{len(checks)} checks passed**.

## Checks

"""
    final_qc += "\n".join(
        f"- [{'x' if value else ' '}] {name}" for name, value in checks.items()
    )
    final_qc += (
        "\n\n## Remaining author-controlled upload gates\n\n"
        "- Confirm live portal file designations and paste-in fields.\n"
        "- Inspect the generated reviewer PDF to confirm anonymization.\n"
    )
    if failed:
        final_qc += f"\n## Failed checks\n\n{', '.join(failed)}\n"
    (ROOT / "docs" / "FINAL_QC_REPORT.md").write_text(
        final_qc,
        encoding="utf-8",
    )
    if failed:
        raise RuntimeError(f"Final audit failed: {failed}")
    print(json.dumps(report["diagnostics"], indent=2))


if __name__ == "__main__":
    main()
