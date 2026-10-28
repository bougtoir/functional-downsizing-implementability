from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from scipy.stats import spearmanr

from src.common import ROOT, detail_stem, load_config, reporting_parameters

RESULTS = ROOT / "outputs" / "results"
FIGURES = ROOT / "outputs" / "figures"
GEOGRAPHY = ROOT / "data" / "processed" / "geography.gpkg"

COLORS = {
    "A": "#4C78A8",
    "B": "#F58518",
    "C": "#54A24B",
    "D": "#E45756",
    "fixed": "#4C78A8",
    "relocation": "#F58518",
    "mobile": "#54A24B",
    "digital": "#B279A2",
}


def save_figure(figure: plt.Figure, stem: str) -> tuple[Path, Path]:
    png = FIGURES / f"{stem}.png"
    pdf = FIGURES / f"{stem}.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png, pdf


def figure_1() -> tuple[Path, Path]:
    figure, axis = plt.subplots(figsize=(11, 6.2))
    axis.set_xlim(0, 11)
    axis.set_ylim(0, 6.2)
    axis.axis("off")
    boxes = [
        (0.4, 3.9, 2.2, 1.2, "Demographic\ncontraction", "#DDEBF7"),
        (3.2, 3.9, 2.2, 1.2, "Inherited assets and\nadministrative borders", "#FCE4D6"),
        (6.0, 3.9, 2.2, 1.2, "Relocation friction\n(RF sweep)", "#FFF2CC"),
        (8.8, 3.9, 1.8, 1.2, "Minimum functional\nguarantee", "#E2F0D9"),
        (2.0, 1.2, 2.4, 1.2, "Move residents\n(voluntary relocation)", "#F4B183"),
        (4.8, 1.2, 2.4, 1.2, "Reconfigure provision\n(fixed / digital / mobile)", "#A9D18E"),
        (7.6, 1.2, 2.4, 1.2, "Reconfigure catchments\n(cross-boundary)", "#9DC3E6"),
    ]
    for x, y, width, height, label, color in boxes:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.08",
                facecolor=color,
                edgecolor="#404040",
                linewidth=1.2,
            )
        )
        axis.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=11,
        )
    for start, end in [
        ((2.6, 4.5), (3.2, 4.5)),
        ((5.4, 4.5), (6.0, 4.5)),
        ((8.2, 4.5), (8.8, 4.5)),
        ((5.0, 3.9), (3.2, 2.4)),
        ((6.8, 3.9), (6.0, 2.4)),
        ((9.4, 3.9), (8.8, 2.4)),
    ]:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=1.3,
                color="#606060",
            )
        )
    axis.text(
        5.5,
        0.45,
        "Compare normalized model objective, residual travel, relocation, substitution, "
        "equity, and cross-boundary provision",
        ha="center",
        fontsize=11,
        weight="bold",
    )
    axis.set_title(
        "Analytical framework: preserve minimum function while changing delivery geography",
        fontsize=15,
        weight="bold",
        pad=12,
    )
    return save_figure(figure, "F1_conceptual_framework")


def figure_2() -> tuple[Path, Path]:
    municipalities = gpd.read_file(GEOGRAPHY, layer="municipalities")
    demand = gpd.read_file(GEOGRAPHY, layer="demand")
    facilities = gpd.read_file(GEOGRAPHY, layer="facilities")
    candidates = facilities[facilities["is_candidate"]]
    figure = plt.figure(figsize=(12, 6.8))
    main_axis = figure.add_axes((0.03, 0.06, 0.72, 0.84))
    island_axis = figure.add_axes((0.75, 0.48, 0.23, 0.36))
    for axis in (main_axis, island_axis):
        municipalities.plot(
            ax=axis,
            facecolor="#F7F7F7",
            edgecolor="#8C8C8C",
            linewidth=0.45,
        )
        candidates[candidates["facility_class"] == "clinic"].plot(
            ax=axis,
            color="#4C78A8",
            marker="o",
            markersize=7,
            alpha=0.65,
        )
        candidates[candidates["facility_class"] == "hospital"].plot(
            ax=axis,
            color="#E45756",
            marker="^",
            markersize=14,
            alpha=0.8,
        )
        demand.plot(
            ax=axis,
            color="#111111",
            markersize=np.sqrt(demand["population"]) * 0.55,
            alpha=0.45,
        )
        axis.set_axis_off()
    bounds = municipalities.total_bounds
    main_axis.set_xlim(bounds[0], bounds[2])
    main_axis.set_ylim(bounds[1], 0)
    island_axis.set_xlim(-235_000, -70_000)
    island_axis.set_ylim(5_000, 148_000)
    island_axis.set_title("Oki Islands", fontsize=10)
    figure.suptitle(
        "Study area, demand clusters, and candidate medical facilities",
        fontsize=15,
        weight="bold",
    )
    main_axis.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color="#111111", label="Demand cluster"),
            Line2D([], [], marker="o", linestyle="", color="#4C78A8", label="Clinic"),
            Line2D([], [], marker="^", linestyle="", color="#E45756", label="Hospital"),
        ],
        loc="upper left",
        frameon=True,
    )
    main_axis.text(
        0.01,
        0.01,
        "Bubble area scales with 2020 WorldPop cluster population.",
        transform=main_axis.transAxes,
        fontsize=9,
    )
    return save_figure(figure, "F2_study_area")


def figure_3(
    primary: pd.DataFrame,
    transition: pd.DataFrame,
) -> tuple[Path, Path]:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    positive_rf = sorted(
        value for value in primary["relocation_friction"].unique() if value > 0
    )
    linear_threshold = positive_rf[0]
    for axis, decline in zip(
        axes.flat,
        sorted(primary["population_decline"].unique()),
        strict=True,
    ):
        subset = primary[primary["population_decline"] == decline]
        for scenario in ("A", "B", "C", "D"):
            frame = subset[subset["scenario"] == scenario].sort_values(
                "relocation_friction"
            )
            axis.plot(
                frame["relocation_friction"],
                frame["cost_per_resident"],
                marker="o",
                linewidth=2,
                color=COLORS[scenario],
                label=f"Scenario {scenario}",
            )
        refined = transition[
            (transition["population_decline"] == decline)
            & (transition["scenario"] == "D")
        ].sort_values("relocation_friction")
        axis.scatter(
            refined["relocation_friction"],
            refined["cost_per_resident"],
            marker="x",
            s=36,
            linewidth=1.2,
            color=COLORS["D"],
            label="Scenario D refinement",
            zorder=4,
        )
        axis.axvspan(1, 2, color="#F2CF5B", alpha=0.14, linewidth=0)
        axis.set_xscale("symlog", linthresh=linear_threshold)
        axis.set_title(f"Population decline: {decline:.0%}")
        axis.grid(alpha=0.25)
        axis.set_ylabel("Normalized objective per resident")
    axes[-1, 0].set_xlabel("Relocation friction (RF)")
    axes[-1, 1].set_xlabel("Relocation friction (RF)")
    axes[0, 0].legend(ncol=2, fontsize=8)
    axes[0, 1].text(
        0.98,
        0.04,
        "Shading: effective RF 1–2\n×: Scenario D refinement",
        transform=axes[0, 1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    figure.suptitle(
        "Normalized model objective across relocation friction and population decline",
        fontsize=15,
        weight="bold",
    )
    figure.tight_layout()
    return save_figure(figure, "F3_cost_rf_depopulation")


def figure_4(primary: pd.DataFrame) -> tuple[Path, Path]:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    reporting = reporting_parameters(load_config())
    baseline = primary[
        (primary["population_decline"] == reporting["baseline_decline"])
        & (primary["scenario"].isin(["B", "C", "D"]))
    ]
    for axis, scenario in zip(axes, ("B", "C", "D"), strict=True):
        frame = baseline[baseline["scenario"] == scenario].sort_values(
            "relocation_friction"
        )
        for column, label, mode in (
            ("fixed_share", "Fixed-site provision", "fixed"),
            ("relocated_share", "Relocation", "relocation"),
            ("mobile_share", "Mobile", "mobile"),
            ("digital_share", "Digital", "digital"),
        ):
            axis.plot(
                frame["relocation_friction"],
                frame[column],
                marker="o",
                markersize=4,
                linewidth=2,
                color=COLORS[mode],
                label=label,
                zorder=4 if mode == "mobile" else 2,
            )
        positive_rf = sorted(
            value for value in frame["relocation_friction"].unique() if value > 0
        )
        axis.set_xscale("symlog", linthresh=positive_rf[0])
        axis.set_title(f"Scenario {scenario}")
        axis.set_xlabel("Relocation friction (RF)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Population-weighted service share")
    axes[-1].legend(loc="upper right", fontsize=9)
    figure.suptitle(
        "Provision shares across relocation-friction regimes",
        fontsize=15,
        weight="bold",
    )
    figure.text(
        0.5,
        0.015,
        "The mobile share remains zero throughout all three scenarios.",
        ha="center",
        fontsize=9,
        color=COLORS["mobile"],
        weight="bold",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    return save_figure(figure, "F4_strategy_substitution")


def figure_5() -> tuple[Path, Path]:
    reporting = reporting_parameters(load_config())
    stem = detail_stem(
        "D",
        reporting["central_rf"],
        reporting["baseline_decline"],
    )
    municipalities = gpd.read_file(GEOGRAPHY, layer="municipalities")
    demand = pd.read_csv(RESULTS / "details" / f"{stem}_demand.csv")
    facilities = pd.read_csv(RESULTS / "details" / f"{stem}_facilities.csv")
    facility_municipalities = facilities.set_index("facility_id")["municipality_code"]
    demand["dominant_facility_municipality"] = demand["dominant_facility_id"].map(
        facility_municipalities
    )
    figure = plt.figure(figsize=(12, 6.8))
    main_axis = figure.add_axes((0.03, 0.06, 0.72, 0.84))
    island_axis = figure.add_axes((0.75, 0.48, 0.23, 0.36))
    open_facilities = facilities[facilities["open"] > 0.5]
    for axis in (main_axis, island_axis):
        municipalities.plot(
            ax=axis,
            facecolor="#FAFAFA",
            edgecolor="#999999",
            linewidth=0.45,
        )
        for row in demand.itertuples():
            if row.dominant_mode != "fixed" or pd.isna(row.dominant_facility_x):
                continue
            cross_boundary = (
                row.municipality_code != row.dominant_facility_municipality
            )
            axis.plot(
                [row.x, row.dominant_facility_x],
                [row.y, row.dominant_facility_y],
                color="#E45756" if cross_boundary else "#BDBDBD",
                linewidth=0.35 + 1.8 * min(1.0, row.population / 40_000),
                alpha=0.6,
                zorder=1,
            )
        axis.scatter(
            demand["x"],
            demand["y"],
            s=np.sqrt(demand["population"]) * 0.45,
            c=demand["cross_boundary_share"],
            cmap="Reds",
            vmin=0,
            vmax=1,
            edgecolors="none",
            alpha=0.75,
            zorder=2,
        )
        axis.scatter(
            open_facilities["x"],
            open_facilities["y"],
            marker="^",
            s=36,
            color="#2166AC",
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        axis.set_axis_off()
    bounds = municipalities.total_bounds
    main_axis.set_xlim(bounds[0], bounds[2])
    main_axis.set_ylim(bounds[1], 0)
    island_axis.set_xlim(-235_000, -70_000)
    island_axis.set_ylim(5_000, 148_000)
    island_axis.set_title("Oki Islands", fontsize=10)
    figure.suptitle(
        "Functional catchments under Scenario D "
        f"(RF = {reporting['central_rf']:g}, "
        f"decline = {reporting['baseline_decline']:.0%})",
        fontsize=15,
        weight="bold",
    )
    main_axis.legend(
        handles=[
            Line2D([], [], color="#E45756", linewidth=1.5, label="Cross-boundary link"),
            Line2D([], [], color="#BDBDBD", linewidth=1.2, label="Within-municipality link"),
            Line2D([], [], marker="^", linestyle="", color="#2166AC", label="Open fixed site"),
        ],
        loc="upper left",
    )
    main_axis.text(
        0.01,
        0.01,
        "Links show each cluster's dominant fixed assignment, not all service flows.",
        transform=main_axis.transAxes,
        fontsize=9,
    )
    return save_figure(figure, "F5_functional_catchments")


def figure_6(primary: pd.DataFrame) -> tuple[Path, Path]:
    status_quo = primary[primary["scenario"] == "A"][
        ["population_decline", "relocation_friction", "cost_per_resident"]
    ].rename(columns={"cost_per_resident": "status_quo_cost"})
    scenario_d = primary[primary["scenario"] == "D"].merge(
        status_quo,
        on=["population_decline", "relocation_friction"],
        validate="one_to_one",
    )
    scenario_d["objective_reduction_percent"] = (
        100
        * (scenario_d["status_quo_cost"] - scenario_d["cost_per_resident"])
        / scenario_d["status_quo_cost"]
    )
    matrix = scenario_d.pivot(
        index="population_decline",
        columns="relocation_friction",
        values="objective_reduction_percent",
    )
    figure, axis = plt.subplots(figsize=(10, 4.8))
    image = axis.imshow(matrix, cmap="YlGnBu", aspect="auto")
    axis.set_xticks(range(len(matrix.columns)), [f"{value:g}" for value in matrix.columns])
    axis.set_yticks(
        range(len(matrix.index)), [f"{value:.0%}" for value in matrix.index]
    )
    axis.set_xlabel("Relocation friction (RF)")
    axis.set_ylabel("Population decline")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            axis.text(column, row, f"{value:.1f}%", ha="center", va="center", fontsize=9)
    figure.colorbar(
        image,
        ax=axis,
        label="Normalized-model-objective reduction versus Scenario A (%)",
    )
    axis.set_title(
        "Scenario D normalized-model-objective reduction relative to Scenario A",
        fontsize=15,
        weight="bold",
    )
    figure.tight_layout()
    return save_figure(figure, "F6_normalized_objective_reduction")


def sensitivity_correlations(sensitivity: pd.DataFrame) -> pd.DataFrame:
    parameters = [
        "relocation_friction",
        "mobile_cost_multiplier",
        "digital_cost_multiplier",
        "impedance_multiplier",
        "capacity_multiplier",
        "mfg_multiplier",
        "population_decline",
    ]
    outcomes = [
        "cost_per_resident",
        "relocated_share",
        "digital_share",
        "cross_boundary_share",
    ]
    records = []
    for outcome in outcomes:
        for parameter in parameters:
            correlation = spearmanr(
                sensitivity[parameter],
                sensitivity[outcome],
            ).statistic
            records.append(
                {
                    "outcome": outcome,
                    "parameter": parameter,
                    "spearman_rho": float(correlation),
                }
            )
    return pd.DataFrame(records)


def figure_7(sensitivity: pd.DataFrame) -> tuple[Path, Path]:
    correlations = sensitivity_correlations(sensitivity)
    matrix = correlations.pivot(
        index="parameter",
        columns="outcome",
        values="spearman_rho",
    )
    figure, axis = plt.subplots(figsize=(9, 6))
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    outcome_labels = {
        "cost_per_resident": "normalized\nobjective\nper resident",
        "cross_boundary_share": "cross-\nboundary\nshare",
        "digital_share": "digital\nshare",
        "relocated_share": "relocated\nshare",
    }
    axis.set_xticks(
        range(len(matrix.columns)),
        [outcome_labels[label] for label in matrix.columns],
    )
    axis.set_yticks(
        range(len(matrix.index)),
        [label.replace("_", " ") for label in matrix.index],
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix.iloc[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )
    figure.colorbar(image, ax=axis, label="Spearman rank correlation")
    axis.set_title(
        "Designed global parameter sensitivity of Scenario D (96 draws)",
        fontsize=15,
        weight="bold",
    )
    figure.tight_layout()
    correlations.to_csv(RESULTS / "sensitivity_correlations.csv", index=False)
    return save_figure(figure, "F7_sensitivity")


def build_editable_deck(images: list[tuple[Path, str]]) -> None:
    presentation = Presentation()
    presentation.slide_width = Inches(13.333)
    presentation.slide_height = Inches(7.5)
    for image, title in images:
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        title_box = slide.shapes.add_textbox(Inches(0.45), Inches(0.15), Inches(12.4), Inches(0.5))
        paragraph = title_box.text_frame.paragraphs[0]
        paragraph.text = title
        paragraph.font.size = Pt(20)
        paragraph.font.bold = True
        with Image.open(image) as opened:
            source_width, source_height = opened.size
        maximum_width = 12.0
        maximum_height = 6.1
        scale = min(maximum_width / source_width, maximum_height / source_height)
        width = source_width * scale
        height = source_height * scale
        left = (13.333 - width) / 2
        slide.shapes.add_picture(
            str(image),
            Inches(left),
            Inches(0.78),
            width=Inches(width),
            height=Inches(height),
        )
        note_box = slide.shapes.add_textbox(Inches(0.55), Inches(7.05), Inches(12), Inches(0.25))
        note = note_box.text_frame.paragraphs[0]
        note.text = (
            "All labels and source notes are editable; the plotted panel is "
            "programmatically generated."
        )
        note.font.size = Pt(8)
    presentation.save(FIGURES / "figures_editable.pptx")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    primary = pd.read_csv(RESULTS / "primary_results.csv")
    transition = pd.read_csv(RESULTS / "transition_refinement.csv")
    sensitivity = pd.read_csv(RESULTS / "sensitivity_results.csv")
    config = load_config()
    outputs = [
        (figure_1()[0], "Figure 1. Analytical framework"),
        (figure_2()[0], "Figure 2. Study area and empirical inputs"),
        (
            figure_3(primary, transition)[0],
            "Figure 3. Normalized objective across RF and depopulation",
        ),
        (figure_4(primary)[0], "Figure 4. Strategy substitution"),
        (figure_5()[0], "Figure 5. Functional catchments"),
        (
            figure_6(primary)[0],
            "Figure 6. Normalized-objective reduction",
        ),
        (
            figure_7(sensitivity)[0],
            "Figure 7. Designed global parameter sensitivity",
        ),
    ]
    build_editable_deck(outputs)
    manifest = {
        "figures": [str(path.relative_to(ROOT)) for path, _ in outputs],
        "model_seed": config["project"]["seed"],
        "source_results": [
            "outputs/results/primary_results.csv",
            "outputs/results/transition_refinement.csv",
            "outputs/results/sensitivity_results.csv",
        ],
    }
    (FIGURES / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
