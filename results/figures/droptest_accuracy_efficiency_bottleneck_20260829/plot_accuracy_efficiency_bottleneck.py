#!/usr/bin/env python3
"""Render the validation-tuned DropTest accuracy/efficiency paper figure."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean, stdev

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyBboxPatch


EXPERIMENT_ROOT = Path(
    "/work/choulin420/physicsnemo_data/experiments/"
    "droptest_official_backbones_20260826"
)
TUNED_EXPERIMENT_ROOT = Path(
    "/work/choulin420/physicsnemo_data/experiments/"
    "droptest_official_backbones_tuned_20260828"
)
OUTPUT_DIR = Path(__file__).resolve().parent
BASELINE_MAIN_TABLE = EXPERIMENT_ROOT / "main_table.csv"
TUNED_MAIN_TABLE = TUNED_EXPERIMENT_ROOT / "deadline_k1_main_table.csv"
MATCHED_PROFILE_DIR = EXPERIMENT_ROOT / "profiling/matched_h200_20260828"
TUNED_MATCHED_PROFILE = (
    TUNED_EXPERIMENT_ROOT
    / "profiling/matched_h200_20260829/lts_ca_fourier_tenv.json"
)
REFERENCE_IMAGE = Path(
    "/home/choulin420/scp/截圖 2026-08-28 上午10.36.59.png"
)

ORDER = (
    "transolver",
    "geots_flare",
    "geotransolver",
    "transolver_plus",
    "transolver3",
    "lts_ca_fourier_tenv",
)

DISPLAY = {
    "transolver": "Transolver",
    "geots_flare": "GeoTS-FLARE",
    "geotransolver": "GeoTransolver",
    "transolver_plus": "Transolver++",
    "transolver3": "Transolver-3",
    "lts": "LTS w/o CA",
    "lts_ca": "LTS w/ CA",
    "lts_ca_fourier_tenv": "LTS",
}

COLORS = {
    "transolver": "#e67e22",
    "geots_flare": "#f2bd3f",
    "geotransolver": "#2d9f73",
    "transolver_plus": "#727c83",
    "transolver3": "#7758b5",
    "lts": "#327cb9",
    "lts_ca": "#d94b5a",
    "lts_ca_fourier_tenv": "#d94b5a",
}

# Every label is centered directly above its marker. A negative value is used
# only when the upper label lane is already occupied by a neighboring model.
LABEL_VERTICAL_DIRECTION = {
    "transolver": 1,
    "transolver3": -1,
    "geots_flare": 1,
    "geotransolver": -1,
    "transolver_plus": 1,
    "lts_ca_fourier_tenv": 1,
}

# Transolver-3 lies inside the much larger Transolver bubble, so its lower
# label must clear the outer Transolver circumference rather than only its own.
LABEL_MINIMUM_CLEARANCE_POINTS = {
    "transolver3": 27.0,
}

LABEL_HORIZONTAL_OFFSET_POINTS = {
    # Centering this long name over the low-latency marker crosses the y-axis.
    "transolver_plus": 22.0,
}

BREAKDOWN_LABELS = (
    "Slice-weight\nComputation",
    "Mesh→slice",
    "Latent",
    "Slice→mesh",
    "Mesh FFN",
    "Component\nadaptation",
    "Other\ninstrumented",
    "Unattributable",
)
BREAKDOWN_COLORS = (
    "#4f83b6",
    "#6ab8ad",
    "#f2c744",
    "#aa6f9d",
    "#f28b25",
    "#ef5350",
    "#58a64e",
    "#a7a7a7",
)
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def formal_sources_available() -> bool:
    profile_paths = [
        MATCHED_PROFILE_DIR / f"{model}.json"
        for model in ORDER
        if model != "lts_ca_fourier_tenv"
    ]
    return all(
        path.is_file()
        for path in (
            BASELINE_MAIN_TABLE,
            TUNED_MAIN_TABLE,
            TUNED_MATCHED_PROFILE,
            *profile_paths,
        )
    )


def load_rows() -> list[dict[str, object]]:
    if not formal_sources_available():
        packaged_data = OUTPUT_DIR / "plot_data.json"
        if not packaged_data.is_file():
            raise FileNotFoundError(
                "Neither the formal cluster sources nor packaged plot_data.json "
                "are available"
            )
        return json.loads(packaged_data.read_text())["rows"]

    with BASELINE_MAIN_TABLE.open(newline="") as handle:
        baseline_table = {row["model"]: row for row in csv.DictReader(handle)}
    with TUNED_MAIN_TABLE.open(newline="") as handle:
        tuned_table = {row["model"]: row for row in csv.DictReader(handle)}

    rows: list[dict[str, object]] = []
    for model in ORDER:
        if model == "lts_ca_fourier_tenv":
            source = tuned_table["lts_ca_fourier_tenv_validation_tuned_k1"]
            profile_path = TUNED_MATCHED_PROFILE
        else:
            source = baseline_table[model]
            profile_path = MATCHED_PROFILE_DIR / f"{model}.json"
        displacement_per_seed = json.loads(
            source["displacement_physical_relative_l2_macro_per_seed"]
        )
        stress_per_seed = json.loads(
            source["stress_physical_relative_l2_macro_per_seed"]
        )
        equal_field_per_seed = [
            0.5 * (float(displacement) + float(stress))
            for displacement, stress in zip(
                displacement_per_seed, stress_per_seed, strict=True
            )
        ]
        if not profile_path.is_file():
            raise FileNotFoundError(
                f"Matched H200 profiler output is required: {profile_path}"
            )
        profile = json.loads(profile_path.read_text())
        if profile.get("status") != "PASS" or profile.get("model") != model:
            raise RuntimeError(f"Invalid matched profiler output: {profile_path}")
        rows.append(
            {
                "model": model,
                "display": DISPLAY[model],
                "seeds": [7, 17, 27],
                "equal_field_relative_l2_fraction_mean": fmean(equal_field_per_seed),
                "equal_field_relative_l2_fraction_sample_std": stdev(equal_field_per_seed),
                "equal_field_relative_l2_percent_mean": 100.0 * fmean(equal_field_per_seed),
                "equal_field_relative_l2_percent_sample_std": 100.0 * stdev(equal_field_per_seed),
                "equal_field_relative_l2_fraction_per_seed": equal_field_per_seed,
                "displacement_physical_relative_l2_fraction_per_seed": displacement_per_seed,
                "stress_physical_relative_l2_fraction_per_seed": stress_per_seed,
                "train_step_latency_ms_matched_h200": float(
                    profile["train_step_latency_ms"]["mean"]
                ),
                "train_step_latency_ms_sample_std": float(
                    profile["train_step_latency_ms"]["sample_std"]
                ),
                "peak_allocated_gib_matched_h200": float(profile["peak_allocated_gib"]),
                "matched_profile": str(profile_path),
                "matched_profile_sha256": sha256(profile_path),
                "component_breakdown_percent": (
                    profile.get("component_breakdown") or {}
                ).get("percent_of_dedicated_total"),
            }
        )
    return rows


def memory_size(memory: float, minimum: float, maximum: float) -> float:
    """Map memory to marker area while preserving ordering and reference readability."""
    if maximum == minimum:
        return 650.0
    normalized = (memory - minimum) / (maximum - minimum)
    return 150.0 + 1_150.0 * normalized


def external_tangents_in_axes(
    ax: mpl.axes.Axes,
    small_center_axes: tuple[float, float],
    large_center_axes: tuple[float, float],
    small_radius_px: float,
    large_radius_px: float,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return exact common-external-tangent endpoints in axes coordinates.

    Geometry is solved in display pixels, so the result remains exact even if
    the axes is not square. Each returned segment runs from the small-circle
    tangent point to the corresponding large-circle tangent point.
    """

    small = np.asarray(ax.transAxes.transform(small_center_axes), dtype=float)
    large = np.asarray(ax.transAxes.transform(large_center_axes), dtype=float)
    displacement = large - small
    distance = float(np.linalg.norm(displacement))
    radius_delta = float(large_radius_px - small_radius_px)
    if distance <= abs(radius_delta):
        raise ValueError("Memory-legend circles do not admit external tangents")
    direction = displacement / distance
    perpendicular = np.array([-direction[1], direction[0]])
    normal_parallel = -radius_delta / distance
    normal_perpendicular = np.sqrt(1.0 - normal_parallel**2)
    inverse = ax.transAxes.inverted()
    segments = []
    for sign in (1.0, -1.0):
        normal = (
            normal_parallel * direction
            + sign * normal_perpendicular * perpendicular
        )
        small_tangent = small + small_radius_px * normal
        large_tangent = large + large_radius_px * normal
        segments.append(
            (
                np.asarray(inverse.transform(small_tangent)),
                np.asarray(inverse.transform(large_tangent)),
            )
        )
    return tuple(segments)  # type: ignore[return-value]


def add_donut(
    ax: mpl.axes.Axes,
    values: tuple[float, ...],
    center_latency: float,
    panel_title: str,
    printed: tuple[int, ...],
) -> None:
    wedges, _ = ax.pie(
        values,
        startangle=90,
        counterclock=False,
        colors=BREAKDOWN_COLORS,
        wedgeprops={"width": 0.36, "edgecolor": "#e8ecef", "linewidth": 1.0},
    )
    ax.set_aspect("equal")
    ax.set_title(panel_title, fontsize=12.5, fontweight="bold", pad=6)
    ax.text(
        0,
        0.07,
        f"{center_latency:.1f}\nms",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        linespacing=0.82,
    )
    ax.text(
        0,
        -0.25,
        "50-step mean",
        ha="center",
        va="center",
        fontsize=7.6,
        color="#667078",
    )
    for index in printed:
        wedge = wedges[index]
        angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2.0)
        radius = 0.82
        ax.text(
            radius * np.cos(angle),
            radius * np.sin(angle),
            f"{values[index]:.1f}%",
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="bold",
        )
    ax.set_xticks([])
    ax.set_yticks([])


def draw(rows: list[dict[str, object]]) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
            "axes.edgecolor": "#9aa3aa",
            "axes.labelcolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "text.color": "#111111",
            "figure.facecolor": "#f8fafb",
            "axes.facecolor": "#f8fafb",
            "savefig.facecolor": "#f8fafb",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    by_model = {str(row["model"]): row for row in rows}
    breakdowns = {}
    for model in ("transolver", "lts_ca_fourier_tenv"):
        breakdowns[model] = tuple(
            by_model[model]["component_breakdown_percent"][label.replace("\n", " ")]
            for label in BREAKDOWN_LABELS
        )
    memories = [float(row["peak_allocated_gib_matched_h200"]) for row in rows]
    minimum_memory, maximum_memory = min(memories), max(memories)

    # Exact dimensions of the supplied reference screenshot: 976 x 624 pixels.
    fig = plt.figure(figsize=(9.76, 6.24), dpi=100)
    ax = fig.add_axes([0.075, 0.115, 0.508, 0.795])

    x_values = [float(row["train_step_latency_ms_matched_h200"]) for row in rows]
    y_values = [float(row["equal_field_relative_l2_percent_mean"]) for row in rows]
    x_upper = 1.15 * max(x_values)
    y_lower = max(0.0, min(y_values) - 0.35)
    y_upper = max(y_values) + 0.35
    ax.set_xlim(0.0, x_upper)
    ax.set_ylim(y_lower, y_upper)
    ax.grid(True, color="#dfe4e8", linewidth=1.0, alpha=0.92)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.tick_params(labelsize=10, width=1.0, length=3)

    for row in rows:
        model = str(row["model"])
        x = float(row["train_step_latency_ms_matched_h200"])
        y = float(row["equal_field_relative_l2_percent_mean"])
        size = memory_size(
            float(row["peak_allocated_gib_matched_h200"]), minimum_memory, maximum_memory
        )
        ax.scatter(
            [x],
            [y],
            s=size,
            color=COLORS[model],
            edgecolor="#f8fafb",
            linewidth=2.0,
            alpha=0.97,
            zorder=3,
        )
        direction = LABEL_VERTICAL_DIRECTION[model]
        marker_radius_points = 0.5 * np.sqrt(size)
        clearance = max(
            marker_radius_points + 4.0,
            LABEL_MINIMUM_CLEARANCE_POINTS.get(model, 0.0),
        )
        dy = direction * clearance
        dx = LABEL_HORIZONTAL_OFFSET_POINTS.get(model, 0.0)
        ax.annotate(
            DISPLAY[model],
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="bottom" if direction > 0 else "top",
            fontsize=10.5,
            fontweight="bold",
            color=COLORS[model],
            zorder=5,
        )

    ax.set_title(
        "(a) Accuracy-Efficiency Comparison",
        fontsize=15.5,
        fontweight="bold",
        pad=6,
    )
    ax.set_xlabel("Train-step Latency (ms) ↓", fontsize=18, fontweight="bold", labelpad=6)
    ax.set_ylabel(
        "Equal-field Rel. L2 (%) ↓", fontsize=16.5, fontweight="bold", labelpad=7
    )

    # Lower-left direction cue, reproduced from the reference figure.
    cue_x = 0.075 * x_upper
    cue_y = y_lower + 0.10 * (y_upper - y_lower)
    ax.annotate(
        "",
        xy=(0.018 * x_upper, y_lower + 0.018 * (y_upper - y_lower)),
        xytext=(cue_x, cue_y),
        arrowprops={"arrowstyle": "->", "color": "#111111", "linewidth": 1.2},
    )
    ax.text(
        cue_x,
        cue_y + 0.018 * (y_upper - y_lower),
        "Better",
        fontsize=10.5,
        fontweight="bold",
        color="#556860",
        ha="left",
    )

    # Memory bubble legend box.
    box_x, box_y, box_w, box_h = 0.59, 0.70, 0.37, 0.25
    memory_box = FancyBboxPatch(
        (box_x, box_y),
        box_w,
        box_h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        transform=ax.transAxes,
        facecolor="#ffffff",
        edgecolor="#cfd5da",
        linewidth=1.1,
        zorder=6,
    )
    ax.add_patch(memory_box)
    ax.text(
        box_x + box_w / 2,
        box_y + box_h - 0.035,
        "Peak Train Alloc. Mem.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11.8,
        fontweight="bold",
        color="#3e464c",
        zorder=8,
    )
    small_center = (box_x + 0.075, box_y + 0.115)
    large_center = (box_x + 0.265, box_y + 0.125)
    small_radius_px = 8.5
    large_radius_px = 25.5
    tangent_segments = external_tangents_in_axes(
        ax,
        small_center,
        large_center,
        small_radius_px,
        large_radius_px,
    )
    for start, end in tangent_segments:
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            transform=ax.transAxes,
            color="#909aa2",
            linewidth=1.0,
            linestyle=(0, (2, 2)),
            dash_capstyle="round",
            zorder=7,
        )
        # A short solid cap removes any dash-phase gap at the exact tangency
        # points without changing the dashed guide between the circles.
        vector = end - start
        length = float(np.linalg.norm(vector))
        unit = vector / length
        cap = 1.5 / float(ax.bbox.width)
        for cap_start, cap_end in (
            (start, start + cap * unit),
            (end - cap * unit, end),
        ):
            ax.plot(
                [cap_start[0], cap_end[0]],
                [cap_start[1], cap_end[1]],
                transform=ax.transAxes,
                color="#909aa2",
                linewidth=1.0,
                solid_capstyle="round",
                zorder=7,
            )
    for center, radius_px in (
        (small_center, small_radius_px),
        (large_center, large_radius_px),
    ):
        ax.add_patch(
            Ellipse(
                center,
                width=2.0 * radius_px / float(ax.bbox.width),
                height=2.0 * radius_px / float(ax.bbox.height),
                transform=ax.transAxes,
                facecolor="#ffffff",
                edgecolor="#9ca8af",
                linewidth=1.0,
                zorder=8,
                clip_on=False,
            )
        )
    ax.text(
        small_center[0],
        box_y + 0.012,
        f"{minimum_memory:.2f} GiB",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.8,
        fontweight="bold",
        zorder=8,
    )
    ax.text(
        large_center[0],
        box_y + 0.012,
        f"{maximum_memory:.2f} GiB",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.8,
        fontweight="bold",
        zorder=8,
    )

    # Reference-style vertical panel divider.
    fig.add_artist(
        Line2D(
            [0.595, 0.595],
            [0.02, 0.98],
            transform=fig.transFigure,
            color="#79838a",
            linewidth=1.1,
        )
    )
    fig.text(
        0.611,
        0.935,
        "(b) Efficiency Bottleneck Analysis",
        fontsize=14.0,
        fontweight="bold",
        ha="left",
    )

    top = fig.add_axes([0.618, 0.48, 0.225, 0.39])
    bottom = fig.add_axes([0.618, 0.075, 0.225, 0.39])
    add_donut(
        top,
        breakdowns["transolver"],
        float(by_model["transolver"]["train_step_latency_ms_matched_h200"]),
        "(a) Transolver",
        tuple(index for index, value in enumerate(breakdowns["transolver"]) if value >= 7.0),
    )
    add_donut(
        bottom,
        breakdowns["lts_ca_fourier_tenv"],
        float(
            by_model["lts_ca_fourier_tenv"]["train_step_latency_ms_matched_h200"]
        ),
        "(b) LTS",
        tuple(
            index
            for index, value in enumerate(breakdowns["lts_ca_fourier_tenv"])
            if value >= 7.0
        ),
    )

    legend_ax = fig.add_axes([0.848, 0.22, 0.15, 0.52])
    legend_ax.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markersize=7.2,
            markerfacecolor=color,
            markeredgecolor="#f8fafb",
        )
        for color in BREAKDOWN_COLORS
    ]
    legend_ax.legend(
        handles,
        BREAKDOWN_LABELS,
        loc="center left",
        frameon=False,
        fontsize=9.8,
        handlelength=0.8,
        handletextpad=0.45,
        labelspacing=0.55,
        borderaxespad=0,
        prop={"weight": "bold", "size": 9.8},
    )

    exact_png = OUTPUT_DIR / "droptest_accuracy_efficiency_bottleneck_20260829.png"
    hires_png = OUTPUT_DIR / "droptest_accuracy_efficiency_bottleneck_20260829_2x.png"
    pdf = OUTPUT_DIR / "droptest_accuracy_efficiency_bottleneck_20260829.pdf"
    svg = OUTPUT_DIR / "droptest_accuracy_efficiency_bottleneck_20260829.svg"
    fig.savefig(exact_png, dpi=100, bbox_inches=None, pad_inches=0)
    fig.savefig(hires_png, dpi=200, bbox_inches=None, pad_inches=0)
    fig.savefig(pdf, bbox_inches=None, pad_inches=0)
    fig.savefig(svg, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    # Matplotlib emits path-data lines with trailing spaces. Normalize them so
    # Git's whitespace checks and the published checksum remain stable.
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )


def write_data(rows: list[dict[str, object]]) -> None:
    rows_by_model = {str(row["model"]): row for row in rows}
    breakdown_percent = {
        model: rows_by_model[model]["component_breakdown_percent"]
        for model in ("transolver", "lts_ca_fourier_tenv")
    }
    payload = {
        "format": "droptest_accuracy_efficiency_bottleneck_20260829_v1",
        "source_baseline_main_table": str(BASELINE_MAIN_TABLE),
        "source_baseline_main_table_sha256": sha256(BASELINE_MAIN_TABLE),
        "source_tuned_lts_main_table": str(TUNED_MAIN_TABLE),
        "source_tuned_lts_main_table_sha256": sha256(TUNED_MAIN_TABLE),
        "reference_image": str(REFERENCE_IMAGE) if REFERENCE_IMAGE.is_file() else None,
        "reference_image_sha256": sha256(REFERENCE_IMAGE) if REFERENCE_IMAGE.is_file() else None,
        "accuracy_definition": (
            "100 * arithmetic mean across seeds 7,17,27 of "
            "0.5 * (displacement_physical_relative_l2_macro + "
            "stress_physical_relative_l2_macro); this is the latest "
            "compact main-table equal-field definition"
        ),
        "latency_definition": (
            "matched single-H200 BF16 batch-1 dedicated profile; preloaded CUDA "
            "sample; 20 warmups + 50 measured steps; BF16 forward + formal loss + "
            "backward + fused Adam optimizer.step; dataloader and zero_grad excluded"
        ),
        "memory_definition": (
            "maximum allocated CUDA memory from the same matched 50-step H200 profile"
        ),
        "right_panel_status": (
            "Matched component profiles for Transolver and validation-tuned LTS; explicit "
            "forward ranges with backward CUDA kernels correlated via autograd "
            "sequence_nr; remaining wall time reported as Unattributable."
        ),
        "breakdown_categories": list(BREAKDOWN_LABELS),
        "breakdown_percent": breakdown_percent,
        "rows": rows,
    }
    (OUTPUT_DIR / "plot_data.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    columns = [
        "model",
        "display",
        "equal_field_relative_l2_fraction_mean",
        "equal_field_relative_l2_fraction_sample_std",
        "equal_field_relative_l2_percent_mean",
        "equal_field_relative_l2_percent_sample_std",
        "train_step_latency_ms_matched_h200",
        "train_step_latency_ms_sample_std",
        "peak_allocated_gib_matched_h200",
        "matched_profile",
        "matched_profile_sha256",
    ]
    with (OUTPUT_DIR / "plot_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_checksums() -> None:
    names = (
        "droptest_accuracy_efficiency_bottleneck_20260829.png",
        "droptest_accuracy_efficiency_bottleneck_20260829_2x.png",
        "droptest_accuracy_efficiency_bottleneck_20260829.pdf",
        "droptest_accuracy_efficiency_bottleneck_20260829.svg",
        "plot_data.csv",
        "plot_data.json",
        "plot_accuracy_efficiency_bottleneck.py",
        "README.md",
    )
    lines = [f"{sha256(OUTPUT_DIR / name)}  {name}" for name in names]
    (OUTPUT_DIR / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    refresh_data = formal_sources_available()
    rows = load_rows()
    if refresh_data:
        write_data(rows)
    draw(rows)
    write_checksums()


if __name__ == "__main__":
    main()
