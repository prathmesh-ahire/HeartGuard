"""FE-05 and the class-conditional overlays (Phase 42).

**This ranking is descriptive and must never feed feature selection.** It scores
every feature against the labels of the *whole* corpus, including the records
that will later sit in test folds. That is fine for a figure whose job is to
show what the matrix looks like, and it is a textbook leak the moment it decides
which columns a model sees. Selection happens inside the training fold in Part
V, scored on that fold alone, and does not import anything from this module --
:func:`rank_by_separation` deliberately returns a table rather than a column
list, so there is nothing here shaped like a selector to reach for by mistake.

The separation statistic is the absolute standardised mean difference (Cohen's
d) between the two binary classes, pooled. It is used because it is scale-free
and reads directly off the overlay: a d of 1.2 is a visible gap between the two
histograms, a d of 0.1 is two curves on top of each other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.feature_extraction.registry import FEATURE_NAMES, family_of
from src.utils.logging_setup import get_logger

__all__ = [
    "FE05_DIRNAME",
    "FE05_MANIFEST",
    "SEPARATION_FILENAME",
    "OVERLAY_FILENAME",
    "TOP_N_PLOTS",
    "TOP_N_OVERLAY",
    "labelled_subset",
    "rank_by_separation",
    "plot_feature_distribution",
    "plot_class_overlays",
    "write_distribution_artifacts",
]

log = get_logger("features.distributions")

FE05_DIRNAME = "feature_distribution_plots"

#: FE-05 is a *directory* of panels, but the evidence index guarantees that every
#: row marked "ok" resolves to a real file -- a directory would satisfy a row
#: while being empty. So the directory carries a manifest, and that manifest is
#: what FE-05 registers: it exists only if panels were written, and it names the
#: feature each panel shows so a reader can trace a figure back to its number.
FE05_MANIFEST = "panel_index.csv"
SEPARATION_FILENAME = "feature_class_separation.csv"
OVERLAY_FILENAME = "class_conditional_top10_D1.png"

#: How many features get their own histogram+boxplot panel (T42.1).
TOP_N_PLOTS = 12

#: How many go into the combined overlay figure (T42.4 says ten).
TOP_N_OVERLAY = 10


def labelled_subset(matrix: Any, dataset: str = "D1") -> Any:
    """The rows of one dataset that are actually usable for a class comparison.

    ``use_in_supervised`` already encodes the exclusions the audit established --
    unlabelled records, duplicates, the ``Heartbeat_Sound`` mirror. Re-deriving
    them here would give two definitions of "the corpus" that could drift.
    """
    subset = matrix[matrix["dataset_source"].astype(str) == dataset]
    if "use_in_supervised" in subset.columns:
        subset = subset[subset["use_in_supervised"].astype(bool)]
    return subset[subset["binary_label"].notna()]


def rank_by_separation(matrix: Any, dataset: str = "D1") -> Any:
    """Per-feature |Cohen's d| between the binary classes. Descriptive only."""
    import pandas as pd

    subset = labelled_subset(matrix, dataset)
    labels = np.asarray(subset["binary_label"], dtype=float)
    classes = sorted(np.unique(labels[np.isfinite(labels)]))
    if len(classes) != 2:
        raise ValueError(
            dataset + " has " + str(len(classes)) + " binary classes, expected 2"
        )

    negative, positive = (labels == classes[0]), (labels == classes[1])
    rows = []
    for name in FEATURE_NAMES:
        values = np.asarray(subset[name], dtype=float)
        left = values[negative & np.isfinite(values)]
        right = values[positive & np.isfinite(values)]
        if left.size < 2 or right.size < 2:
            d = float("nan")
        else:
            pooled = np.sqrt((left.var(ddof=1) + right.var(ddof=1)) / 2.0)
            d = float((right.mean() - left.mean()) / pooled) if pooled > 0 else 0.0
        rows.append(
            {
                "feature": name,
                "family": family_of(name),
                "dataset": dataset,
                "n_class0": int(left.size),
                "n_class1": int(right.size),
                "mean_class0": float(left.mean()) if left.size else float("nan"),
                "mean_class1": float(right.mean()) if right.size else float("nan"),
                "cohens_d": d,
                "abs_cohens_d": abs(d),
            }
        )

    frame = pd.DataFrame(rows).sort_values(
        "abs_cohens_d", ascending=False, na_position="last"
    )
    frame["rank"] = range(1, len(frame) + 1)
    return frame.reset_index(drop=True)


def _class_names(subset: Any) -> dict[float, str]:
    """Map the numeric binary label back onto its name, from master metadata."""
    names: dict[float, str] = {}
    if "binary_label_name" in subset.columns:
        for label, group in subset.groupby("binary_label"):
            value = group["binary_label_name"].dropna()
            if not value.empty:
                names[float(label)] = str(value.iloc[0])
    return names


def plot_feature_distribution(
    path: str | Path, subset: Any, name: str, separation: float
) -> Path:
    """One feature: class-conditional histogram beside a boxplot (T42.1)."""
    from src.reporting.plot_style import class_color, figure, save_figure

    labels = np.asarray(subset["binary_label"], dtype=float)
    values = np.asarray(subset[name], dtype=float)
    classes = sorted(np.unique(labels[np.isfinite(labels)]))
    names = _class_names(subset)

    groups = [values[(labels == label) & np.isfinite(values)] for label in classes]
    pooled = np.concatenate([group for group in groups if group.size])

    # The x-range is clipped to the central 99% so one extreme record cannot
    # compress every bar into the leftmost bin. The clipped count is stated in
    # the caption rather than dropped silently.
    low, high = (float(x) for x in np.percentile(pooled, [0.5, 99.5]))
    if high <= low:
        low, high = float(pooled.min()), float(pooled.max()) or 1.0
    n_clipped = int(((pooled < low) | (pooled > high)).sum())
    edges = np.linspace(low, high, 41)

    fig, axes = figure("double", ncols=2, width_ratios=[3, 1])
    for index, (label, group) in enumerate(zip(classes, groups, strict=True)):
        axes[0].hist(
            np.clip(group, low, high),
            bins=edges,
            alpha=0.6,
            density=True,
            color=class_color(index),
            label=names.get(float(label), "class " + str(int(label)))
            + " (n="
            + str(group.size)
            + ")",
        )
    axes[0].set_xlabel(name)
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=7)
    axes[0].set_title(name + "  |d| = " + format(abs(separation), ".3f"))

    box = axes[1].boxplot(
        [np.clip(group, low, high) for group in groups],
        patch_artist=True,
        showfliers=False,
        widths=0.6,
    )
    for index, patch in enumerate(box["boxes"]):
        patch.set_facecolor(class_color(index))
        patch.set_alpha(0.6)
    axes[1].set_xticklabels(
        [names.get(float(label), str(int(label)))[:8] for label in classes], fontsize=7
    )
    axes[1].set_ylabel(name, fontsize=7)
    fig.tight_layout()

    return save_figure(
        fig,
        path,
        source=(
            "FE-05 | source: all_features_matrix.parquet | "
            + str(len(subset))
            + " labelled records | axes clipped to the central 99% ("
            + str(n_clipped)
            + " values clipped)\nCohen's d is computed over all labelled records "
            "and is descriptive only -- it does not select features (rule 2)."
        ),
    )


def plot_class_overlays(path: str | Path, subset: Any, ranking: Any) -> Path:
    """T42.4 -- the ten most separating features on one page, as density overlays."""
    from src.reporting.plot_style import class_color, figure, save_figure

    top = ranking.head(TOP_N_OVERLAY)
    labels = np.asarray(subset["binary_label"], dtype=float)
    classes = sorted(np.unique(labels[np.isfinite(labels)]))
    names = _class_names(subset)

    fig, axes = figure((10.0, 7.5), nrows=5, ncols=2)
    flat = np.asarray(axes).ravel()

    for axis, row in zip(flat, top.itertuples(index=False), strict=False):
        values = np.asarray(subset[row.feature], dtype=float)
        finite = values[np.isfinite(values)]
        low, high = (float(x) for x in np.percentile(finite, [0.5, 99.5]))
        if high <= low:
            low, high = float(finite.min()), float(finite.max()) or 1.0
        edges = np.linspace(low, high, 41)

        for index, label in enumerate(classes):
            group = values[(labels == label) & np.isfinite(values)]
            axis.hist(
                np.clip(group, low, high),
                bins=edges,
                alpha=0.55,
                density=True,
                color=class_color(index),
                label=names.get(float(label), "class " + str(int(label))),
            )
        axis.set_title(
            "#"
            + str(row.rank)
            + "  "
            + row.feature
            + "  |d|="
            + format(row.abs_cohens_d, ".2f"),
            fontsize=7,
        )
        axis.tick_params(labelsize=6)
        axis.set_yticks([])

    flat[0].legend(fontsize=6, loc="upper right")
    for axis in flat[len(top) :]:
        axis.axis("off")
    fig.tight_layout()

    return save_figure(
        fig,
        path,
        source=(
            "T42.4 | source: all_features_matrix.parquet, D1 | "
            + str(len(subset))
            + " labelled records | ranked by |Cohen's d|, axes clipped to the\n"
            "central 99%. Descriptive: this ranking never selects features -- "
            "selection is fitted inside the training fold (rule 2)."
        ),
    )


def write_distribution_artifacts(
    matrix: Any, out_dir: str | Path | None = None, *, dataset: str = "D1"
) -> dict[str, Path]:
    """Emit FE-05, its ranking CSV and the T42.4 overlay page."""
    from src.feature_extraction.quality import features_dir
    from src.utils.io import ensure_dir

    directory = features_dir(out_dir)
    plots_dir = ensure_dir(directory / FE05_DIRNAME)

    ranking = rank_by_separation(matrix, dataset)
    subset = labelled_subset(matrix, dataset)

    written: dict[str, Path] = {}
    ranking_path = directory / SEPARATION_FILENAME
    ranking.to_csv(ranking_path, index=False)
    written["separation_csv"] = ranking_path

    panels = []
    for row in ranking.head(TOP_N_PLOTS).itertuples(index=False):
        target = plots_dir / (
            format(row.rank, "02d") + "_" + row.feature + ".png"
        )
        plot_feature_distribution(target, subset, row.feature, row.cohens_d)
        panels.append(
            {
                "rank": int(row.rank),
                "feature": row.feature,
                "family": row.family,
                "cohens_d": float(row.cohens_d),
                "n_class0": int(row.n_class0),
                "n_class1": int(row.n_class1),
                "filename": target.name,
            }
        )

    import pandas as pd

    manifest = plots_dir / FE05_MANIFEST
    pd.DataFrame(panels).to_csv(manifest, index=False)
    written["FE-05"] = manifest
    written["FE-05-dir"] = plots_dir

    written["overlays"] = plot_class_overlays(
        directory / OVERLAY_FILENAME, subset, ranking
    )

    log.info(
        "FE-05: %d panels in %s; top feature %s (|d|=%.3f)",
        TOP_N_PLOTS,
        plots_dir,
        ranking.iloc[0]["feature"],
        ranking.iloc[0]["abs_cohens_d"],
    )
    return written
