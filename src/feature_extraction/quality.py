"""Feature quality assurance over FE-03 (Phase 41).

Five reports and one figure, all descriptive. **Nothing here drops, clips or
transforms a feature.** That distinction is the whole point of the phase: the
matrix is inspected once, globally, so the write-up can say what is in it -- but
every decision this inspection motivates (dropping a constant column, clipping a
tail, standardising) is fitted *inside the training fold* in Part V, never here.
Applying a global threshold now would fit it on the test folds too, which is
exactly the leak research rule 2 exists to prevent.

So a feature that is constant across the whole corpus is *reported* here and
still handed to the pipeline; the fold's own selector removes it, using only the
rows that fold is allowed to see.

Reports
-------
FE-04  ``feature_missing_values.csv``    per-feature NaN/Inf with named records
       ``feature_variance_report.csv``   constant and near-zero-variance columns
       ``feature_outlier_report.csv``    tails, ranges and the clipping policy
       ``feature_domain_shift.csv``      per-dataset means, for EXP-D1
FE-10  ``feature_correlation_matrix.png`` + ``.csv``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.feature_extraction.registry import FAMILY_ORDER, FEATURE_NAMES, family_of
from src.utils.logging_setup import get_logger

__all__ = [
    "QA_ARTIFACTS",
    "MAX_NAMED_RECORDS",
    "NEAR_ZERO_VARIANCE_STD",
    "OUTLIER_ROBUST_Z",
    "CLIPPING_POLICY",
    "missing_value_report",
    "variance_report",
    "outlier_report",
    "domain_shift_report",
    "correlation_matrix",
    "plot_correlation_matrix",
    "reproducibility_check",
    "write_quality_artifacts",
]

log = get_logger("features.quality")

QA_ARTIFACTS: dict[str, str] = {
    "FE-04": "feature_missing_values.csv",
    "FE-10": "feature_correlation_matrix.png",
}

#: Extra tables that are not themselves numbered evidence.
EXTRA_ARTIFACTS: dict[str, str] = {
    "variance": "feature_variance_report.csv",
    "outliers": "feature_outlier_report.csv",
    "domain_shift": "feature_domain_shift.csv",
    "correlation_csv": "feature_correlation_matrix.csv",
}

#: How many offending ``record_uid`` values FE-04 spells out per feature. The
#: count is always exact; the list is truncated so one pathological feature
#: cannot turn the report into a 7,536-entry cell.
MAX_NAMED_RECORDS = 20

#: Below this standard deviation a column carries essentially no information.
NEAR_ZERO_VARIANCE_STD = 1e-8

#: |median-centred value| / (1.4826 * MAD) beyond this counts as an extreme tail.
OUTLIER_ROBUST_Z = 10.0

CLIPPING_POLICY = (
    "Report only -- nothing is clipped in Phase 41. The recommended treatment is "
    "a symmetric winsorisation at the 0.5th and 99.5th percentiles for features "
    "flagged unbounded_tail, followed by the fold's standard scaler. Both the "
    "percentiles and the scaler MUST be fitted on the training fold alone "
    "(research rule 2); fitting them on this whole-corpus table would leak the "
    "test folds. Features flagged heavy_tail but bounded need no treatment "
    "beyond scaling."
)


def features_dir(out_dir: str | Path | None = None) -> Path:
    from src.utils.config import load_config
    from src.utils.io import ensure_dir

    if out_dir is not None:
        return ensure_dir(out_dir)
    return ensure_dir(load_config("paths").require("outputs.features"))


def _values(matrix: Any, name: str) -> np.ndarray:
    return np.asarray(matrix[name], dtype=np.float64)


# ---------------------------------------------------------------------------
# T41.1 -- FE-04
# ---------------------------------------------------------------------------


def missing_value_report(matrix: Any) -> Any:
    """Per-feature NaN and Inf counts, with the responsible records named.

    Every non-finite cell in the matrix is accounted for by exactly one row of
    this table: the gate re-derives the corpus total from the per-feature counts
    and compares. A report that undercounts is worse than none, because it reads
    as a clean bill of health.
    """
    import pandas as pd

    uids = matrix["record_uid"].astype(str).to_numpy()
    rows = []
    for name in FEATURE_NAMES:
        values = _values(matrix, name)
        nan_mask = np.isnan(values)
        inf_mask = np.isinf(values)
        offenders = uids[nan_mask | inf_mask]
        rows.append(
            {
                "feature": name,
                "family": family_of(name),
                "n_records": int(values.size),
                "n_nan": int(nan_mask.sum()),
                "n_inf": int(inf_mask.sum()),
                "n_nonfinite": int(offenders.size),
                "pct_nonfinite": round(100.0 * offenders.size / values.size, 6),
                "record_uids": "; ".join(sorted(offenders)[:MAX_NAMED_RECORDS]),
                "record_uids_truncated": bool(offenders.size > MAX_NAMED_RECORDS),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# T41.2 -- constant and near-zero-variance
# ---------------------------------------------------------------------------


def variance_report(matrix: Any) -> Any:
    """Constant and near-zero-variance columns. Reported, deliberately not dropped."""
    import pandas as pd

    rows = []
    for name in FEATURE_NAMES:
        values = _values(matrix, name)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            rows.append(
                {
                    "feature": name,
                    "family": family_of(name),
                    "n_finite": 0,
                    "n_unique": 0,
                    "std": float("nan"),
                    "variance": float("nan"),
                    "unique_ratio": float("nan"),
                    "verdict": "all_nonfinite",
                }
            )
            continue

        std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
        n_unique = int(np.unique(finite).size)
        if n_unique == 1:
            verdict = "constant"
        elif std < NEAR_ZERO_VARIANCE_STD:
            verdict = "near_zero_variance"
        else:
            verdict = "ok"
        rows.append(
            {
                "feature": name,
                "family": family_of(name),
                "n_finite": int(finite.size),
                "n_unique": n_unique,
                "std": std,
                "variance": float(std**2),
                "unique_ratio": round(n_unique / finite.size, 6),
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# T41.3 -- tails, ranges, and the policy
# ---------------------------------------------------------------------------


def outlier_report(matrix: Any) -> Any:
    """Robust-z tails and observed ranges, with a per-feature policy column."""
    import pandas as pd

    rows = []
    for name in FEATURE_NAMES:
        values = _values(matrix, name)
        finite = values[np.isfinite(values)]
        if finite.size < 3:
            rows.append(
                {
                    "feature": name,
                    "family": family_of(name),
                    "n_finite": int(finite.size),
                    "min": float("nan"),
                    "p00_5": float("nan"),
                    "median": float("nan"),
                    "p99_5": float("nan"),
                    "max": float("nan"),
                    "robust_scale": float("nan"),
                    "max_robust_z": float("nan"),
                    "n_extreme": 0,
                    "pct_extreme": 0.0,
                    "verdict": "too_few_values",
                    "policy": "none",
                }
            )
            continue

        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        scale = 1.4826 * mad
        if scale > 0.0:
            robust_z = np.abs(finite - median) / scale
        else:
            # A feature whose middle half is a single value: fall back to the
            # standard deviation rather than reporting an infinite z.
            std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
            robust_z = np.abs(finite - median) / std if std > 0 else np.zeros_like(finite)

        n_extreme = int((robust_z > OUTLIER_ROBUST_Z).sum())
        low, high = (float(x) for x in np.percentile(finite, [0.5, 99.5]))
        span = float(finite.max() - finite.min())
        inner_span = high - low

        # "Unbounded" here means the observed tail dwarfs the central 99%: the
        # feature has no natural ceiling and one record can dominate a scaler.
        unbounded = inner_span > 0 and span > 50.0 * inner_span
        if n_extreme == 0:
            verdict, policy = "ok", "none"
        elif unbounded:
            verdict, policy = "unbounded_tail", "winsorise_0.5_99.5_in_fold"
        else:
            verdict, policy = "heavy_tail", "scale_only"

        rows.append(
            {
                "feature": name,
                "family": family_of(name),
                "n_finite": int(finite.size),
                "min": float(finite.min()),
                "p00_5": low,
                "median": median,
                "p99_5": high,
                "max": float(finite.max()),
                "robust_scale": scale,
                "max_robust_z": float(robust_z.max()),
                "n_extreme": n_extreme,
                "pct_extreme": round(100.0 * n_extreme / finite.size, 6),
                "verdict": verdict,
                "policy": policy,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# T41.5 -- domain shift
# ---------------------------------------------------------------------------


def domain_shift_report(matrix: Any) -> Any:
    """Per-dataset means and the D1 -> D4 standardised difference.

    EXP-D1 trains on PhysioNet adults and tests on CirCor children. A large drop
    there is a population effect, and this table is the evidence for saying so
    rather than asserting it: it shows the feature distributions themselves
    differ before any model is involved.
    """
    import pandas as pd

    groups = {
        str(name): group for name, group in matrix.groupby("dataset_source", sort=True)
    }
    rows = []
    for name in FEATURE_NAMES:
        row: dict[str, Any] = {"feature": name, "family": family_of(name)}
        stats: dict[str, tuple[float, float]] = {}
        for dataset, group in groups.items():
            values = _values(group, name)
            finite = values[np.isfinite(values)]
            mean = float(finite.mean()) if finite.size else float("nan")
            std = float(finite.std(ddof=1)) if finite.size > 1 else float("nan")
            stats[dataset] = (mean, std)
            row["mean_" + dataset] = mean
            row["std_" + dataset] = std

        if "D1" in stats and "D4" in stats:
            (mean_1, std_1), (mean_4, std_4) = stats["D1"], stats["D4"]
            pooled = np.sqrt((std_1**2 + std_4**2) / 2.0)
            row["smd_D1_vs_D4"] = (
                float((mean_4 - mean_1) / pooled)
                if np.isfinite(pooled) and pooled > 0
                else float("nan")
            )
            row["abs_smd_D1_vs_D4"] = abs(row["smd_D1_vs_D4"])
        rows.append(row)

    frame = pd.DataFrame(rows)
    if "abs_smd_D1_vs_D4" in frame:
        frame = frame.sort_values("abs_smd_D1_vs_D4", ascending=False, na_position="last")
    return frame.reset_index(drop=True)


# ---------------------------------------------------------------------------
# T41.6 -- FE-10
# ---------------------------------------------------------------------------


def correlation_matrix(matrix: Any, method: str = "pearson") -> Any:
    """138x138 correlation over the finite values, in registry order."""
    frame = matrix[list(FEATURE_NAMES)].astype("float64")
    return frame.corr(method=method).reindex(
        index=list(FEATURE_NAMES), columns=list(FEATURE_NAMES)
    )


def plot_correlation_matrix(path: str | Path, corr: Any) -> Path:
    """FE-10, with the six family blocks marked so the structure is readable.

    A 138x138 grid of unlabelled cells is a decoration. The thing worth seeing
    here is *block* structure -- the 39 MFCC columns correlating with each other
    far more than with anything else -- and that is invisible without the family
    boundaries drawn on.
    """
    from src.reporting.plot_style import DIVERGING_CMAP, figure, save_figure

    values = corr.to_numpy(dtype=float)
    fig, axis = figure("tall")
    image = axis.imshow(
        values, cmap=DIVERGING_CMAP, vmin=-1.0, vmax=1.0, interpolation="nearest"
    )

    boundaries: list[float] = []
    centres: list[float] = []
    labels: list[str] = []
    cursor = 0
    for family in FAMILY_ORDER:
        count = sum(1 for name in FEATURE_NAMES if family_of(name) == family)
        centres.append(cursor + count / 2.0 - 0.5)
        labels.append(family + "\n(" + str(count) + ")")
        cursor += count
        boundaries.append(cursor - 0.5)

    for edge in boundaries[:-1]:
        axis.axhline(edge, color="black", linewidth=0.8)
        axis.axvline(edge, color="black", linewidth=0.8)

    axis.set_xticks(centres)
    axis.set_xticklabels(labels)
    axis.set_yticks(centres)
    axis.set_yticklabels(labels)
    axis.set_title(
        "Feature correlation matrix ("
        + str(len(FEATURE_NAMES))
        + " features, registry order)"
    )
    bar = fig.colorbar(image, ax=axis, shrink=0.85)
    bar.set_label("Pearson r")

    off_diagonal = values[~np.eye(values.shape[0], dtype=bool)]
    finite = off_diagonal[np.isfinite(off_diagonal)]
    strong = int((np.abs(finite) > 0.9).sum() // 2)
    return save_figure(
        fig,
        path,
        source=(
            "FE-10 | source: feature_correlation_matrix.csv | Pearson, pairwise\n"
            + str(strong)
            + " feature pairs correlate above |r| = 0.9; diagonal blocks are\n"
            "within-family. Correlated features are NOT dropped here -- selection\n"
            "happens inside the training fold (research rule 2)."
        ),
    )


# ---------------------------------------------------------------------------
# T41.4 -- reproducibility
# ---------------------------------------------------------------------------


def reproducibility_check(
    matrix: Any, n_records: int = 50, *, seed: int = 42
) -> dict[str, Any]:
    """Re-extract a random sample and compare bit-for-bit against the cache.

    This calls the batch runner's own worker, ``_extract_one``, rather than
    re-assembling the preprocess-then-extract steps here. A private
    reimplementation would be free to drift from the code that actually built
    the shards -- and then the check would prove the reimplementation
    reproducible, which is not the claim being made.

    Comparison is exact, not ``approx``. Rule 5 says two runs of the same
    command produce identical numbers, and a tolerance here would hide exactly
    the drift -- an unseeded window, a dict iteration order, a library default --
    that the rule exists to catch. NaN is matched against NaN deliberately:
    ``nan != nan``, so an unexplained NaN would otherwise register as a
    mismatch on every re-run of an already-flagged record.
    """
    from src.feature_extraction.batch import _extract_one
    from src.utils.config import load_config

    sample = matrix.sample(n=min(n_records, len(matrix)), random_state=seed)
    root = str(Path(load_config("paths").require("project_root")))

    mismatches: list[dict[str, Any]] = []
    for row in sample.itertuples(index=False):
        uid = str(row.record_uid)
        fresh = _extract_one(uid, str(row.file_path), str(row.dataset_source), root)
        for name in FEATURE_NAMES:
            cached = float(getattr(row, name))
            new_value = float(fresh[name])
            if np.isnan(cached) and np.isnan(new_value):
                continue
            if cached != new_value:
                mismatches.append(
                    {
                        "record_uid": uid,
                        "feature": name,
                        "cached": cached,
                        "reextracted": new_value,
                    }
                )

    return {
        "n_checked": len(sample),
        "n_values": int(len(sample) * len(FEATURE_NAMES)),
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:50],
        "identical": not mismatches,
    }


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------


def write_quality_artifacts(
    matrix: Any, out_dir: str | Path | None = None
) -> dict[str, Path]:
    """Write FE-04, FE-10 and the three supporting tables."""
    directory = features_dir(out_dir)
    written: dict[str, Path] = {}

    tables = {
        "FE-04": (QA_ARTIFACTS["FE-04"], missing_value_report(matrix)),
        "variance": (EXTRA_ARTIFACTS["variance"], variance_report(matrix)),
        "outliers": (EXTRA_ARTIFACTS["outliers"], outlier_report(matrix)),
        "domain_shift": (EXTRA_ARTIFACTS["domain_shift"], domain_shift_report(matrix)),
    }
    for key, (filename, frame) in tables.items():
        path = directory / filename
        frame.to_csv(path, index=False)
        written[key] = path

    corr = correlation_matrix(matrix)
    corr_csv = directory / EXTRA_ARTIFACTS["correlation_csv"]
    corr.to_csv(corr_csv)
    written["correlation_csv"] = corr_csv
    written["FE-10"] = plot_correlation_matrix(directory / QA_ARTIFACTS["FE-10"], corr)

    policy_path = directory / "feature_clipping_policy.txt"
    policy_path.write_text(CLIPPING_POLICY + "\n", encoding="utf-8")
    written["policy"] = policy_path

    for key, path in written.items():
        log.info("%s -> %s", key, path)
    return written
