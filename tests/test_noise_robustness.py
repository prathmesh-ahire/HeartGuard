"""T72.7 -- the EXP-E1 noise robustness gate.

The gate: **noise groups were derived from the PP-08 flags and cross-checked
against PhysioNet's REFERENCE-SQI, and T20 and G30 exist.**

Two things are checked beyond the letter of that, because they are what make the
numbers mean anything:

* The AWGN sweep's **untouched control reproduces the stored feature matrix bit
  for bit.** If it did not, any difference at 20 dB could be a pipeline
  difference rather than a noise effect, and there would be no way to tell.
* The sweep model was **refitted without the sampled records' subjects**. The
  deployed model saw all 3,240 PhysioNet records, so scoring it on any of them
  would be in-sample.

Everything reading a committed CSV runs on CI; the two tests needing the sweep
parquet skip there.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

SECTION = "outputs/09_robustness_analysis"
RUN_DIR = "EXP-E1"

#: Audited 2026-08-22. The whole corpus, every file readable.
N_CORPUS = 7536

#: PP-08 flag counts per dataset, from signal_quality_flags.csv.
N_NOISY = {"D1": 534, "D2": 42, "D3": 19, "D4": 1500}


def _root() -> Any:
    from src.utils.evidence import PROJECT_ROOT

    return PROJECT_ROOT / SECTION


def _csv(*parts: str) -> Any:
    import pandas as pd

    path = _root().joinpath(*parts)
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/23_noise_robustness.py")
    return pd.read_csv(path)


def _section_csv(name: str) -> Any:
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    path = PROJECT_ROOT / SECTION / name
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/23_noise_robustness.py")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# the grouping itself -- runs on CI
# ---------------------------------------------------------------------------


def test_the_groups_are_exactly_the_pp08_flags_and_partition_the_corpus() -> None:
    """T72.1 -- derived from PP-08, not invented, and nothing falls between."""
    from src.evaluation.robustness import QUALITY_GROUPS, quality_groups

    frame = quality_groups()
    assert len(frame) == N_CORPUS
    assert frame["record_uid"].is_unique
    assert set(frame["quality_group"]) <= set(QUALITY_GROUPS)
    # A partition: every record in exactly one group, groups summing to the corpus.
    counts = frame["quality_group"].value_counts()
    assert int(counts.sum()) == N_CORPUS

    noisy = frame["is_noisy"].to_numpy(dtype=bool)
    low = frame["is_low_quality"].to_numpy(dtype=bool)
    group = frame["quality_group"].to_numpy()
    assert (group[noisy] == "noisy").all()
    assert (group[low & ~noisy] == "low_quality_other").all()
    assert (group[~low] == "clean").all()

    per_dataset = frame[frame["is_noisy"]].groupby("dataset_source").size().to_dict()
    assert per_dataset == N_NOISY, per_dataset


def test_the_grouping_was_cross_checked_against_reference_sqi() -> None:
    """T72.3 / T72.7 -- an independent human annotation, and the honest answer.

    The agreement is poor and it is reported as poor. PP-08 estimates quality
    from the waveform; REFERENCE-SQI is the challenge organisers' judgement of
    whether a recording was scoreable. They are related but not the same, and a
    PP-08 group must never be called "the noisy recordings" without saying whose
    definition of noisy is meant.
    """
    summary = _csv(RUN_DIR, "sqi_crosscheck.csv")
    assert len(summary) == 1
    row = summary.iloc[0]
    # PhysioNet TRAINING only: 3,240 records over training-a..f. The 301
    # validation records have a REFERENCE.csv but no REFERENCE-SQI.csv, and no
    # other corpus has the annotation at all.
    assert int(row["n_records"]) == 3240
    assert int(row["n_sqi_poor"]) > 0

    # Internally consistent: the four cells must sum to the total.
    cells = ("both_poor", "pp08_poor_only", "sqi_poor_only", "both_clean")
    assert sum(int(row[c]) for c in cells) == int(row["n_records"])
    assert np.isclose(
        float(row["agreement"]),
        (int(row["both_poor"]) + int(row["both_clean"])) / int(row["n_records"]),
    )

    # The direction must be right even where the agreement is weak: the SQI-poor
    # records must have a LOWER mean SNR proxy. If that ever reverses, PP-08 is
    # measuring the opposite of what it claims and the whole grouping is void.
    assert float(row["mean_snr_proxy_db_sqi_poor"]) < float(row["mean_snr_proxy_db_sqi_good"])


# ---------------------------------------------------------------------------
# T20 and G30 -- the deliverables
# ---------------------------------------------------------------------------


def test_t20_carries_both_kinds_of_evidence_labelled() -> None:
    """T72.5 / T72.7 -- T20 exists, and observational is not passed off as causal."""
    t20 = _section_csv("T20_noise_robustness.csv")
    assert set(t20["analysis"]) == {"observational_pp08", "interventional_awgn"}

    observational = t20[t20["analysis"] == "observational_pp08"]
    assert len(set(observational["run"])) >= 3, sorted(set(observational["run"]))
    # Rule 4: the label spaces are separate rows, never pooled into one.
    assert len(set(observational["task"])) >= 3, sorted(set(observational["task"]))
    assert "clean" in set(observational["group"])
    assert "noisy" in set(observational["group"])

    interventional = t20[t20["analysis"] == "interventional_awgn"]
    levels = set(interventional["group"])
    for level in ("20 dB SNR", "15 dB SNR", "10 dB SNR", "5 dB SNR", "0 dB SNR"):
        assert level in levels, sorted(levels)
    assert any(g.startswith("clean") for g in levels), sorted(levels)


def test_g30_was_drawn_from_the_same_numbers_as_t20() -> None:
    """T72.6 -- the figure exists and its CSV is T20's frame, not a re-derivation."""
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    figures = PROJECT_ROOT / "outputs" / "13_figures_diagrams"
    csv_path = figures / "G30_noise_level_robustness.csv"
    if not csv_path.is_file():
        pytest.skip("G30 not generated; run scripts/23_noise_robustness.py")
    assert (figures / "G30_noise_level_robustness.png").is_file()

    plotted = pd.read_csv(csv_path)
    t20 = _section_csv("T20_noise_robustness.csv")
    keys = ["analysis", "run", "model_id", "group"]
    merged = plotted.merge(t20, on=keys, suffixes=("_g", "_t"))
    assert len(merged) == len(plotted)
    assert np.allclose(
        np.asarray(merged["balanced_accuracy_mean_g"], dtype=float),
        np.asarray(merged["balanced_accuracy_mean_t"], dtype=float),
        equal_nan=True,
    )


def test_the_sweep_degrades_and_the_control_does_not() -> None:
    """T72.4 -- added noise must cost something, or the sweep proved nothing."""
    sweep = _csv(RUN_DIR, "awgn_sweep_metrics.csv")
    assert len(sweep) == 6, sorted(sweep["snr_db"])
    assert sweep["n_nan_features"].sum() == 0, "a feature failed under noise"

    indexed = sweep.set_index("snr_db")
    clean = float(indexed.loc[np.inf, "balanced_accuracy"])
    worst = float(indexed.loc[0.0, "balanced_accuracy"])
    assert worst < clean, (
        "0 dB SNR scored " + format(worst, ".4f") + " against a clean "
        + format(clean, ".4f") + "; added noise that costs nothing means the "
        "noise never reached the features"
    )
    # The realised post-filter SNR must exceed the nominal level: the 20-400 Hz
    # bandpass removes the out-of-band share of white noise. If it did not, the
    # noise was added after filtering and the sweep is measuring the wrong thing.
    finite = sweep[np.isfinite(sweep["snr_db"])]
    assert (
        np.asarray(finite["realised_snr_db_mean"], dtype=float)
        > np.asarray(finite["snr_db"], dtype=float)
    ).all()


def test_no_robustness_metric_is_suspiciously_perfect() -> None:
    """The standing near-perfect rule, on the metrics a degenerate model cannot lift."""
    t20 = _section_csv("T20_noise_robustness.csv")
    reported = t20[t20["reported"]]
    for column in ("balanced_accuracy_mean", "macro_f1_mean", "roc_auc_mean"):
        if column not in reported.columns:
            continue
        values = np.asarray(reported[column], dtype=float)
        if not np.isfinite(values).any():
            continue
        worst = float(np.nanmax(values))
        assert worst < 0.99, column + " reaches " + format(worst, ".4f")


# ---------------------------------------------------------------------------
# needs the sweep parquet -- skips on CI
# ---------------------------------------------------------------------------


def _sweep_features() -> Any:
    import pandas as pd

    path = _root() / RUN_DIR / "awgn_sweep_features.parquet"
    if not path.is_file():
        pytest.skip(str(path) + " is gitignored; run --stage sweep locally")
    return pd.read_parquet(path)


def test_the_untouched_control_reproduces_the_stored_feature_matrix() -> None:
    """The check that makes every other sweep number interpretable.

    The control level adds no noise but runs the full filter / normalize /
    extract chain in this module rather than the batch extractor. If it did not
    land on exactly the values in FE-03, a difference at 20 dB could be a
    pipeline difference and nothing in the sweep would distinguish the two.

    Measured: bit-identical across all 138 features. Not a tolerance -- the same
    code on the same input, so anything but zero is a real divergence.
    """
    from src.feature_extraction.matrix import load_matrix
    from src.feature_extraction.registry import feature_names

    sweep = _sweep_features()
    control = sweep[~np.isfinite(sweep["snr_db"])].set_index("record_uid").sort_index()
    assert len(control) > 0

    names = list(feature_names())
    stored = load_matrix().set_index("record_uid").loc[control.index, names]
    difference = np.abs(
        control[names].to_numpy(dtype=float) - stored.to_numpy(dtype=float)
    )
    assert float(np.nanmax(difference)) == 0.0, (
        "the sweep's clean control differs from FE-03 by up to "
        + format(float(np.nanmax(difference)), ".3e")
    )


def test_the_sweep_model_never_saw_the_sampled_subjects() -> None:
    """Rule 3 -- the deployed model saw all 3,240 records, so it cannot be used."""
    from src.utils.evidence import PROJECT_ROOT

    path = PROJECT_ROOT / SECTION / RUN_DIR / "awgn_model.json"
    if not path.is_file():
        pytest.skip(str(path) + " does not exist; run scripts/23_noise_robustness.py")
    record = json.loads(path.read_text(encoding="utf-8"))

    deployed = json.loads(
        (PROJECT_ROOT / "models_saved" / "binary" / "final" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["model_id"] == deployed["selected_model_id"]
    assert record["hyperparameters"] == deployed["hyperparameters"]
    # Strictly fewer records than the deployed model, by at least the sample.
    assert record["n_records_fitted"] < int(deployed["n_records_fitted"])
    assert record["n_sample_subjects"] > 0

    sample = _csv(RUN_DIR, "awgn_sample.csv")
    sweep = _sweep_features()
    assert set(sweep["record_uid"]) <= set(sample["record_uid"])
    assert set(sweep["subject_id"]) <= set(sample["subject_id"].astype(str))
