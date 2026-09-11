"""The T92.7, T93.7 and T94.7 gates: G11-G35 exist, and each reconciles with its source.

T92.7 -- G11-G19 exist, the ROC legend AUCs match T08, and every confusion
matrix uses the fixed class ordering.
T93.7 -- G20-G28 exist, and the feature-count curve matches the T57.4 sweep.
T94.7 -- G29-G35 exist, and the robustness charts reconcile with T20-T22.

Every expected value is re-derived from an upstream results file, never typed
in: a literal in a test is a hand-typed number under rule 1 exactly as much as
a literal in a figure caption. Where possible the check goes through a SECOND
code path -- G11 against the per-fold files rather than T08 alone, G17's
recomputed OvR AUCs against the experiment's own macro AUC, G22 against the raw
sweep rather than the curve summarising it -- because a figure agreeing with
its own input proves nothing.

Everything read here is committed, so the gate runs on CI; a checkout where the
figures were never generated reports skipped, not passed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.reporting import graphs as gr
from src.reporting.result_graphs import (
    G11_METRICS,
    T08_FILE,
    T11_FILE,
    T12_FILE,
    T13_FILE,
    T14_FILE,
    T16_FILE,
    TS5_FILE,
    class_order,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
BINARY = OUT / "06_binary_results"
MULTICLASS = OUT / "07_multiclass_results"
CIRCOR = OUT / "08_circor_external_validation"
SEARCH = OUT / "05_search_optimization"

PHASE_92 = tuple("G" + str(n) for n in range(11, 20))
PHASE_93 = tuple("G" + str(n) for n in range(20, 29))
PHASE_94 = tuple("G" + str(n) for n in range(29, 36))
ALL_G = tuple("G" + format(n, "02d") for n in range(1, 36))

TOL = 1e-9


@pytest.fixture(scope="module")
def registry() -> dict[str, dict[str, str]]:
    rows = {row["figure_id"]: row for row in gr.read_registry()}
    if "G11" not in rows:
        pytest.skip("G11-G35 not generated here (run scripts/38_result_graphs.py)")
    return rows


def _stem(registry: dict[str, dict[str, str]], figure_id: str) -> Path:
    return gr.figures_dir() / Path(registry[figure_id]["filename"]).stem


def _plotted(registry: dict[str, dict[str, str]], figure_id: str) -> Any:
    return pd.read_csv(gr.figures_dir() / registry[figure_id]["source_csv"])


def _da02(task: str) -> dict[str, int]:
    frame = pd.read_csv(OUT / "01_dataset_audit" / "class_distribution.csv")
    frame = frame[(frame["scope"] == "supervised") & (frame["task"] == task)]
    return {str(k): int(v) for k, v in zip(frame["class"], frame["n_records"], strict=True)}


# ---------------------------------------------------------------------------
# existence, provenance and numbering -- all three gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("figure_id", PHASE_92 + PHASE_93 + PHASE_94)
def test_figure_exists_with_png_plotted_csv_and_meta(
    registry: dict[str, dict[str, str]], figure_id: str
) -> None:
    from src.reporting.tables import content_digest

    assert figure_id in registry, figure_id + " is not in figure_registry.csv"
    stem = _stem(registry, figure_id)
    png = stem.with_suffix(".png")
    csv = gr.figures_dir() / registry[figure_id]["source_csv"]
    meta_path = Path(str(stem) + ".meta.json")
    for path in (png, csv, meta_path):
        assert path.is_file() and path.stat().st_size > 0, figure_id + " lacks " + path.name
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert png.stat().st_size > 20_000, figure_id + " PNG is too small for a 300 dpi figure"
    assert len(pd.read_csv(csv)) > 0

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["figure_id"] == figure_id
    assert meta["dpi"] == 300
    assert meta["framework"] == "PV-MEPCG / PulseVision"
    assert meta["sources"], figure_id + " records no upstream source"
    assert meta["plotted_csv_sha256"] == content_digest(csv)[0], (
        figure_id + " was drawn from a different CSV than the one on disk"
    )
    assert registry[figure_id]["caption"].strip()


def test_every_g_figure_holds_its_own_index_as_its_number(
    registry: dict[str, dict[str, str]],
) -> None:
    """G32 was figure 11 until the Phase 94 renumbering. It must stay 32."""
    for figure_id in ALL_G:
        assert figure_id in registry, figure_id + " missing from the registry"
        assert int(registry[figure_id]["figure_number"]) == int(figure_id[1:]), figure_id
        meta = json.loads(
            Path(str(_stem(registry, figure_id)) + ".meta.json").read_text(encoding="utf-8")
        )
        assert int(meta["figure_number"]) == int(figure_id[1:]), figure_id + " meta disagrees"


def test_no_figure_provenance_records_a_machine_path(
    registry: dict[str, dict[str, str]],
) -> None:
    """An absolute D:/ path resolves on one machine and fails every CI re-check."""
    for figure_id, row in registry.items():
        for source in filter(None, (s.strip() for s in row["upstream_sources"].split(";"))):
            assert not Path(source).is_absolute() and ":" not in source, (
                figure_id + " records " + source
            )
        meta = json.loads(
            Path(str(_stem(registry, figure_id)) + ".meta.json").read_text(encoding="utf-8")
        )
        paths = [s.get("path", "") for s in meta.get("sources") or []]
        paths += list((meta.get("written") or {}).values())
        for path in paths:
            assert ":" not in str(path), figure_id + " meta records " + str(path)


def test_there_is_one_figure_registry_and_g23_g24_live_in_it() -> None:
    assert not (OUT / "09_ablation" / "figure_registry.csv").exists(), (
        "a second registry numbers G23/G24 in a series of their own"
    )
    assert not list((OUT / "09_ablation").glob("G2[34]_*"))


# ---------------------------------------------------------------------------
# T92.7 -- G11 and the ROC legend against T08
# ---------------------------------------------------------------------------


def test_g11_bars_and_whiskers_are_t08_and_the_per_fold_files(
    registry: dict[str, dict[str, str]],
) -> None:
    plotted = _plotted(registry, "G11")
    t08 = pd.read_csv(BINARY / T08_FILE)
    assert len(plotted) == len(t08) * len(G11_METRICS)
    for row in plotted.to_dict("records"):
        source = t08[(t08["run"] == row["run"]) & (t08["model_id"] == row["model_id"])].iloc[0]
        assert row["mean"] == pytest.approx(source[row["metric"] + "_mean"], abs=TOL)
        assert row["sd"] == pytest.approx(source[row["metric"] + "_sd"], abs=TOL)
        # The second code path: the 25 per-fold values themselves.
        folds = pd.read_csv(BINARY / row["run"] / "per_fold_metrics.csv")
        values = folds[folds["model_id"] == row["model_id"]][row["metric"]]
        assert len(values) == row["n_folds"]
        assert row["mean"] == pytest.approx(values.mean(), abs=TOL)
        assert row["sd"] == pytest.approx(values.std(ddof=1), abs=TOL)


def test_g12_legend_auc_is_t08_roc_auc_for_every_run_and_model(
    registry: dict[str, dict[str, str]],
) -> None:
    plotted = _plotted(registry, "G12")
    t08 = pd.read_csv(BINARY / T08_FILE)
    pairs = plotted[["run", "model_id"]].drop_duplicates()
    assert len(pairs) == len(t08), "G12 does not draw every model T08 lists"
    for run, model_id in pairs.itertuples(index=False):
        block = plotted[(plotted["run"] == run) & (plotted["model_id"] == model_id)]
        expected = float(
            t08[(t08["run"] == run) & (t08["model_id"] == model_id)]["roc_auc_mean"].iloc[0]
        )
        assert block["roc_auc_mean"].nunique() == 1
        assert float(block["roc_auc_mean"].iloc[0]) == pytest.approx(expected, abs=TOL)
        # What a reader sees is the legend text: T85.6's three decimals of it.
        assert block["legend"].iloc[0].endswith("AUC " + format(expected, ".3f")), (
            run + " " + model_id + ": legend '" + block["legend"].iloc[0] + "'"
        )


def test_g12_and_g13_curves_are_the_committed_curve_points(
    registry: dict[str, dict[str, str]],
) -> None:
    for figure_id, kind in (("G12", "roc"), ("G13", "pr")):
        plotted = _plotted(registry, figure_id)
        for run in plotted["run"].unique():
            curves = pd.read_csv(BINARY / run / "roc_pr_curve_points.csv")
            curves = curves[curves["kind"] == kind]
            merged = plotted[plotted["run"] == run].merge(
                curves, on=["model_id", "x"], suffixes=("", "_src")
            )
            assert len(merged) == int((plotted["run"] == run).sum())
            assert np.allclose(merged["mean"], merged["mean_src"], atol=TOL)
            assert np.allclose(merged["sd"], merged["sd_src"], atol=TOL)


def test_g13_average_precision_is_the_aggregate_pr_auc(
    registry: dict[str, dict[str, str]],
) -> None:
    plotted = _plotted(registry, "G13")
    for (run, model_id), block in plotted.groupby(["run", "model_id"]):
        aggregate = pd.read_csv(BINARY / str(run) / "aggregate_metrics.csv").set_index("model_id")
        expected = float(aggregate.loc[model_id, "pr_auc_mean"])
        assert float(block["pr_auc_mean"].iloc[0]) == pytest.approx(expected, abs=TOL)
        prevalence = float(aggregate.loc[model_id, "n_positive_mean"]) / float(
            aggregate.loc[model_id, "support_mean"]
        )
        assert float(block["prevalence"].iloc[0]) == pytest.approx(prevalence, abs=TOL)


# ---------------------------------------------------------------------------
# T92.7 -- confusion matrices in the fixed class ordering
# ---------------------------------------------------------------------------

MATRICES = (
    ("G14", "binary", BINARY / "EXP-A2"),
    ("G15", "pascal_a", MULTICLASS / "EXP-B1"),
    ("G16", "pascal_b", MULTICLASS / "EXP-B2"),
)


@pytest.mark.parametrize(("figure_id", "task", "exp_dir"), MATRICES)
def test_confusion_matrix_uses_the_fixed_class_order(
    registry: dict[str, dict[str, str]], figure_id: str, task: str, exp_dir: Path
) -> None:
    from src.utils.constants import LABEL_MAPS

    plotted = _plotted(registry, figure_id)
    expected = [name for name, _ in sorted(LABEL_MAPS[task].items(), key=lambda i: i[1])]
    assert class_order(task) == expected
    for _model, block in plotted.groupby("model_id"):
        rows = block.sort_values(["true_index", "pred_index"])
        assert list(rows.drop_duplicates("true_index")["true_class"]) == expected
        assert list(rows.drop_duplicates("pred_index").sort_values("pred_index")["pred_class"]) == (
            expected
        )
        for row in rows.to_dict("records"):
            assert expected[int(row["true_index"])] == row["true_class"]
            assert expected[int(row["pred_index"])] == row["pred_class"]


@pytest.mark.parametrize(("figure_id", "task", "exp_dir"), MATRICES)
def test_confusion_matrix_counts_reconcile_with_da02_and_the_run(
    registry: dict[str, dict[str, str]], figure_id: str, task: str, exp_dir: Path
) -> None:
    plotted = _plotted(registry, figure_id)
    payload = json.loads((exp_dir / "confusion_matrices.json").read_text(encoding="utf-8"))
    counts = _da02(task)
    for model_id, block in plotted.groupby("model_id"):
        # Repeat 0 is a complete partition: each record once, rows = class counts.
        row_totals = block.groupby("true_class")["count_repeat0"].sum().to_dict()
        assert row_totals == counts, str(model_id) + " repeat-0 rows are not DA-02"
        total = np.asarray(payload["models"][model_id]["total"])
        for row in block.to_dict("records"):
            assert row["count_all_folds"] == total[row["true_index"], row["pred_index"]]
        shares = block.groupby("true_index")["row_share_all_folds"].sum()
        assert np.allclose(shares, 1.0, atol=TOL)


def test_g14_shows_the_final_model(registry: dict[str, dict[str, str]]) -> None:
    final = pd.read_csv(BINARY / "final_model_selection.csv").sort_values("rank")
    assert str(final["model_id"].iloc[0]) in set(_plotted(registry, "G14")["model_id"])


# ---------------------------------------------------------------------------
# T92.7 -- G17 against the experiment's own macro OvR AUC
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("task", "exp_dir", "table"),
    [("pascal_a", MULTICLASS / "EXP-B1", T11_FILE), ("pascal_b", MULTICLASS / "EXP-B2", T12_FILE)],
)
def test_g17_class_aucs_reproduce_the_experiments_macro_ovr_auc(
    registry: dict[str, dict[str, str]], task: str, exp_dir: Path, table: str
) -> None:
    plotted = _plotted(registry, "G17")
    block = plotted[plotted["task"] == task]
    assert not block.empty
    classes = block.drop_duplicates("class_index").sort_values("class_index")
    assert list(classes["class_name"]) == class_order(task)

    model_id = str(block["model_id"].iloc[0])
    ranked = pd.read_csv(MULTICLASS / table)
    assert model_id == str(ranked.loc[ranked["macro_f1_mean"].idxmax(), "model_id"])
    aggregate = pd.read_csv(exp_dir / "aggregate_metrics.csv").set_index("model_id")
    expected = float(aggregate.loc[model_id, "ovr_auc_macro_mean"])
    assert float(classes["auc_mean"].mean()) == pytest.approx(expected, abs=TOL)
    assert float(block["ovr_auc_macro_mean"].iloc[0]) == pytest.approx(expected, abs=TOL)
    for record in classes.to_dict("records"):
        assert record["legend"].endswith("AUC " + format(record["auc_mean"], ".3f"))


# ---------------------------------------------------------------------------
# T93.7 -- G20-G22 against the searches, G22 against the T57.4 sweep
# ---------------------------------------------------------------------------


def test_g20_keeps_the_two_objectives_apart_and_plots_the_running_best(
    registry: dict[str, dict[str, str]],
) -> None:
    plotted = _plotted(registry, "G20")
    assert set(plotted["family"]) == {"hyperparameter", "feature_mask"}
    for (search, family, model_id), block in plotted.groupby(["search", "family", "model_id"]):
        values = block.sort_values("step")["best_so_far"].to_numpy(float)
        steps = np.diff(values)
        if family == "hyperparameter":
            assert (steps >= -TOL).all(), str(search) + " best-so-far fell while maximising"
        else:
            assert (steps <= TOL).all(), str(search) + " best-so-far rose while minimising"
        source = pd.read_csv(SEARCH / str(search) / "convergence.csv")
        source = source[source["model_id"] == model_id]
        assert np.allclose(np.sort(source["best_so_far"]), np.sort(values), atol=TOL)


def test_g21_is_the_held_out_fold_summary(registry: dict[str, dict[str, str]]) -> None:
    plotted = _plotted(registry, "G21").set_index("configuration")
    folds = pd.read_csv(SEARCH / "SO-04" / "all_features_vs_selected.csv")
    for configuration, block in folds.groupby("configuration"):
        for metric in ("macro_f1", "balanced_accuracy", "sensitivity", "specificity"):
            assert plotted.loc[configuration, metric + "_mean"] == pytest.approx(
                block[metric].mean(), abs=TOL
            )
            assert plotted.loc[configuration, metric + "_se"] == pytest.approx(
                block[metric].sem(), abs=TOL
            )


def test_g22_feature_count_curve_matches_the_t57_4_sweep(
    registry: dict[str, dict[str, str]],
) -> None:
    """T93.7. Checked against the raw sweep, not only the curve built from it."""
    plotted = _plotted(registry, "G22")
    curve = pd.read_csv(SEARCH / "SO-04" / "feature_count_curve.csv")
    sweep = pd.read_csv(SEARCH / "SO-04" / "feature_count_sweep.csv")
    assert len(plotted) == len(curve)
    merged = plotted.merge(curve, on=["ranker", "k"], suffixes=("", "_curve"))
    assert len(merged) == len(curve)
    grouped = sweep.groupby(["ranker", "k"])
    for metric in ("macro_f1", "accuracy", "balanced_accuracy"):
        assert np.allclose(merged[metric + "_mean"], merged[metric + "_mean_curve"], atol=TOL)
        means = grouped[metric].mean()
        stds = grouped[metric].std(ddof=1)
        for row in plotted.to_dict("records"):
            key = (row["ranker"], row["k"])
            assert row[metric + "_mean"] == pytest.approx(means.loc[key], abs=TOL)
            assert row[metric + "_std"] == pytest.approx(stds.loc[key], abs=TOL)
    counts = grouped.size()
    for row in plotted.to_dict("records"):
        assert row["n_evaluations"] == counts.loc[(row["ranker"], row["k"])]
    settings = json.loads((SEARCH / "SO-04" / "so04_settings.json").read_text(encoding="utf-8"))
    assert int(plotted["chosen_k"].iloc[0]) == int(settings["chosen"]["k"])


def test_g28_values_come_from_their_tables(registry: dict[str, dict[str, str]]) -> None:
    plotted = _plotted(registry, "G28")
    inner = plotted[plotted["panel"] == "within_corpus"]
    assert list(dict.fromkeys(inner["task"])) == [
        "binary",
        "pascal_a",
        "pascal_b",
        "circor_murmur",
        "circor_outcome",
    ]
    locations = {
        T08_FILE: BINARY,
        T11_FILE: MULTICLASS,
        T12_FILE: MULTICLASS,
        T13_FILE: CIRCOR,
        T14_FILE: CIRCOR,
    }
    for row in inner.to_dict("records"):
        table = pd.read_csv(locations[row["source_table"]] / row["source_table"])
        if "run" in table:
            run = "EXP-A2" if row["task"] == "binary" else "EXP-C1-three_class"
            table = table[table["run"] == run]
        if "level" in table:
            table = table[table["level"] == "recording"]
        values = table[table["model_id"] == row["model_id"]]["balanced_accuracy_mean"]
        assert len(values) == 1
        assert row["value"] == pytest.approx(float(values.iloc[0]), abs=TOL)

    outer = plotted[plotted["panel"] == "beyond_corpus"]
    ts5 = pd.read_csv(OUT / "09_ablation" / TS5_FILE).set_index("model_id")
    t16 = pd.read_csv(CIRCOR / T16_FILE)
    t16 = t16[(t16["level"] == "recording") & (t16["rule"] == "none")].set_index("metric")
    for row in outer.to_dict("records"):
        metric = row["metric"]
        if "EXP-F3" in row["regime"]:
            expected = ts5.loc[row["model_id"], metric + "_holdout"]
        elif "EXP-D1" in row["regime"]:
            expected = t16.loc[metric, "external_value"]
        elif "EXP-A1" in row["regime"]:
            expected = ts5.loc[row["model_id"], metric + "_pooled"]
        else:
            t08 = pd.read_csv(BINARY / T08_FILE)
            expected = t08[(t08["run"] == "EXP-A2") & (t08["model_id"] == row["model_id"])][
                metric + "_mean"
            ].iloc[0]
        assert row["value"] == pytest.approx(float(expected), abs=TOL), row["regime"]


# ---------------------------------------------------------------------------
# T94.7 -- the robustness charts reconcile with T20, T21 and T22
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("figure_id", "table"),
    [
        ("G30", OUT / "09_robustness_analysis" / "T20_noise_robustness.csv"),
        ("G31", OUT / "09_robustness_analysis" / "T21_duration_robustness.csv"),
        ("G32", CIRCOR / "T22_auscultation_location_analysis.csv"),
    ],
)
def test_robustness_chart_plots_exactly_its_table(
    registry: dict[str, dict[str, str]], figure_id: str, table: Path
) -> None:
    plotted = _plotted(registry, figure_id)
    source = pd.read_csv(table)
    assert list(plotted.columns) == list(source.columns)
    pd.testing.assert_frame_equal(plotted, source, check_exact=False, atol=TOL, rtol=0)


def test_g29_in_domain_and_external_values_are_t16s(
    registry: dict[str, dict[str, str]],
) -> None:
    """G29 is the cross-dataset drop; every value it shares with T16 must agree."""
    plotted = _plotted(registry, "G29")
    t16 = pd.read_csv(CIRCOR / T16_FILE)
    t16 = t16[t16["level"] == "recording"]
    numeric = plotted.select_dtypes(include="number").to_numpy(float).ravel()
    numeric = numeric[np.isfinite(numeric)]
    for value in t16["in_domain_mean"].dropna().unique():
        assert np.isclose(numeric, value, atol=TOL).any(), "G29 lacks in-domain " + str(value)
    for value in t16["external_value"].dropna().unique():
        assert np.isclose(numeric, value, atol=TOL).any(), "G29 lacks external " + str(value)
