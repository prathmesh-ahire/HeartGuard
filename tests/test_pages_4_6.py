"""Phase 115's gate: is every number on pages 4-6 traceable to a source CSV?

Same two directions as `tests/test_pages_1_3.py`, against the result pages.

**Source to screen.** Each metric cell is recomputed from
`aggregate_metrics.csv` through `tables.format_value` and asserted to appear in
the exported HTML. Each feature-family count is recomputed by grouping
`feature_inventory.csv`. Each search-run frame cell is recomputed from its own
CSV under `outputs/05_search_optimization/`.

**Screen to source.** Everything metric-shaped in the rendered text must come
from a generated payload, so a number typed into a page fails here.

Two things this file checks that the pages-1-3 gate does not, because they are
specific to results:

* **The registry order.** `feature_inventory.csv` is emitted in the locked
  column order of the 138-vector. note.md records T05 coming out alphabetical
  once, with every count correct. The features page must present it in `index`
  order, so the first and last feature names on the page are asserted against
  the registry's own first and last.
* **The curve aggregation claim.** `curves.py` states that the area under the
  mean curve is not the reported AUC and that the folds are not pooled. That
  sentence has to reach the page, because a reader integrating the drawn curve
  and comparing it against the AUC column would otherwise find a disagreement
  with no explanation.

Skips when `frontend/out/` is absent. Runs in the frontend CI job, which is the
only job that builds it.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "frontend" / "out"
GENERATED = PROJECT_ROOT / "frontend" / "lib" / "generated"
OUTPUTS = PROJECT_ROOT / "outputs"

PAGES = {
    "/features/": "features/index.html",
    "/models/": "models/index.html",
    "/optimization/": "optimization/index.html",
}

METRIC_LIKE = re.compile(r"\b\d+\.\d{3,}\b|\b\d+(?:\.\d+)?%")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)?")

#: Task references printed in an eyebrow. Listed individually; a pattern wide
#: enough to cover them would be wide enough to hide a metric.
STRUCTURAL: frozenset[str] = frozenset({"115.1", "115.2", "115.3", "115.4", "115.5"})


pytestmark = pytest.mark.skipif(
    not all((OUT / name).is_file() for name in PAGES.values()),
    reason="frontend/out/ is not built in this checkout; run npm run build",
)


def _text(page: str) -> str:
    raw = (OUT / PAGES[page]).read_text(encoding="utf-8")
    raw = re.sub(r"<script.*?</script>", " ", raw, flags=re.S)
    raw = re.sub(r"<style.*?</style>", " ", raw, flags=re.S)
    return html.unescape(re.sub(r"<[^>]+>", " ", raw))


def _generated(name: str) -> Any:
    return json.loads((GENERATED / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# T115.1 / T115.2 -- features
# ---------------------------------------------------------------------------


def test_the_family_counts_are_recomputed_from_the_inventory() -> None:
    import pandas as pd

    from src.reporting.tables import format_value

    inventory = pd.read_csv(OUTPUTS / "03_features" / "feature_inventory.csv")
    payload = _generated("features.json")
    text = _text("/features/")

    counted = inventory.groupby("family", sort=False).size().to_dict()
    assert sum(counted.values()) == 138, "the registry is not 138 features"
    assert payload["n_features"] == 138

    for entry in payload["families"]:
        expected = int(counted[entry["family"]])
        assert entry["n_features"] == expected, entry["family"]
        assert entry["n_features_display"] == format_value(expected, "count")
        assert entry["n_features_display"] in text, (
            entry["family"] + "'s count is not on the features page"
        )


def test_the_features_page_presents_the_registry_in_its_locked_order() -> None:
    """Not alphabetical, and not the exporter's whim: `index` order."""
    import pandas as pd

    inventory = pd.read_csv(OUTPUTS / "03_features" / "feature_inventory.csv")
    ordered = inventory.sort_values("index")
    payload = _generated("features.json")

    assert [item["name"] for item in payload["features"]] == [str(name) for name in ordered["name"]]
    assert [item["index"] for item in payload["features"]] == list(range(138))

    families = [entry["family"] for entry in payload["families"]]
    assert families != sorted(families), (
        "the family order is alphabetical; the registry order is time, frequency, "
        "mfcc, chroma, dwt, envelope and it is load-bearing"
    )

    text = _text("/features/")
    assert payload["features"][0]["name"] in text
    assert payload["features"][-1]["name"] in text


def test_the_example_vector_is_one_real_record_at_full_width() -> None:
    payload = _generated("features.json")["example_vector"]
    if not payload["available"]:
        pytest.skip("no example feature vector was exported: " + str(payload["reason"]))
    assert payload["n_features"] == 138
    assert payload["record_uid"].startswith("D1_")
    assert len(payload["values"]) == 138
    assert [item["index"] for item in payload["values"]] == list(range(138))


def test_the_selected_subset_reports_how_many_folds_kept_each_feature() -> None:
    """ "Selected" is not one claim: 3 of 25 folds and 25 of 25 are different."""
    payload = _generated("features.json")["selected"]
    if not payload["available"]:
        pytest.skip(str(payload["reason"]))

    text = _text("/features/")
    for row in payload["features"]:
        assert 0 < int(row["selected_in_folds"]) <= int(row["n_folds"])
    assert "of " + str(payload["features"][0]["n_folds"]) + " folds" in " ".join(text.split())


# ---------------------------------------------------------------------------
# T115.3 / T115.4 -- model comparison
# ---------------------------------------------------------------------------


def test_every_metric_cell_equals_its_aggregate_metrics_cell() -> None:
    import pandas as pd

    from src.reporting.tables import format_value

    payload = _generated("experiments.json")
    text = _text("/models/")
    checked = 0

    for experiment in payload["experiments"]:
        if not experiment["available"]:
            continue
        frame = pd.read_csv(PROJECT_ROOT / experiment["directory"] / "aggregate_metrics.csv")
        by_model = frame.set_index("model_id")
        for model in experiment["models"]:
            row = by_model.loc[model["model_id"]]
            for name, value in model["metrics"].items():
                expected_mean = format_value(row[name + "_mean"], "metric")
                expected_sd = format_value(row[name + "_sd"], "metric")
                assert value["mean_display"] == expected_mean, (
                    experiment["exp_id"] + " " + model["model_id"] + " " + name
                )
                assert value["display"] == expected_mean + " +/- " + expected_sd
                checked += 1

    assert checked > 200, "too few cells checked to mean anything: " + str(checked)
    # The selected experiment's own table is what renders; spot-check that the
    # first experiment's cells are actually present in the served HTML.
    first = next(item for item in payload["experiments"] if item["available"])
    for model in first["models"]:
        for value in model["metrics"].values():
            assert value["display"] in text, (
                first["exp_id"] + " shows " + value["display"] + " nowhere on /models/"
            )


def test_the_confusion_matrix_counts_are_the_committed_ones() -> None:
    payload = _generated("experiments.json")
    for experiment in payload["experiments"]:
        if not experiment["available"] or not experiment["confusion"]["available"]:
            continue
        source = json.loads(
            (PROJECT_ROOT / experiment["directory"] / "confusion_matrices.json").read_text(
                encoding="utf-8"
            )
        )
        for model_id, block in experiment["confusion"]["models"].items():
            assert block["total"] == source["models"][model_id]["total"], (
                experiment["exp_id"] + " " + model_id
            )
        assert experiment["confusion"]["note"], "the support note is missing"


def test_the_page_states_that_the_confusion_support_is_not_the_corpus() -> None:
    """Four cells summing to five times the corpus needs its reason on the page."""
    text = " ".join(_text("/models/").split())
    assert "element-wise sum over folds" in text or "once per repeat" in text


def test_the_page_states_how_the_curves_were_aggregated() -> None:
    payload = _generated("experiments.json")
    curved = [
        item for item in payload["experiments"] if item["available"] and item["curves"]["available"]
    ]
    if not curved:
        pytest.skip("no experiment exported curve points")

    text = " ".join(_text("/models/").split())
    note = " ".join(curved[0]["curves"]["aggregation_note"].split())
    assert note in text, "the curve aggregation note is not rendered"
    assert "not the reported AUC" in note


def test_an_experiment_that_has_not_run_is_listed_with_its_reason() -> None:
    payload = _generated("experiments.json")
    assert payload["n_declared"] >= payload["n_available"]
    text = _text("/models/")
    for experiment in payload["experiments"]:
        if experiment["available"]:
            continue
        assert experiment["exp_id"] in text
        assert experiment["reason"] in " ".join(text.split())


def test_the_metric_order_puts_sensitivity_before_accuracy() -> None:
    payload = _generated("experiments.json")
    for experiment in payload["experiments"]:
        if not experiment["available"]:
            continue
        names = [item["name"] for item in experiment["metrics"]]
        if "sensitivity" in names and "accuracy" in names:
            assert names.index("sensitivity") < names.index("accuracy"), experiment["exp_id"]
    for item in payload["experiments"]:
        for metric in item.get("metrics") or []:
            if metric["name"] in ("brier", "ece"):
                assert metric["higher_is_better"] is False, (
                    metric["name"] + " is marked higher-is-better; sorting it "
                    "descending would present the worst-calibrated model as the best"
                )


# ---------------------------------------------------------------------------
# T115.5 -- search optimization
# ---------------------------------------------------------------------------


SEARCH_FRAMES = {
    "pareto": "outputs/05_search_optimization/SO-06/pareto_front.csv",
    "weight_stability": "outputs/05_search_optimization/SO-05/weight_stability.csv",
    "equal_vs_optimized": "outputs/05_search_optimization/SO-05/equal_vs_optimized.csv",
    "feature_count_curve": "outputs/05_search_optimization/SO-04/feature_count_curve.csv",
    "method_comparison": "outputs/05_search_optimization/search_method_comparison.csv",
}


@pytest.mark.parametrize(("key", "relative"), sorted(SEARCH_FRAMES.items()))
def test_every_search_frame_cell_equals_its_source_csv_cell(key: str, relative: str) -> None:
    import pandas as pd

    from src.reporting.tables import format_value, infer_kind

    payload = _generated("optimization.json")[key]
    source = PROJECT_ROOT / relative
    if not source.is_file():
        assert payload["available"] is False
        pytest.skip(relative + " has not been produced")

    frame = pd.read_csv(source)
    assert payload["available"] is True
    assert payload["n_rows"] == len(frame)

    for column in payload["columns"]:
        name = column["name"]
        kind = infer_kind(name, frame[name])
        assert column["kind"] == kind
        for position, shown in enumerate(column["display"]):
            assert shown == format_value(frame[name].iloc[position], kind), (
                key + "." + name + " row " + str(position)
            )


def test_the_optimization_page_leads_with_fold_safety() -> None:
    payload = _generated("optimization.json")
    text = " ".join(_text("/optimization/").split())
    assert " ".join(payload["fold_safety_note"].split()) in text
    assert "never seen by a search" in text


def test_every_declared_search_run_appears_with_its_state() -> None:
    payload = _generated("optimization.json")
    text = _text("/optimization/")
    assert payload["n_runs"] == len(payload["runs"])
    for run in payload["runs"]:
        assert run["run_id"] in text, run["run_id"] + " is not on the page"
        if run["convergence"] is not None and not run["convergence"]["available"]:
            assert run["convergence"]["reason"], run["run_id"] + " has no stated reason"


# ---------------------------------------------------------------------------
# screen -> source
# ---------------------------------------------------------------------------


def _allowed() -> set[str]:
    allowed: set[str] = set(STRUCTURAL)

    def admit(value: str) -> None:
        allowed.add(value)
        allowed.update(METRIC_LIKE.findall(value))

    for name in ("tables.json", "figures.json"):
        for entry in _generated(name).values():
            for column in entry.get("columns", []):
                for value in column.get("display", []):
                    admit(str(value))

    experiments = _generated("experiments.json")
    for experiment in experiments["experiments"]:
        for model in experiment.get("models") or []:
            admit(model["n_folds_display"])
            for value in model["metrics"].values():
                admit(value["display"])
                admit(value["mean_display"])
                admit(value["sd_display"])
            for row in model.get("per_class") or []:
                for key, value in row.items():
                    if isinstance(value, dict) and "display" in value:
                        admit(str(value["display"]))
                    elif key == "support_display":
                        admit(str(value))

    features = _generated("features.json")
    for entry in features["families"]:
        admit(entry["n_features_display"])
    for entry in features["features"]:
        admit(entry["abs_cohens_d_display"])
    for entry in features["example_vector"]["values"]:
        admit(entry["display"])
    for row in features["selected"].get("features", []):
        admit(str(row["share_display"]))

    optimization = _generated("optimization.json")
    for key in SEARCH_FRAMES:
        for column in optimization[key].get("columns", []):
            for value in column["display"]:
                admit(str(value))

    return allowed


@pytest.mark.parametrize("page", sorted(PAGES))
def test_nothing_that_looks_like_a_metric_is_unaccounted_for(page: str) -> None:
    allowed = _allowed()
    text = _TIMESTAMP.sub(" ", _text(page))
    unexplained = sorted(
        {
            token
            for token in METRIC_LIKE.findall(text)
            if token not in allowed and token.rstrip("%") not in allowed
        }
    )
    assert unexplained == [], (
        page + " states values with no source in generated/: " + ", ".join(unexplained)
    )


@pytest.mark.parametrize("page", sorted(PAGES))
def test_the_page_uses_screening_language_and_the_framework_name(page: str) -> None:
    text = _text(page)
    assert "PV-MEPCG" in text or "PulseVision" in text
    assert "HeartGuard" not in text
    lowered = text.lower()
    assert "not a diagnostic device" in lowered or "does not diagnose" in lowered
    for forbidden in ("diagnosis of", "treatment plan", "replaces a doctor"):
        assert forbidden not in lowered, "clinical language on " + page + ": " + forbidden
