"""F11-F20: the pipeline and contribution diagrams (Phase 96).

Same rule as F01-F10: **structure is drawn here, facts are read.** Every count,
class vocabulary, parameter, route and verdict in these ten figures comes from a
file or a module the pipeline itself uses:

* corpus and class counts -- the dataset audit (``dataset_summary_payload`` and
  DA-02 ``class_distribution.csv``);
* protocols, label spaces and emitted tables -- ``configs/experiments.yaml``;
* the aggregation rules, noise levels, clip lengths and duration bands -- the
  constants the evaluation modules run with (``AGGREGATION_RULES``,
  ``SNR_LEVELS``, ``TRUNCATION_SECONDS``, ``configs/signal.yaml``);
* the population mismatch in F14 -- EXP-D1's ``population_mismatch.json``,
  which was written before any metric existed;
* the dashboard routes in F16 -- the FastAPI application's own route table, and
  the exporter's own list of generated files;
* the prediction bounds in F17 -- ``src.inference.predictor``'s constants;
* the objectives, their modules and their verdicts in F18-F20 --
  ``objectives.OBJECTIVES``, ``conclusion_tables.OBJECTIVE_EVIDENCE`` and T30;
* the stage-by-stage deltas in F19 -- T19, formatted here to three places.

A figure that restated any of those would be a second copy that drifts from the
first, and ``tests/test_pipeline_figures.py`` compares the text actually drawn
on each canvas against its source so the drift fails rather than hides.

**Rows are placed from measured heights.** Every box that carries a sentence is
sized by ``fit_height`` first and the next row starts below the tallest box of
the row above (:func:`_row_height`). Fixed row positions were the cause of every
overlap in the first render of these ten, and the canvas now refuses an overlap
outright.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.reporting.diagrams.architecture_figures import _config, _datasets, _features
from src.reporting.diagrams.canvas import DiagramCanvas
from src.reporting.diagrams.catalogue import diagram

__all__ = [
    "build_f11",
    "build_f12",
    "build_f13",
    "build_f14",
    "build_f15",
    "build_f16",
    "build_f17",
    "build_f18",
    "build_f19",
    "build_f20",
]

#: Colour of the "never" annotations, matching the forbidden edge.
_WARN = "#D55E00"

#: A row of boxes: ``(key, label, kind, col, width, sublabel)``.
Row = list[tuple[str, str, str, float, float, str]]


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _outputs(key: str) -> Path:
    return _root() / str(_config("paths").require("outputs." + key))


def _experiment(exp_id: str) -> dict[str, Any]:
    return dict(_config("experiments").require("experiments." + exp_id))


def _scheme(name: str) -> dict[str, Any]:
    return dict(_config("experiments").require("cv_schemes." + name))


def _model_name(model_id: str) -> str:
    return str(_config("models").require("models." + model_id + ".name"))


def _count(value: Any) -> str:
    """An integer count with a thousands separator, formatted once, here."""
    return format(int(value), ",")


def class_counts(task: str) -> list[tuple[str, int]]:
    """``[(class, n_records), ...]`` for one task, from DA-02, largest first."""
    import pandas as pd

    frame = pd.read_csv(_outputs("dataset_audit") / "class_distribution.csv")
    rows = frame[(frame["scope"] == "supervised") & (frame["task"] == task)]
    if rows.empty:
        raise ValueError("class_distribution.csv has no supervised rows for task " + task)
    rows = rows.sort_values("n_records", ascending=False)
    return [(str(r["class"]), int(r["n_records"])) for _, r in rows.iterrows()]


def _counts_text(task: str) -> str:
    return ", ".join(name + " " + _count(n) for name, n in class_counts(task))


def final_binary_model() -> dict[str, Any]:
    """The deployed binary model, from its own manifest."""
    import json

    root = _root() / str(_config("paths").require("models_saved"))
    manifest = json.loads((root / "binary" / "final" / "manifest.json").read_text("utf-8"))
    return dict(manifest)


def population_mismatch() -> dict[str, Any]:
    """EXP-D1's population metadata, written before the transfer was scored."""
    import json

    path = _outputs("circor_external_validation") / "EXP-D1" / "population_mismatch.json"
    return dict(json.loads(path.read_text(encoding="utf-8")))


def api_routes() -> list[tuple[str, str]]:
    """``[(METHOD, path), ...]`` for every route the FastAPI app declares.

    Read from the application object rather than listed, so a route added to
    or removed from ``src/api/main.py`` changes the figure. The OpenAPI
    documentation routes FastAPI adds on its own are dropped: they describe the
    API, they are not part of it.
    """
    from src.api.main import create_app

    app = create_app(static_root=_root() / "frontend" / "__no_static__", preload=False)
    routes: list[tuple[str, str]] = []
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = str(getattr(route, "path", ""))
        if not methods or path.startswith(("/docs", "/redoc", "/openapi")):
            continue
        for method in sorted(set(methods) - {"HEAD"}):
            routes.append((method, path))
    return routes


def t19_stages() -> list[dict[str, Any]]:
    """T19's optimization stages in order, as dictionaries."""
    import pandas as pd

    frame = pd.read_csv(_outputs("ablation") / "T19_search_and_optimization_ablation.csv")
    return [dict(row) for _, row in frame.sort_values("stage_order").iterrows()]


def t30_rows() -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_csv(_outputs("evidence_index") / "T30_final_conclusion_matrix.csv")
    return [dict(row) for _, row in frame.sort_values("objective").iterrows()]


def verdict_headline(verdict: str) -> str:
    """The first clause of a T30 verdict -- its claim, without its qualifiers."""
    for separator in (";", ":", " -- "):
        verdict = verdict.split(separator)[0]
    return verdict.strip()


def _signed(value: float) -> str:
    return format(float(value), "+.3f")


# ---------------------------------------------------------------------------
# layout helpers
# ---------------------------------------------------------------------------


def _row_height(canvas: DiagramCanvas, row: Row) -> float:
    """The height of the tallest box in a row, measured from its own text."""
    return max(
        canvas.fit_height(label, width=width, sublabel=sub, kind=kind)
        for _, label, kind, _, width, sub in row
    )


def _place_row(canvas: DiagramCanvas, row: Row, top: float, *, gap_inches: float = 0.42) -> float:
    """Draw a row of boxes at one shared height; return where the next row starts."""
    height = _row_height(canvas, row)
    for key, label, kind, col, width, sub in row:
        canvas.node(
            key, label, kind=kind, col=col, row=top, width=width, height=height, sublabel=sub
        )
    return top + height + canvas.y_units(gap_inches)


# ---------------------------------------------------------------------------
# F11 -- the binary pipeline
# ---------------------------------------------------------------------------


@diagram("F11")
def build_f11() -> DiagramCanvas:
    """PhysioNet binary screening, EXP-A2, from recordings to the chosen model.

    Laid out as a snake -- left to right, then right to left inside the fold
    lane, then left to right again -- so no arrow has to cross the page.
    """
    from src.evaluation.tuned import DEFAULT_TRIALS
    from src.optimization.base import DEFAULT_INNER_SPLITS

    exp = _experiment("EXP-A2")
    scheme = _scheme(str(exp["cv"]))
    defaults = _config("experiments").require("defaults")
    rule = [str(metric) for metric in defaults["selection_rule"]]
    metrics = [str(metric) for metric in defaults["report_metrics_binary"]]
    d1 = {row["dataset_source"]: row for row in _datasets()}["D1"]
    features = _features()
    final = final_binary_model()
    final_id = str(final["selected_model_id"])
    splits, total = int(scheme["n_splits"]), int(scheme["total_folds"])

    canvas = DiagramCanvas(
        columns=12.0,
        rows=11.0,
        size=(9.0, 8.2),
        title="F11  Binary classification pipeline (EXP-A2)",
        subtitle=(
            "Normal versus abnormal on PhysioNet 2016. Protocol, models, search budget, "
            "selection rule and reported metrics are read from configs/experiments.yaml and "
            "the deployed model's manifest."
        ),
        legend_rows=2,
    )

    top = _place_row(
        canvas,
        [
            (
                "data",
                str(d1["dataset_name"]),
                "input",
                0.2,
                3.6,
                d1["n_modelled_display"]
                + " recordings, "
                + d1["n_subjects_display"]
                + " subjects; "
                + _counts_text("binary"),
            ),
            (
                "extract",
                "Preprocess, extract features",
                "process",
                4.25,
                3.5,
                str(features["n_features"]) + " values per recording",
            ),
            (
                "map",
                "Stored fold map",
                "store",
                8.2,
                3.6,
                str(total) + " folds, grouped on " + str(scheme["group_key"]),
            ),
        ],
        0.2,
    )
    canvas.edge("data", "extract", kind="flow")
    canvas.edge("extract", "map", kind="flow")

    lane_top = top
    row_a = lane_top + canvas.y_units(0.36)
    row_b = _place_row(
        canvas,
        [
            (
                "search",
                "Nested Bayesian search",
                "process",
                0.5,
                3.6,
                str(DEFAULT_TRIALS)
                + " trials per model, "
                + str(DEFAULT_INNER_SPLITS)
                + " inner splits, scored on "
                + str(defaults["scoring"]["binary"]),
            ),
            (
                "fit",
                "Imputer, scaler, selector",
                "process",
                4.5,
                3.5,
                "all "
                + str(final["n_features"])
                + " columns in the headline run; the "
                + str(features["selected"]["n_selected"])
                + "-feature SO-04 subset is a variant",
            ),
            (
                "train",
                "Training fold",
                "process",
                8.5,
                3.0,
                str(splits - 1) + " of " + str(splits) + " subject groups",
            ),
        ],
        row_a,
        gap_inches=0.5,
    )
    lane_bottom = _place_row(
        canvas,
        [
            (
                "models",
                " ".join(str(m) for m in exp["models"]),
                "model",
                0.5,
                3.6,
                "each tuned on the training fold only",
            ),
            (
                "score",
                "Transform, predict, score once",
                "process",
                4.5,
                3.5,
                "per_fold_metrics.csv",
            ),
            (
                "test",
                "Outer test fold",
                "process",
                8.5,
                3.0,
                "1 of " + str(splits) + " subject groups",
            ),
        ],
        row_b,
        gap_inches=0.12,
    )
    canvas.lane(
        "inside each of the " + str(total) + " outer folds",
        col=0.2,
        row=lane_top,
        width=11.6,
        height=lane_bottom - lane_top,
    )
    canvas.edge("map", "train", kind="flow")
    canvas.edge("train", "fit", kind="flow")
    canvas.edge("fit", "search", kind="flow")
    canvas.edge("search", "models", kind="flow")
    canvas.edge("models", "score", kind="flow")
    canvas.edge("test", "score", kind="flow")
    canvas.edge("test", "fit", kind="forbidden", label="never fitted on")

    _place_row(
        canvas,
        [
            ("rule", "Selection rule", "process", 0.3, 3.9, ", then ".join(rule)),
            (
                "final",
                "Final model " + final_id,
                "model",
                4.6,
                3.2,
                _model_name(final_id) + ", refitted on all " + _count(final["n_records_fitted"]),
            ),
            (
                "tables",
                ", ".join(str(t) for t in exp.get("emits", [])) or "results",
                "artifact",
                8.2,
                3.6,
                ", ".join(metrics),
            ),
        ],
        lane_bottom + canvas.y_units(0.30),
    )
    canvas.edge("score", "rule", kind="flow", source_side="bottom", target_side="top")
    canvas.edge("rule", "final", kind="flow")
    canvas.edge("final", "tables", kind="flow")

    canvas.text(
        "Every number this pipeline reports is a cross-validated estimate within the "
        "PhysioNet 2016 corpus, and the rule never ranks by accuracy. How far it transfers "
        "is measured separately (F14, T-S5).",
        col=6.0,
        row=canvas.rows - canvas.y_units(0.04),
        ha="center",
        va="bottom",
    )
    return canvas


# ---------------------------------------------------------------------------
# F12 -- PASCAL A and B, two targets
# ---------------------------------------------------------------------------


@diagram("F12")
def build_f12() -> DiagramCanvas:
    """Two multiclass tracks side by side, and the merge that never happens."""
    tracks = [("EXP-B1", "D2", "pascal_a"), ("EXP-B2", "D3", "pascal_b")]
    datasets = {row["dataset_source"]: row for row in _datasets()}
    scoring = _config("experiments").require("defaults.scoring.multiclass")

    canvas = DiagramCanvas(
        columns=12.0,
        rows=8.4,
        size=(9.0, 6.8),
        title="F12  PASCAL multiclass pipeline",
        subtitle=(
            "PASCAL A and PASCAL B are two separate targets with their own folds. Class "
            "counts are read from the dataset audit (DA-02); protocols from "
            "configs/experiments.yaml."
        ),
        legend_rows=2,
    )

    bottoms: list[float] = []
    for index, (exp_id, source, task) in enumerate(tracks):
        exp = _experiment(exp_id)
        scheme = _scheme(str(exp["cv"]))
        left = 0.2 + index * 6.1
        suffix = str(index)
        group_note = (
            "record-level groups (no subject ids)"
            if str(scheme["group_key"]) == "split_group"
            else "grouped on " + str(scheme["group_key"])
        )
        row = canvas.y_units(0.42)
        for key, label, kind, sub in (
            (
                "data",
                str(datasets[source]["dataset_name"]),
                "input",
                datasets[source]["n_modelled_display"] + " labelled recordings",
            ),
            ("labels", str(len(exp["label_space"])) + " classes", "process", _counts_text(task)),
            (
                "cv",
                str(scheme["total_folds"]) + " folds, " + str(exp["cv"]),
                "process",
                group_note + "; class_weight " + str(exp.get("class_weight", "none")),
            ),
        ):
            row = _place_row(canvas, [(key + suffix, label, kind, left + 0.3, 5.1, sub)], row)
        bottom = _place_row(
            canvas,
            [
                (
                    "models" + suffix,
                    " ".join(str(m) for m in exp["models"]),
                    "model",
                    left + 0.3,
                    3.0,
                    "tuned on " + str(scoring),
                ),
                (
                    "out" + suffix,
                    ", ".join(str(t) for t in exp.get("emits", [])),
                    "artifact",
                    left + 3.55,
                    1.85,
                    "macro-F1, per-class recall",
                ),
            ],
            row,
            gap_inches=0.12,
        )
        bottoms.append(bottom)
        canvas.edge("data" + suffix, "labels" + suffix, kind="flow")
        canvas.edge("labels" + suffix, "cv" + suffix, kind="flow")
        canvas.edge("cv" + suffix, "models" + suffix, kind="flow")
        canvas.edge("models" + suffix, "out" + suffix, kind="flow")

    for index, (exp_id, _, _) in enumerate(tracks):
        canvas.lane(
            exp_id + "  " + str(_experiment(exp_id)["title"]),
            col=0.2 + index * 6.1,
            row=0.05,
            width=5.7,
            height=max(bottoms) - 0.05,
        )

    labels0 = canvas.nodes["labels0"]
    canvas.edge("labels0", "labels1", kind="forbidden")
    canvas.text(
        "never merged",
        col=6.0,
        row=labels0.row - canvas.y_units(0.02),
        ha="center",
        va="bottom",
        width=2.0,
        color=_WARN,
    )

    if any(name == "artifact" for name, _ in class_counts("pascal_a")):
        canvas.text(
            "In PASCAL A, 'artifact' is a recording-quality label, not a cardiac class: the "
            "four-class model is never described as a four-class cardiac classifier.",
            col=0.2,
            row=canvas.rows - canvas.y_units(0.04),
            ha="left",
            va="bottom",
        )
    return canvas


# ---------------------------------------------------------------------------
# F13 -- CirCor recording to patient
# ---------------------------------------------------------------------------


@diagram("F13")
def build_f13() -> DiagramCanvas:
    """Several recordings per patient, one referral decision per patient."""
    from src.evaluation.aggregation import AGGREGATION_RULES

    d4 = {row["dataset_source"]: row for row in _datasets()}["D4"]
    locations = _experiment("EXP-C3")
    excluded = sorted(str(x) for x in locations.get("exclude_locations", []))
    kept = [str(loc) for loc in locations["locations"] if str(loc) not in excluded]
    murmur, outcome = _experiment("EXP-C1"), _experiment("EXP-C2")
    scheme = _scheme(str(murmur["cv"]))

    # Paraphrases of the three branches of aggregate_predictions, checked there.
    descriptions = {
        "max": "highest recording probability, thresholded",
        "mean": "mean recording probability, thresholded",
        "any_present": "positive if any recording was classified positive",
    }

    canvas = DiagramCanvas(
        columns=12.0,
        rows=9.4,
        size=(9.0, 7.2),
        title="F13  CirCor recording-to-patient aggregation",
        subtitle=(
            "The model scores recordings; CirCor labels patients. The rules are "
            "src.evaluation.aggregation.AGGREGATION_RULES, and all of them are reported."
        ),
        legend_rows=2,
    )

    span = 8.6 / len(kept)
    top_row: Row = [
        (
            "patient",
            "One CirCor patient",
            "input",
            0.2,
            2.8,
            "of " + d4["n_subjects_display"] + "; " + d4["n_modelled_display"] + " recordings",
        )
    ]
    top_row += [
        ("loc" + str(i), loc, "input", 3.3 + i * span, span - 0.3, "recording")
        for i, loc in enumerate(kept)
    ]
    row = _place_row(canvas, top_row, 0.2, gap_inches=0.55)
    canvas.edge("patient", "loc0", kind="link")
    if excluded:
        canvas.text(
            "excluded: " + ", ".join(excluded) + " (too few recordings)",
            col=0.2,
            row=row - canvas.y_units(0.30),
            ha="left",
            va="center",
            width=3.0,
        )

    row = _place_row(
        canvas,
        [
            (
                "model",
                "Recording-level model",
                "model",
                3.3,
                5.4,
                "folds grouped on " + str(scheme["group_key"]) + ": a patient never splits",
            )
        ],
        row,
    )
    for index in range(len(kept)):
        canvas.edge(
            "loc" + str(index), "model", kind="flow", source_side="bottom", target_side="top"
        )

    rule_span = 12.0 / len(AGGREGATION_RULES)
    row = _place_row(
        canvas,
        [
            (
                "rule" + str(i),
                rule,
                "process",
                i * rule_span + 0.25,
                rule_span - 0.5,
                descriptions.get(rule, ""),
            )
            for i, rule in enumerate(AGGREGATION_RULES)
        ],
        row,
    )
    emits = sorted({str(t) for t in murmur.get("emits", []) + outcome.get("emits", [])})
    _place_row(
        canvas,
        [
            (
                "patient_out",
                "Patient-level indication",
                "artifact",
                3.3,
                5.4,
                "murmur and outcome, scored separately: " + ", ".join(emits),
            )
        ],
        row,
    )
    for index in range(len(AGGREGATION_RULES)):
        canvas.edge(
            "model", "rule" + str(index), kind="flow", source_side="bottom", target_side="top"
        )
        canvas.edge(
            "rule" + str(index), "patient_out", kind="flow", source_side="bottom", target_side="top"
        )
    canvas.text(
        "No rule is declared the winner: choosing one is a decision about missed "
        "referrals, which T15 reports rather than makes.",
        col=0.2,
        row=canvas.rows - canvas.y_units(0.04),
        ha="left",
        va="bottom",
        width=11.6,
    )
    return canvas


# ---------------------------------------------------------------------------
# F14 -- EXP-D1
# ---------------------------------------------------------------------------


@diagram("F14")
def build_f14() -> DiagramCanvas:
    """Adult PhysioNet to paediatric CirCor, with the mismatch recorded first."""
    meta = population_mismatch()
    exp = _experiment("EXP-D1")
    train, test = meta["train"], meta["test"]
    model_id = str(meta["model"]["selected_model_id"])

    canvas = DiagramCanvas(
        columns=12.0,
        rows=8.8,
        size=(9.0, 7.0),
        title="F14  Cross-dataset external validation workflow (EXP-D1)",
        subtitle=(
            "Trained on adult PhysioNet, applied unchanged to paediatric CirCor. Population "
            "figures are read from EXP-D1/population_mismatch.json, written before any metric."
        ),
        legend_rows=2,
    )

    row = _place_row(
        canvas,
        [
            (
                "train",
                str(train["dataset"]),
                "input",
                0.2,
                3.6,
                str(train["cohort"])
                + ": median age "
                + format(float(train["age_median_years"]), ".0f")
                + " years, "
                + format(100.0 * float(train["share_under_18"]), ".1f")
                + "% under 18",
            ),
            (
                "test",
                str(test["dataset"]),
                "input",
                8.2,
                3.6,
                str(test["cohort"])
                + ": "
                + format(100.0 * float(test["share_paediatric_of_recorded"]), ".1f")
                + "% paediatric of recorded ages",
            ),
        ],
        0.2,
    )
    row = _place_row(
        canvas,
        [
            (
                "fit",
                "Final model " + model_id,
                "model",
                0.2,
                3.6,
                "tuned in " + str(exp.get("tuning_source", "")) + ", then frozen",
            ),
            (
                "mismatch",
                "population_mismatch.json",
                "artifact",
                4.2,
                3.6,
                "written before any metric: "
                + str(bool(meta.get("written_before_any_metric"))).lower(),
            ),
        ],
        row,
    )
    canvas.edge("train", "fit", kind="flow")
    canvas.edge("train", "mismatch", kind="flow", source_side="right", target_side="top")
    canvas.edge("test", "mismatch", kind="flow", source_side="left", target_side="top")

    row = _place_row(
        canvas,
        [
            (
                "apply",
                "Apply unchanged",
                "process",
                4.2,
                3.6,
                "retuning allowed: " + str(bool(meta.get("retuning_allowed"))).lower(),
            )
        ],
        row,
    )
    canvas.edge("fit", "apply", kind="flow", source_side="bottom", target_side="left")
    canvas.edge("mismatch", "apply", kind="flow")
    canvas.edge("test", "apply", kind="flow", source_side="bottom", target_side="right")

    _place_row(
        canvas,
        [
            (
                "score",
                "Score against CirCor outcome",
                "process",
                4.2,
                3.6,
                "recording level and patient level",
            ),
            (
                "out",
                ", ".join(str(t) for t in exp.get("emits", [])),
                "artifact",
                8.2,
                3.6,
                "drop against the in-domain estimate",
            ),
        ],
        row,
    )
    canvas.edge("apply", "score", kind="flow")
    canvas.edge("score", "out", kind="flow")

    canvas.text(
        "Framing fixed in advance: "
        + str(meta["framing_rule"]).split(". ")[0]
        + ". A second cause is recording source, which EXP-F3 isolates.",
        col=0.2,
        row=canvas.rows - canvas.y_units(0.04),
        ha="left",
        va="bottom",
        width=11.6,
    )
    return canvas


# ---------------------------------------------------------------------------
# F15 -- robustness
# ---------------------------------------------------------------------------


def _level(value: float, unit: str, control: str) -> str:
    if value == float("inf"):
        return control
    return format(value, "g") + " " + unit


@diagram("F15")
def build_f15() -> DiagramCanvas:
    """Three stressors, each observed on the corpus and, where possible, applied."""
    from src.evaluation.duration import DURATION_BANDS, TRUNCATION_SECONDS
    from src.evaluation.robustness import QUALITY_GROUPS, SNR_LEVELS

    bands = _config("signal").require("integrity.duration_bands")
    short, long_ = bands["short_below"], bands["long_above"]
    e1, e2, c3 = _experiment("EXP-E1"), _experiment("EXP-E2"), _experiment("EXP-C3")
    excluded = {str(x) for x in c3.get("exclude_locations", [])}
    locations = [str(loc) for loc in c3["locations"] if str(loc) not in excluded]
    band_text = {
        "short": "under " + format(short, "g") + " s",
        "medium": format(short, "g") + "-" + format(long_, "g") + " s",
        "long": "over " + format(long_, "g") + " s",
    }

    canvas = DiagramCanvas(
        columns=12.0,
        rows=7.0,
        size=(9.0, 5.9),
        title="F15  Noise and duration robustness framework",
        subtitle=(
            "Left: groups observed on the stored out-of-fold predictions. Right: stressors "
            "applied to a held-out sample. Groups, levels and bands are the constants the "
            "evaluation modules run with."
        ),
        legend_rows=1,
    )

    stressors = [
        (
            "noise",
            "Noise: quality groups",
            ", ".join(QUALITY_GROUPS) + " (PP-08, checked against REFERENCE-SQI)",
            "Added white noise",
            ", ".join(_level(v, "dB", "clean") for v in SNR_LEVELS),
            ", ".join(str(t) for t in e1.get("emits", [])),
        ),
        (
            "duration",
            "Duration bands",
            ", ".join(b + " " + band_text.get(b, "") for b in DURATION_BANDS),
            "Random-window clips",
            ", ".join(_level(v, "s", "full length") for v in TRUNCATION_SECONDS),
            ", ".join(str(t) for t in e2.get("emits", [])),
        ),
        (
            "location",
            "Auscultation location",
            "CirCor sites " + ", ".join(locations),
            "No intervention",
            "a site cannot be applied to a recording",
            ", ".join(str(t) for t in c3.get("emits", [])),
        ),
    ]
    lane_top = 0.05
    row = lane_top + canvas.y_units(0.40)
    for key, observed, groups, applied, levels, emits in stressors:
        row = _place_row(
            canvas,
            [
                (key, observed, "process", 0.45, 4.9, groups),
                (key + "_applied", applied, "process", 6.0, 4.0, levels),
                (key + "_out", emits, "artifact", 10.4, 1.45, ""),
            ],
            row,
            gap_inches=0.30,
        )
        canvas.edge(key, key + "_applied", kind="flow")
        canvas.edge(key + "_applied", key + "_out", kind="flow")
    canvas.lane(
        "observed: stored out-of-fold predictions",
        col=0.2,
        row=lane_top,
        width=5.4,
        height=row - lane_top - canvas.y_units(0.18),
    )
    canvas.lane(
        "applied: refitted, held-out sample",
        col=5.75,
        row=lane_top,
        width=4.5,
        height=row - lane_top - canvas.y_units(0.18),
    )

    canvas.text(
        "Each applied sweep keeps an untouched control that reproduces the stored features "
        "exactly. A noise result is never pooled across PASCAL and the other corpora: "
        "PASCAL's 'artifact' is itself a quality label.",
        col=0.2,
        row=canvas.rows - canvas.y_units(0.04),
        ha="left",
        va="bottom",
        width=11.6,
    )
    return canvas


# ---------------------------------------------------------------------------
# F16 -- dashboard architecture
# ---------------------------------------------------------------------------


@diagram("F16")
def build_f16() -> DiagramCanvas:
    """Build time on the left, runtime on the right, and one boundary between."""
    from src.reporting.frontend_export import DEFAULT_SOURCE_DIRS, GENERATED_FILES

    routes = api_routes()
    compute = [path for method, path in routes if method == "POST"]
    status = [path for method, path in routes if method == "GET"]
    sources = [
        Path(str(_config("paths").require("outputs." + key))).name for key in DEFAULT_SOURCE_DIRS
    ]

    canvas = DiagramCanvas(
        columns=12.0,
        rows=12.2,
        size=(9.0, 9.4),
        title="F16  Dashboard system architecture",
        subtitle=(
            "Every precomputed number reaches the browser through build-time codegen; only "
            "the POST routes compute at runtime. Routes are read from the FastAPI "
            "application, the generated files from the exporter."
        ),
        legend_rows=3,
    )

    left, right, width = 0.5, 6.6, 4.9
    top = canvas.y_units(0.42)
    row = top
    for key, label, kind, sub in (
        ("outputs", "outputs/", "store", ", ".join(sources) + ", plus results when included"),
        (
            "export",
            "scripts/17_export_frontend_data.py",
            "process",
            "formats every number in Python",
        ),
        (
            "generated",
            "frontend/lib/generated/",
            "store",
            str(len(GENERATED_FILES)) + " generated files, plus one module per figure",
        ),
    ):
        row = _place_row(canvas, [(key, label, kind, left, width, sub)], row, gap_inches=0.34)
    row = _place_row(
        canvas, [("static", "frontend/out/", "store", left, width, "Next.js static export")], row
    )
    build_bottom = _place_row(
        canvas,
        [
            (
                "guard",
                "Metric-literal guard, fails the build",
                "process",
                left,
                width,
                "scripts/16_check_no_hardcoded_metrics.py",
            )
        ],
        row,
        gap_inches=0.12,
    )
    canvas.edge("outputs", "export", kind="derive")
    canvas.edge("export", "generated", kind="derive")
    canvas.edge("generated", "static", kind="derive", source_side="bottom", target_side="top")
    canvas.edge("guard", "static", kind="link")

    row = top
    for key, label, kind, sub in (
        ("browser", "Browser", "process", "renders pages from generated/"),
        (
            "api",
            "FastAPI (src/api/main.py)",
            "process",
            "serves frontend/out/ and answers " + ", ".join(status),
        ),
        ("predict", "POST " + ", ".join(compute), "process", "the only routes that compute"),
        (
            "bundle",
            "models_saved/<task>/final/",
            "store",
            "loaded once per process; an upload is deleted after scoring",
        ),
    ):
        row = _place_row(canvas, [(key, label, kind, right, width, sub)], row, gap_inches=0.34)
    height = max(row, build_bottom) - 0.05
    canvas.lane("build time", col=0.15, row=0.05, width=5.6, height=height)
    canvas.lane("runtime", col=6.25, row=0.05, width=5.6, height=height)

    canvas.edge("static", "api", kind="derive", source_side="right", target_side="left")
    canvas.edge("browser", "api", kind="flow")
    canvas.edge("api", "predict", kind="flow")
    canvas.edge("predict", "bundle", kind="flow")
    canvas.edge("browser", "outputs", kind="forbidden")

    canvas.text(
        "The crossed line is the fetch that does not exist: the browser never reads outputs/ "
        "at runtime. A value it shows either came through generated/ at build time or is a "
        "live prediction whose display strings were rounded by the API, never by the client.",
        col=0.2,
        row=canvas.rows - canvas.y_units(0.04),
        ha="left",
        va="bottom",
        width=11.6,
    )
    return canvas


# ---------------------------------------------------------------------------
# F17 -- one prediction, one report
# ---------------------------------------------------------------------------


@diagram("F17")
def build_f17() -> DiagramCanvas:
    """What one upload passes through, from WAV to screening report."""
    from src.inference.predictor import (
        ALLOWED_SUFFIXES,
        LOW_CONFIDENCE_MARGIN,
        MAX_DURATION_SECONDS,
        MIN_DURATION_SECONDS,
        TASKS,
    )
    from src.preprocessing.pipeline import PIPELINE_STEPS

    features = _features()
    canvas = DiagramCanvas(
        columns=12.0,
        rows=8.8,
        size=(9.0, 7.2),
        title="F17  Prediction and report generation flow",
        subtitle=(
            "src.inference.predictor.predict_recording and "
            "src.reporting.sample_report.render_sample_report. Bounds and thresholds are the "
            "module constants."
        ),
        legend_rows=2,
    )

    cols = (0.25, 4.25, 8.25)
    row = _place_row(
        canvas,
        [
            (
                "upload",
                "Uploaded recording",
                "input",
                cols[0],
                3.5,
                "suffix " + ", ".join(sorted(ALLOWED_SUFFIXES)) + ", spooled to a temporary file",
            ),
            (
                "validate",
                "Validate",
                "decision",
                cols[1],
                3.5,
                format(MIN_DURATION_SECONDS, "g") + "-" + format(MAX_DURATION_SECONDS, "g") + " s",
            ),
            (
                "bundle",
                "Load task bundle",
                "store",
                cols[2],
                3.5,
                str(len(TASKS)) + " tasks, never merged: " + ", ".join(TASKS),
            ),
        ],
        0.2,
    )
    canvas.edge("upload", "validate", kind="flow")
    canvas.edge("validate", "bundle", kind="flow")

    # The middle row runs right to left, so the turn at each end is a short
    # vertical arrow rather than a diagonal across the page.
    row = _place_row(
        canvas,
        [
            (
                "proba",
                "Class probabilities",
                "model",
                cols[0],
                3.5,
                "fitted pipeline, BLAS pinned to one thread",
            ),
            (
                "extract",
                "Extract features",
                "process",
                cols[1],
                3.5,
                str(features["n_features"]) + " values, in the bundle's column order",
            ),
            ("prep", "Preprocess", "process", cols[2], 3.5, " -> ".join(PIPELINE_STEPS)),
        ],
        row,
    )
    canvas.edge("bundle", "prep", kind="flow")
    canvas.edge("prep", "extract", kind="flow")
    canvas.edge("extract", "proba", kind="flow")

    _place_row(
        canvas,
        [
            (
                "margin",
                "Top-two margin",
                "decision",
                cols[0],
                3.5,
                "flag below " + format(LOW_CONFIDENCE_MARGIN, ".2f"),
            ),
            (
                "response",
                "JSON response",
                "artifact",
                cols[1],
                3.5,
                "probabilities, display strings, timings, model version, disclaimer",
            ),
            (
                "report",
                "Screening report (DOCX)",
                "artifact",
                cols[2],
                3.5,
                "waveform, spectrogram, contributions, model block, disclaimer",
            ),
        ],
        row,
    )
    canvas.edge("proba", "margin", kind="flow")
    canvas.edge("margin", "response", kind="flow")
    canvas.edge("response", "report", kind="flow")

    canvas.text(
        "Screening and decision support only; the disclaimer travels with every response and "
        "every report. The uploaded file is deleted whatever the outcome.",
        col=0.2,
        row=canvas.rows - canvas.y_units(0.04),
        ha="left",
        va="bottom",
        width=11.6,
    )
    return canvas


# ---------------------------------------------------------------------------
# F18 -- objective to module traceability
# ---------------------------------------------------------------------------


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


@diagram("F18")
def build_f18() -> DiagramCanvas:
    """Each locked objective, the code that answers it, and the tables that show it."""
    from src.reporting.conclusion_tables import OBJECTIVE_EVIDENCE
    from src.reporting.objectives import OBJECTIVES

    evidence = {row.number: row for row in OBJECTIVE_EVIDENCE}
    canvas = DiagramCanvas(
        columns=12.0,
        rows=10.4,
        size=(9.0, 8.6),
        title="F18  Objective-to-module traceability",
        subtitle=(
            "Objectives from src/reporting/objectives.py (blueprint wording, digest-locked); "
            "modules and tables from conclusion_tables.OBJECTIVE_EVIDENCE. Both are verified "
            "against the repository when exported."
        ),
        legend_rows=1,
    )

    row = 0.1
    for objective in OBJECTIVES:
        link = evidence.get(objective.number)
        modules = sorted({*objective.modules, *(link.modules if link else ())})
        tables = list(link.tables) if link else []
        key = "o" + str(objective.number)
        row = _place_row(
            canvas,
            [
                (key, "Objective " + str(objective.number), "input", 0.2, 2.9, objective.handle),
                (
                    key + "m",
                    str(len(modules)) + " modules",
                    "process",
                    3.4,
                    4.9,
                    ", ".join(_basename(m) for m in modules),
                ),
                (
                    key + "t",
                    ", ".join(tables) if tables else "no table yet",
                    "artifact",
                    8.7,
                    3.1,
                    link.datasets if link else "",
                ),
            ],
            row,
            gap_inches=0.18,
        )
        canvas.edge(key, key + "m", kind="flow")
        canvas.edge(key + "m", key + "t", kind="flow")
    return canvas


# ---------------------------------------------------------------------------
# F19 -- what the framework adds, stage by stage
# ---------------------------------------------------------------------------


@diagram("F19")
def build_f19() -> DiagramCanvas:
    """The baseline, each optimisation stage on top of it, and what each bought."""
    stages = t19_stages()
    metric = "sensitivity"
    second = "balanced_accuracy"

    canvas = DiagramCanvas(
        columns=12.0,
        rows=10.6,
        size=(9.0, 8.0),
        title="F19  Novelty contribution diagram",
        subtitle=(
            "Left: each optimisation stage over the default single model, with the change it "
            "made, read from T19 (25 folds each). Right: what the framework contributes that "
            "is not a score."
        ),
        legend_rows=2,
    )

    box_h = canvas.y_units(0.78)
    gap = canvas.y_units(0.28)
    best = max(stages[1:], key=lambda s: float(s[metric + "_incremental"]))
    for index, stage in enumerate(stages):
        key = "s" + str(index)
        is_ensemble = str(stage["model_id"]) in {"M6", "M7"}
        head = str(stage["source_run"]) + " " + str(stage["model_id"]) + ": "
        if index == 0:
            sub = head + "baseline, " + metric + " " + format(float(stage[metric]), ".3f")
        else:
            sub = (
                head
                + metric
                + " "
                + _signed(float(stage[metric + "_incremental"]))
                + ", balanced acc. "
                + _signed(float(stage[second + "_incremental"]))
            )
        canvas.node(
            key,
            str(stage["comparison"]),
            kind="model" if is_ensemble else "process",
            col=0.2,
            row=0.2 + index * (box_h + gap),
            width=6.0,
            height=box_h,
            sublabel=sub,
        )
        if index:
            canvas.edge("s" + str(index - 1), key, kind="flow")

    extras = [
        ("tracks", "Five separate label spaces", "binary, PASCAL A, PASCAL B, CirCor x2"),
        ("nested", "Nested, subject-grouped CV", "search never scored on the outer fold"),
        (
            "subset",
            str(_features()["selected"]["n_selected"]) + "-feature subset",
            "SO-04, one-standard-error guard (T18)",
        ),
        ("external", "External bounds reported", "EXP-D1 transfer and EXP-F3 hold-out"),
    ]
    for index, (key, label, sub) in enumerate(extras):
        canvas.node(
            key,
            label,
            kind="artifact",
            col=6.8,
            row=0.2 + index * (box_h + gap) * 1.25,
            width=5.0,
            height=box_h,
            sublabel=sub,
        )

    canvas.text(
        "Increments are against the previous stage. The largest "
        + metric
        + " step is '"
        + str(best["comparison"])
        + "' ("
        + _signed(float(best[metric + "_incremental"]))
        + "). A stage with a negative or near-zero step is drawn anyway: its contribution is "
        "measured, not assumed.",
        col=0.2,
        row=canvas.rows - canvas.y_units(0.04),
        ha="left",
        va="bottom",
        width=11.6,
    )
    return canvas


# ---------------------------------------------------------------------------
# F20 -- the contribution summary
# ---------------------------------------------------------------------------


@diagram("F20")
def build_f20() -> DiagramCanvas:
    """Six objectives, what was found, and the tables that carry each finding."""
    rows = t30_rows()
    canvas = DiagramCanvas(
        columns=12.0,
        rows=5.8,
        size=(9.0, 5.1),
        title="F20  Final research-contribution summary",
        subtitle=(
            "One card per objective: the verdict is the first clause of T30's own verdict, "
            "and the tables named are the ones it cites. Qualifiers and limitations are in "
            "T30 itself."
        ),
        legend_rows=1,
    )

    per_row = 2
    width = 12.0 / per_row
    cards = [
        (
            "c" + str(row["objective"]),
            "Objective " + str(row["objective"]) + ": " + str(row["handle"]),
            "artifact",
            (index % per_row) * width + 0.2,
            width - 0.4,
            verdict_headline(str(row["verdict"])) + ". Tables: " + str(row["tables"]),
        )
        for index, row in enumerate(rows)
    ]
    top = 0.1
    for start in range(0, len(cards), per_row):
        top = _place_row(canvas, cards[start : start + per_row], top, gap_inches=0.2)
    return canvas
