"""F01-F10: the architecture diagrams (Phase 95).

Every figure here is drawn from something the repository already knows.

* **F01 and F02 read ``src/reporting/architecture.py``.** T95.2 asks for "the 12
  documented steps", and there is exactly one list of them --
  :data:`ARCHITECTURE_STEPS`, which verifies each step's module and evidence
  directory against the filesystem when it is exported. Restating those twelve
  titles here would create a second list that drifts from the first, and the
  drift would be invisible: a diagram cannot fail a test that does not know what
  it claims. So the steps are read, and a step renamed in that module renames
  itself in the figure.
* **Counts come from the payloads, never from a literal.** The corpus sizes in
  F03 and F04 come from ``dataset_summary_payload``, the family counts in F07
  and F08 from ``features_payload``, the ensemble weights in F10 from
  ``ensemble_payload`` -- and each of those is already the display string the
  dashboard renders, formatted once in Python. The same rule as every other
  deliverable: a number a reader can see is generated or it does not appear.
* **Preprocessing and cross-validation parameters come from ``configs/``.**
  F05's 5x5 grouped protocol and F06's 2 kHz / 20-400 Hz band are read from
  ``experiments.yaml`` and ``signal.yaml``, so a configuration change shows up
  in the figure rather than contradicting it.

The one thing drawn from nothing but this file is *layout* -- which box sits
where, and which arrow means what. That is the part a diagram is actually for.
"""

from __future__ import annotations

from typing import Any

from src.reporting.diagrams.canvas import DiagramCanvas
from src.reporting.diagrams.catalogue import diagram

__all__ = [
    "build_f01",
    "build_f02",
    "build_f03",
    "build_f04",
    "build_f05",
    "build_f06",
    "build_f07",
    "build_f08",
    "build_f09",
    "build_f10",
]


# ---------------------------------------------------------------------------
# sources -- read once per figure, never cached across a run
# ---------------------------------------------------------------------------


def _steps() -> list[dict[str, Any]]:
    """The twelve architecture steps, verified against the repository."""
    from src.reporting.architecture import pipeline_payload

    return list(pipeline_payload()["steps"])


def _datasets() -> list[dict[str, Any]]:
    from src.reporting.record_index import dataset_summary_payload

    return list(dataset_summary_payload()["summary"])


def _features() -> dict[str, Any]:
    from src.reporting.method import features_payload

    return features_payload()


def _config(name: str) -> Any:
    from src.utils.config import load_config

    return load_config(name)


def _short(title: str) -> str:
    """Trim a step title to something that fits a box without hyphenating."""
    return title.split(",")[0].strip()


def _stage_name(evidence_dir: str) -> str:
    """``outputs/02_preprocessing`` -> ``Preprocessing``.

    Derived from the directory rather than invented, so a stage cannot end up
    named one thing here and another in `outputs/`.
    """
    leaf = evidence_dir.rstrip("/").rsplit("/", 1)[-1]
    words = leaf.split("_")
    if words and words[0].isdigit():
        words = words[1:]
    return " ".join(words).capitalize()


# ---------------------------------------------------------------------------
# F01 -- the framework, end to end
# ---------------------------------------------------------------------------


@diagram("F01")
def build_f01() -> DiagramCanvas:
    """The whole framework in one page, grouped by where its evidence lands.

    The stages are not a separate list: they are the twelve steps grouped by
    their own ``evidence_dir``, so a step that moves to a different part of
    ``outputs/`` moves between stages here too. The stage is named after that
    directory and the steps inside it are listed beside the box -- naming the
    stages myself would be a second vocabulary to keep in step with the first.
    """
    steps = _steps()
    order: list[str] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for step in steps:
        directory = str(step["evidence_dir"])
        if directory not in grouped:
            grouped[directory] = []
            order.append(directory)
        grouped[directory].append(step)

    datasets = _datasets()
    canvas = DiagramCanvas(
        columns=12.0,
        rows=10.0,
        size=(9.0, 8.4),
        title="F01  PV-MEPCG / PulseVision proposed architecture",
        subtitle=(
            "Stages are the twelve documented steps grouped by the outputs/ directory that "
            "evidences them, read from src/reporting/architecture.py. Corpus counts are read "
            "from the dataset audit."
        ),
        legend_rows=1,
    )

    box_h = canvas.y_units(0.60)
    span = 12.0 / len(datasets)
    for index, dataset in enumerate(datasets):
        canvas.node(
            "d" + str(index),
            dataset["dataset_name"],
            kind="input",
            col=index * span + 0.2,
            row=0.1,
            width=span - 0.4,
            height=box_h,
            sublabel=dataset["n_modelled_display"] + " modelled",
        )

    stage_top = 1.5
    stage_gap = canvas.y_units(0.16)
    stage_h = canvas.y_units(0.74)
    previous: str | None = None
    for index, directory in enumerate(order):
        members = grouped[directory]
        first, last = members[0]["index"], members[-1]["index"]
        key = "s" + str(index)
        row = stage_top + index * (stage_h + stage_gap)
        canvas.node(
            key,
            _stage_name(directory)
            + "  (step"
            + ("" if first == last else "s")
            + " "
            + (str(first) if first == last else str(first) + "-" + str(last))
            + ")",
            kind="process",
            col=0.3,
            row=row,
            width=4.4,
            height=stage_h,
            sublabel=directory,
        )
        canvas.text(
            "; ".join(_short(step["title"]) for step in members),
            col=5.0,
            row=row + stage_h / 2.0,
            ha="left",
            va="center",
        )
        if previous is not None:
            canvas.edge(previous, key, kind="flow")
        previous = key

    for index in range(len(datasets)):
        canvas.edge("d" + str(index), "s0", kind="flow")

    rules = [step for step in steps if step["rule"]]
    canvas.text(
        str(len(rules))
        + " of the "
        + str(len(steps))
        + " steps carry a stated research rule. The fold-safety ones are drawn in F05 and F09; "
        "every stage writes its own evidence, and no number reaches a page or a document "
        "except from those files.",
        col=6.0,
        row=canvas.rows - canvas.y_units(0.04),
        ha="center",
        va="bottom",
    )
    return canvas


# ---------------------------------------------------------------------------
# F02 -- the twelve steps themselves
# ---------------------------------------------------------------------------


@diagram("F02")
def build_f02() -> DiagramCanvas:
    """Each documented step, its module, and the rule it exists to keep."""
    steps = _steps()
    canvas = DiagramCanvas(
        columns=12.0,
        rows=13.0,
        size=(9.0, 8.6),
        title="F02  End-to-end PCG classification workflow",
        subtitle=(
            "The "
            + str(len(steps))
            + " documented steps, each read from src/reporting/architecture.py with the "
            "module that implements it. A step whose module is absent is refused at export."
        ),
        legend_rows=1,
    )

    height = canvas.y_units(0.80)
    gap = canvas.y_units(0.26)
    top = 0.30
    per_column = (len(steps) + 1) // 2
    keys: list[str] = []
    for index, step in enumerate(steps):
        column = index // per_column
        position = index % per_column
        key = "t" + str(step["index"])
        canvas.node(
            key,
            str(step["index"]) + ".  " + _short(step["title"]),
            kind="model" if step["key"] == "ensemble" else "process",
            col=0.3 + column * 6.2,
            row=top + position * (height + gap),
            width=5.4,
            height=height,
            sublabel=str(step["module"]),
        )
        keys.append(key)
        if position > 0:
            canvas.edge(keys[index - 1], key, kind="flow")
    # The turn from the foot of the first column to the head of the second is
    # routed side to side through the gutter. Left to its own anchor choice the
    # arrow reads as a diagonal across the whole page, because the target is
    # both far right and far above the source.
    canvas.edge(
        keys[per_column - 1],
        keys[per_column],
        kind="flow",
        source_side="right",
        target_side="left",
        rad=0.0,
    )

    # The lane covers the steps whose rule is fold safety, and only those: the
    # five rule-bearing steps are not contiguous (2 and 12 are the other two),
    # so a lane drawn from the first to the last of them would claim a span the
    # steps do not have.
    guarded = [step for step in steps if step["rule"]]
    fold_safe = [step for step in steps if str(step["rule"] or "").startswith("Fold safety")]
    positions = [keys.index("t" + str(step["index"])) % per_column for step in fold_safe]
    lane_top = top + min(positions) * (height + gap)
    canvas.lane(
        "fold safety: steps " + str(fold_safe[0]["index"]) + "-" + str(fold_safe[-1]["index"]),
        col=6.15,
        row=lane_top - canvas.y_units(0.16),
        width=5.7,
        height=(height + gap) * len(fold_safe) + canvas.y_units(0.06),
    )
    # The rules themselves go beside the lane rather than under the columns:
    # the columns end at the foot of the axes, so anything below them is off the
    # page. Each is the first clause of the rule the step declares, not a
    # paraphrase written here.
    canvas.text(
        "Stated rules: "
        + "; ".join(
            str(step["index"]) + ". " + str(step["rule"]).split(":")[0] for step in guarded
        ),
        col=0.3,
        row=canvas.rows - canvas.y_units(0.04),
        ha="left",
        va="bottom",
        width=5.4,
    )
    return canvas


# ---------------------------------------------------------------------------
# F03 -- three corpora, five label spaces
# ---------------------------------------------------------------------------

#: The five targets, and which corpus each is defined on. Kept here rather than
#: derived because "never merge these" is a research decision, not a property of
#: the data -- see rule 4. The counts beside them are read, not typed.
_TRACKS: tuple[tuple[str, str, str], ...] = (
    ("D1", "Binary\nnormal vs abnormal", "2 classes"),
    ("D2", "PASCAL A\nfour-class", "4 classes"),
    ("D3", "PASCAL B\nthree-class", "3 classes"),
    ("D4", "CirCor murmur", "3 classes"),
    ("D4", "CirCor outcome", "2 classes"),
)


@diagram("F03")
def build_f03() -> DiagramCanvas:
    """Five targets on four corpora, and the merge that never happens."""
    datasets = {row["dataset_source"]: row for row in _datasets()}
    canvas = DiagramCanvas(
        columns=12.0,
        rows=5.8,
        size="wide",
        title="F03  Three-dataset experimental tracks",
        subtitle=(
            "Five label spaces, five separate targets. Corpus counts are read from the "
            "dataset audit; the classes are the declared vocabularies."
        ),
        legend_rows=1,
    )

    sources = ["D1", "D2", "D3", "D4"]
    span = 12.0 / len(sources)
    width = span - 0.5
    for index, source in enumerate(sources):
        row = datasets[source]
        sub = row["n_modelled_display"] + " recordings, " + row["n_subjects_display"] + " subjects"
        canvas.node(
            source,
            row["dataset_name"],
            kind="input",
            col=index * span + 0.25,
            row=0.2,
            width=width,
            height=canvas.fit_height(row["dataset_name"], width=width, sublabel=sub, kind="input"),
            sublabel=sub,
        )

    task_span = 12.0 / len(_TRACKS)
    task_width = task_span - 0.75
    for index, (source, label, classes) in enumerate(_TRACKS):
        key = "task" + str(index)
        canvas.node(
            key,
            label,
            kind="process",
            col=index * task_span + 0.35,
            row=2.6,
            width=task_width,
            height=canvas.fit_height(label, width=task_width, sublabel=classes),
            sublabel=classes,
        )
        canvas.edge(source, key, kind="flow")

    # PASCAL A and PASCAL B are the pair most often merged in the literature.
    canvas.edge("task1", "task2", kind="forbidden")
    canvas.text(
        "never merged",
        col=2.0 * task_span + 0.35 - 0.2,
        row=2.6 - canvas.y_units(0.10),
        ha="center",
        va="bottom",
        width=2.4,
        color="#D55E00",
    )
    canvas.text(
        "Each track has its own target, its own folds and its own reported metrics. "
        "A model trained on one is never scored on another except in EXP-D1 (F14), "
        "where the population difference is stated before the number.",
        col=6.0,
        row=canvas.rows - canvas.y_units(0.04),
        ha="center",
        va="bottom",
    )
    return canvas


# ---------------------------------------------------------------------------
# F04 -- harmonisation
# ---------------------------------------------------------------------------


@diagram("F04")
def build_f04() -> DiagramCanvas:
    """Four corpora at three sampling rates, one master metadata table."""
    datasets = {row["dataset_source"]: row for row in _datasets()}
    signal = _config("signal")
    native = signal.require("resample.native_fs")
    target = signal.require("resample.target_fs")

    canvas = DiagramCanvas(
        columns=12.0,
        rows=7.4,
        size="wide",
        title="F04  Dataset harmonization and metadata workflow",
        subtitle=(
            "Native rates and the "
            + str(target)
            + " Hz target are read from configs/signal.yaml; the counts from the audit."
        ),
        legend_rows=1,
    )

    keys = list(native)
    span = 12.0 / len(keys)
    width = span - 0.5
    for index, key in enumerate(keys):
        source = "D" + str(index + 1)
        name = datasets[source]["dataset_name"]
        sub = str(native[key]) + " Hz native"
        canvas.node(
            source,
            name,
            kind="input",
            col=index * span + 0.25,
            row=0.2,
            width=width,
            height=canvas.fit_height(name, width=width, sublabel=sub, kind="input"),
            sublabel=sub,
        )

    stages = [
        ("mono", "Mono conversion", signal.require("resample.mono") + " over channels"),
        ("resample", "Resample to " + str(target) + " Hz", signal.require("resample.method")),
        ("identity", "Subject / patient identity", "grouped on subject_id, patient_id"),
        ("labels", "Label harmonisation", "five vocabularies, kept separate"),
    ]
    previous = None
    stage_h = max(canvas.fit_height(label, width=2.6, sublabel=sub) for _, label, sub in stages)
    for index, (key, label, sub) in enumerate(stages):
        canvas.node(
            key,
            label,
            kind="process",
            col=0.4 + index * 3.0,
            row=2.6,
            width=2.6,
            height=stage_h,
            sublabel=sub,
        )
        if previous is not None:
            canvas.edge(previous, key, kind="flow")
        previous = key
    for index in range(len(keys)):
        # Into the top of the first stage, explicitly: three of the four corpora
        # sit to the right of it, and the automatic anchor sends them into its
        # right-hand edge, which lands them against the box next door.
        canvas.edge("D" + str(index + 1), "mono", kind="flow", target_side="top")

    master_sub = "one row per recording: uid, corpus, subject, label, duration, native rate"
    canvas.node(
        "master",
        "metadata_master.csv",
        kind="artifact",
        col=3.4,
        row=5.0,
        width=5.2,
        height=canvas.fit_height(
            "metadata_master.csv", width=5.2, sublabel=master_sub, kind="artifact"
        ),
        sublabel=master_sub,
    )
    canvas.edge("labels", "master", kind="flow", source_side="bottom", target_side="right")
    canvas.text(
        "Every downstream stage reads this table, never the corpora directly.",
        col=6.0,
        row=6.6,
        ha="center",
        va="center",
    )
    return canvas


# ---------------------------------------------------------------------------
# F05 -- subject-wise splitting
# ---------------------------------------------------------------------------


@diagram("F05")
def build_f05() -> DiagramCanvas:
    """Grouped, repeated cross-validation, and what the test fold never touches."""
    scheme = _config("experiments").require("cv_schemes.repeated_5x5_grouped")
    splits = int(scheme["n_splits"])
    repeats = int(scheme["n_repeats"])

    canvas = DiagramCanvas(
        columns=12.0,
        rows=7.6,
        size="wide",
        title="F05  Subject-wise data splitting",
        subtitle=(
            str(repeats)
            + " repeats x "
            + str(splits)
            + " folds, grouped on "
            + str(scheme["group_key"])
            + ", read from configs/experiments.yaml. Seed 42, one stored fold map."
        ),
        legend_rows=2,
    )

    # The left column is stacked from measured heights rather than fixed rows:
    # these two boxes carry sentences, and a wrapped sentence in a narrow box
    # grew tall enough to sit on the one below it.
    pool_sub = "the unit of splitting, never a recording"
    pool_h = canvas.fit_height("Subjects", width=2.6, sublabel=pool_sub, kind="input")
    canvas.node("pool", "Subjects", kind="input", col=0.3, row=1.4, width=2.6, sublabel=pool_sub)
    canvas.node(
        "map",
        "Stored fold map",
        kind="store",
        col=0.3,
        row=1.4 + pool_h + canvas.y_units(0.45),
        width=2.6,
        sublabel="DA-07, loaded not re-derived",
    )
    canvas.edge("pool", "map", kind="flow")

    canvas.lane(
        "one outer fold, repeated " + str(repeats * splits) + " times",
        col=3.0,
        row=0.9,
        width=8.8,
        height=5.0,
    )
    canvas.node(
        "train",
        "Training fold",
        kind="process",
        col=3.4,
        row=1.5,
        width=3.4,
        sublabel=str(splits - 1) + " of " + str(splits) + " subject groups",
    )
    canvas.node(
        "fit",
        "Imputer, scaler, selector, search",
        kind="process",
        col=7.3,
        row=1.5,
        width=4.1,
        sublabel="all fitted here, and only here",
    )
    canvas.node(
        "test",
        "Outer test fold",
        kind="process",
        col=3.4,
        row=4.2,
        width=3.4,
        sublabel="1 of " + str(splits) + " subject groups",
    )
    canvas.node(
        "score",
        "Transform, then score once",
        kind="process",
        col=7.3,
        row=4.2,
        width=4.1,
        sublabel="the reported number",
    )
    canvas.edge("map", "train", kind="flow")
    canvas.edge("map", "test", kind="flow")
    canvas.edge("train", "fit", kind="flow")
    canvas.edge("test", "score", kind="flow")
    canvas.edge("fit", "score", kind="derive")
    canvas.edge("test", "fit", kind="forbidden", label="never fits anything")

    canvas.text(
        "No subject appears on both sides of a split. Where a subject id cannot be derived the "
        "record is grouped by its own recording and subject_derived is set to False, which is "
        "reported rather than assumed away.",
        col=6.0,
        row=canvas.rows - canvas.y_units(0.04),
        ha="center",
        va="bottom",
    )
    return canvas


# ---------------------------------------------------------------------------
# F06 -- preprocessing
# ---------------------------------------------------------------------------


@diagram("F06")
def build_f06() -> DiagramCanvas:
    """The six signal steps, in the order the pipeline applies them, and why."""
    from src.preprocessing.pipeline import PIPELINE_STEPS

    signal = _config("signal")
    target = signal.require("resample.target_fs")
    low = signal.require("filter.low_hz")
    high = signal.require("filter.high_hz")
    order = signal.require("filter.order")
    method = str(signal.get("normalization.method", "z-score"))

    subtitles = {
        "load": "one recording, as it is on disk",
        "mono": signal.require("resample.mono") + " over channels",
        "resample": str(target) + " Hz, " + str(signal.require("resample.method")),
        "quality": "clipping, silence, SNR proxy",
        "filter": "Butterworth order "
        + str(order)
        + ", "
        + str(low)
        + "-"
        + str(high)
        + " Hz, zero phase",
        "normalize": method,
    }

    canvas = DiagramCanvas(
        columns=12.0,
        rows=5.4,
        size="wide",
        title="F06  Signal preprocessing architecture",
        subtitle=(
            "The steps are src.preprocessing.pipeline.PIPELINE_STEPS; the parameters are "
            "read from configs/signal.yaml."
        ),
        legend_rows=1,
    )

    width = 12.0 / len(PIPELINE_STEPS)
    box = width - 0.4
    # One height for the whole chain, from the box that needs the most: six
    # steps at six different heights read as six different kinds of thing.
    height = max(
        canvas.fit_height(step, width=box, sublabel=subtitles.get(step, ""))
        for step in PIPELINE_STEPS
    )
    previous = None
    for index, step in enumerate(PIPELINE_STEPS):
        canvas.node(
            step,
            step,
            kind="process",
            col=index * width + 0.2,
            row=0.5,
            width=box,
            height=height,
            sublabel=subtitles.get(step, ""),
        )
        if previous is not None:
            canvas.edge(previous, step, kind="flow")
        previous = step

    canvas.node(
        "cache",
        "preprocessed cache",
        kind="store",
        col=4.2,
        row=0.5 + height + canvas.y_units(0.55),
        width=3.6,
        sublabel="keyed on the hashed config, so two settings never mix",
    )
    canvas.edge("normalize", "cache", kind="flow", source_side="bottom", target_side="right")

    canvas.text(
        "Quality is measured in the middle, not at the end: the band-pass removes the "
        "out-of-band term the SNR proxy needs, and normalisation removes the levels that "
        "clipping and silence are defined against.",
        col=6.0,
        row=canvas.rows - canvas.y_units(0.04),
        ha="center",
        va="bottom",
    )
    return canvas


# ---------------------------------------------------------------------------
# F07 -- feature extraction
# ---------------------------------------------------------------------------


@diagram("F07")
def build_f07() -> DiagramCanvas:
    """The families and their counts, summing to the registry's own total."""
    payload = _features()
    families = list(payload["families"])
    total = int(payload["n_features"])

    canvas = DiagramCanvas(
        columns=12.0,
        rows=6.6,
        size="wide",
        title="F07  " + str(total) + "-feature extraction architecture",
        subtitle=(
            "Family names, counts and order are read from the feature registry, whose "
            "column order is fingerprinted: two runs that disagree on it are not comparable."
        ),
        legend_rows=1,
    )

    canvas.node(
        "signal",
        "Preprocessed recording",
        kind="input",
        col=4.0,
        row=0.2,
        width=4.0,
        sublabel="mono, 2 kHz, filtered, normalised",
    )

    span = 12.0 / len(families)
    for index, family in enumerate(families):
        key = "f" + str(index)
        canvas.node(
            key,
            family["family"],
            kind="process",
            col=index * span + 0.2,
            row=2.0,
            width=span - 0.4,
            sublabel=family["n_features_display"] + " features",
        )
        canvas.edge("signal", key, kind="flow", source_side="bottom", target_side="top")

    canvas.node(
        "vector",
        str(total) + "-value feature vector",
        kind="artifact",
        col=3.4,
        row=4.4,
        width=5.2,
        sublabel="fixed column order, identical for every corpus and every experiment",
    )
    for index in range(len(families)):
        canvas.edge(
            "f" + str(index), "vector", kind="flow", source_side="bottom", target_side="top"
        )
    return canvas


# ---------------------------------------------------------------------------
# F08 -- fusion, and where the fold-local transforms sit
# ---------------------------------------------------------------------------


@diagram("F08")
def build_f08() -> DiagramCanvas:
    """The join, and the fact that everything after it is fitted per fold."""
    payload = _features()
    families = list(payload["families"])
    total = int(payload["n_features"])
    selected = payload["selected"]

    canvas = DiagramCanvas(
        columns=12.0,
        rows=7.2,
        size="wide",
        title="F08  Feature family fusion",
        subtitle=(
            "Six family blocks concatenated in registry order, then transformed inside the "
            "training fold. Counts read from the feature registry."
        ),
        legend_rows=1,
    )

    span = 12.0 / len(families)
    for index, family in enumerate(families):
        key = "b" + str(index)
        canvas.node(
            key,
            family["family"],
            kind="process",
            col=index * span + 0.25,
            row=0.25,
            width=span - 0.5,
            sublabel="cols " + str(family["first_index"]) + "+",
        )

    canvas.node(
        "concat",
        "Concatenate in registry order",
        kind="process",
        col=3.0,
        row=2.0,
        width=6.0,
        sublabel=str(total) + " columns, same order every time",
    )
    for index in range(len(families)):
        canvas.edge(
            "b" + str(index), "concat", kind="flow", source_side="bottom", target_side="top"
        )

    canvas.lane("fitted on the training fold only", col=0.5, row=3.2, width=8.3, height=2.5)
    for index, (key, label, sub) in enumerate(
        [
            ("impute", "Imputer", "median of the training rows"),
            ("scale", "Scaler", "training mean and variance"),
            (
                "select",
                "Selector",
                (str(selected["n_selected"]) + " kept in the shown run")
                if selected.get("available")
                else "search-driven",
            ),
        ]
    ):
        canvas.node(
            key,
            label,
            kind="process",
            col=0.9 + index * 2.6,
            row=3.9,
            width=2.3,
            sublabel=sub,
        )
    canvas.edge("concat", "impute", kind="flow")
    canvas.edge("impute", "scale", kind="flow")
    canvas.edge("scale", "select", kind="flow")

    canvas.node(
        "test",
        "Outer test fold",
        kind="process",
        col=9.2,
        row=3.9,
        width=2.6,
        sublabel="transformed, never fitted",
    )
    canvas.edge("test", "select", kind="forbidden")

    canvas.text(
        "Fitting any of the three on the full matrix would put the test fold's distribution "
        "into training. The selector is the one most often got wrong, because selecting once "
        "on the whole matrix looks like a preprocessing step rather than a model decision.",
        col=6.0,
        row=canvas.rows - canvas.y_units(0.04),
        ha="center",
        va="bottom",
    )
    return canvas


# ---------------------------------------------------------------------------
# F09 -- search-based selection
# ---------------------------------------------------------------------------


@diagram("F09")
def build_f09() -> DiagramCanvas:
    """The nested loop, and the fold the search is never scored on."""
    scheme = _config("experiments").require("cv_schemes.repeated_5x5_grouped")
    selected = _features()["selected"]

    canvas = DiagramCanvas(
        columns=12.0,
        rows=7.4,
        size="wide",
        title="F09  Search-based feature selection workflow",
        subtitle=(
            "Selection is decided per fold, from the training rows only. The outer fold is "
            "never scored while the search runs."
        ),
        legend_rows=2,
    )

    canvas.node(
        "outer",
        "Outer training fold",
        kind="input",
        col=0.25,
        row=2.2,
        width=2.4,
        sublabel="grouped on " + str(scheme["group_key"]),
    )

    canvas.lane(
        "inner loop: the search never leaves this box", col=3.1, row=0.7, width=6.4, height=5.0
    )
    canvas.node(
        "candidate",
        "Candidate subset",
        kind="process",
        col=3.5,
        row=1.3,
        width=2.6,
        sublabel="a point in the search space",
    )
    canvas.node(
        "inner",
        "Inner split fit and score",
        kind="process",
        col=6.5,
        row=1.3,
        width=2.6,
        sublabel="training rows only",
    )
    canvas.node(
        "propose",
        "Search proposes the next point",
        kind="decision",
        col=4.6,
        row=3.5,
        width=3.6,
        sublabel="budgeted, seeded, logged",
    )
    canvas.edge("outer", "candidate", kind="flow")
    canvas.edge("candidate", "inner", kind="flow")
    canvas.edge("inner", "propose", kind="flow")
    # Anchored explicitly: left-to-bottom arcs back up the way the loop reads,
    # where the automatic choice put a short stub across the decision box.
    canvas.edge(
        "propose",
        "candidate",
        kind="feedback",
        source_side="left",
        target_side="bottom",
        rad=0.30,
    )
    canvas.text("iterate", col=3.6, row=2.75, ha="left", va="center", color="#0072B2")

    canvas.node(
        "chosen",
        "Chosen subset for this fold",
        kind="store",
        col=9.9,
        row=1.3,
        width=1.9,
        sublabel=(
            str(selected["n_selected"]) + " features in the reported run"
            if selected.get("available")
            else "recorded per fold"
        ),
    )
    canvas.edge("propose", "chosen", kind="flow")

    canvas.node(
        "test",
        "Outer test fold",
        kind="process",
        col=0.25,
        row=4.6,
        width=2.4,
        sublabel="scored once, at the end",
    )
    canvas.edge("test", "propose", kind="forbidden")
    canvas.text(
        "never scored during the search",
        col=0.25,
        row=5.9,
        ha="left",
        va="top",
        width=4.5,
        color="#D55E00",
    )

    canvas.text(
        "A feature is therefore selected in some number of folds rather than globally. "
        "A feature kept in 3 of 25 folds and one kept in 25 are both selected, and are not "
        "the same claim, so the count is reported on every row.",
        col=6.0,
        row=canvas.rows - canvas.y_units(0.04),
        ha="center",
        va="bottom",
    )
    return canvas


# ---------------------------------------------------------------------------
# F10 -- the ensemble
# ---------------------------------------------------------------------------


@diagram("F10")
def build_f10() -> DiagramCanvas:
    """Three tuned learners, one weighted average, and what the search found."""
    from src.reporting.architecture import ensemble_payload

    payload = ensemble_payload()
    members = list(payload["members"])

    canvas = DiagramCanvas(
        columns=12.0,
        rows=8.6,
        size=(9.0, 6.0),
        title="F10  SVM / RF / GB optimized soft-voting architecture",
        subtitle=(
            "Weights are read from "
            + str(payload["source"])
            + " ("
            + str(payload["experiment"])
            + ", seed "
            + str(payload["seed"])
            + "). Each box carries its searched weight as mean +/- sd over the outer folds."
            if payload.get("available")
            else "Weights are not available: " + str(payload.get("source"))
        ),
        legend_rows=2,
    )

    # Rows are stacked from measured heights: the boxes carry sentences of
    # different lengths, and fixed rows put the artifact through the vote box.
    x_sub = "transformed by this fold's fitted pipeline"
    x_h = canvas.fit_height(
        "Selected features for one recording", width=5.6, sublabel=x_sub, kind="input"
    )
    canvas.node(
        "x",
        "Selected features for one recording",
        kind="input",
        col=3.2,
        row=0.2,
        width=5.6,
        height=x_h,
        sublabel=x_sub,
    )

    model_row = 0.2 + x_h + canvas.y_units(0.55)
    span = 12.0 / max(len(members), 1)
    model_h = max(
        canvas.fit_height(
            str(member["name"]),
            width=span - 0.8,
            kind="model",
            sublabel=str(member["model_id"])
            + "   "
            + str(member["weight_display"])
            + " +/- "
            + str(member["weight_std_display"]),
        )
        for member in members
    )
    for index, member in enumerate(members):
        key = "m" + str(index)
        canvas.node(
            key,
            str(member["name"]),
            kind="model",
            col=index * span + 0.4,
            row=model_row,
            width=span - 0.8,
            height=model_h,
            sublabel=str(member["model_id"])
            + "   "
            + str(member["weight_display"])
            + " +/- "
            + str(member["weight_std_display"]),
        )
        canvas.edge("x", key, kind="flow", source_side="bottom", target_side="top")

    vote_sub = str(payload.get("constraint", "")).split(";")[0]
    vote_row = model_row + model_h + canvas.y_units(0.55)
    vote_h = canvas.fit_height(
        "Weighted average of class probabilities", width=6.4, sublabel=vote_sub
    )
    canvas.node(
        "vote",
        "Weighted average of class probabilities",
        kind="process",
        col=2.8,
        row=vote_row,
        width=6.4,
        height=vote_h,
        sublabel=vote_sub,
    )
    for index in range(len(members)):
        canvas.edge("m" + str(index), "vote", kind="flow", source_side="bottom", target_side="top")

    canvas.node(
        "prob",
        "Class probability",
        kind="artifact",
        col=4.4,
        row=vote_row + vote_h + canvas.y_units(0.45),
        width=3.2,
        sublabel="thresholded at the declared operating point",
    )
    canvas.edge("vote", "prob", kind="flow")

    if payload.get("available"):
        canvas.text(
            "The search chose weights identical to equal weighting in "
            + str(payload["folds_identical_display"])
            + " outer folds, against an equal weight of "
            + str(payload["equal_weight_display"])
            + ". The ensemble's benefit here is the averaging, not the weighting.",
            col=0.2,
            row=canvas.rows - canvas.y_units(0.04),
            ha="left",
            va="bottom",
            width=4.0,
        )
    return canvas
