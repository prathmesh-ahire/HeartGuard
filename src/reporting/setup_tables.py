"""T01-T07, the setup tables (Phase 86).

Seven tables that describe *what was run* rather than *what came out*: the
corpus, its class balance, its durations and sampling rates, the preprocessing
settings, the feature registry, the model configuration and the search space.
They are the tables an examiner reads first, and the ones a reader needs before
any result means anything.

Every one is a **view of a CSV another phase already wrote**, and this module
computes as little as possible on top:

===  ==========================================  ===========================
T01  Dataset Inventory                           DA-01
T02  Class Distribution and Imbalance Ratio      DA-02
T03  Recording Duration and Sampling Summary     DA-03 + DA-04 (joined)
T04  Preprocessing Configuration                 PP-07
T05  Feature Inventory and Counts                FE-01 + FE-02 (cross-checked)
T06  Model Hyperparameter Configuration          model_registry + search_spaces
T07  Search Space and Best Parameters            search_space_and_best_parameters
===  ==========================================  ===========================

Three decisions worth knowing about
-----------------------------------
**T02 and T03 report the SUPERVISED scope.** DA-02 and DA-03 both carry a
``scope`` column with ``supervised`` and ``all_records`` rows -- 3,240 labeled
PhysioNet records against 3,541 files, 124 labeled PASCAL A against 176. The
modelling tables report what was modelled. ``all_records`` is a corpus fact and
belongs in the audit report, not in a table a reader will compare against a
result. The scope is stated in the caption of both, so the smaller number is
never mistaken for an error.

**T05 recomputes the per-family counts from FE-01 rather than trusting FE-02.**
FE-02 already holds them, so copying would be shorter. Grouping FE-01's 138 rows
and asserting the result equals FE-02 turns the table into a check on the locked
138-feature registry: if the two ever disagree, T05 refuses to build instead of
publishing a count that does not match the features actually extracted.

**T07 renders hyperparameter values as text, not as rounded metrics.** ``C =
0.41967346752031676`` is the value that reproduces the run (rule 5). Rounding it
to ``0.420`` under the metric rule would produce a table that looks tidier and
cannot be pasted back into a config. Scores in the same table ARE metrics and do
round to 3 decimals; the table carries a note saying so, because a reader
noticing two precisions in one table should find the reason there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.reporting.tables import Column, Table, TableSpec, build_table
from src.utils.logging_setup import get_logger

__all__ = [
    "SETUP_TABLE_IDS",
    "SUPERVISED_SCOPE",
    "audit_dir",
    "outputs_dir",
    "build_t01",
    "build_t02",
    "build_t03",
    "build_t04",
    "build_t05",
    "build_t06",
    "build_t07",
    "build_setup_tables",
    "destination_for",
]

log = get_logger("reporting.setup_tables")

SETUP_TABLE_IDS: tuple[str, ...] = ("T01", "T02", "T03", "T04", "T05", "T06", "T07")

#: DA-02 and DA-03 both carry ``supervised`` and ``all_records`` rows.
SUPERVISED_SCOPE = "supervised"

#: Which ``outputs/`` directory each table is written into. A table lives beside
#: the data it summarises, which is the convention T64.6 and T66.6 already set
#: by putting T08/T10 in ``06_binary_results`` and T11/T12 in
#: ``07_multiclass_results``. Phases 103 and 104 collect them into the paper and
#: thesis asset packs; this is the working location, not the canonical one.
_DESTINATIONS: dict[str, str] = {
    "T01": "dataset_audit",
    "T02": "dataset_audit",
    "T03": "dataset_audit",
    "T04": "preprocessing",
    "T05": "features",
    "T06": "models",
    "T07": "search_optimization",
}


def outputs_dir(key: str) -> Path:
    """One ``outputs/`` subdirectory, resolved through ``configs/paths.yaml``."""
    from src.utils.config import load_config

    return Path(load_config("paths").require("outputs." + key))


def audit_dir() -> Path:
    return outputs_dir("dataset_audit")


def destination_for(table_id: str) -> Path:
    return outputs_dir(_DESTINATIONS[table_id])


def _relative(path: Path) -> str:
    """A source path as it should appear in provenance: repo-relative, posix."""
    from src.utils.config import load_config

    root = Path(load_config("paths").require("project_root"))
    try:
        return path.resolve().relative_to(root).as_posix()
    except (ValueError, OSError):
        return path.as_posix()


def _read(path: Path) -> Any:
    import pandas as pd

    if not path.is_file():
        raise FileNotFoundError(
            "source CSV not found: "
            + str(path)
            + " -- the phase that writes it has not run in this checkout"
        )
    return pd.read_csv(path)


def _supervised(frame: Any, source: Path) -> Any:
    if "scope" not in frame.columns:
        return frame
    subset = frame[frame["scope"] == SUPERVISED_SCOPE]
    if subset.empty:
        raise ValueError(
            str(source)
            + " has no rows with scope="
            + SUPERVISED_SCOPE
            + " -- refusing to fall back to all_records, which would silently"
            " report 3,541 PhysioNet files where 3,240 were modelled"
        )
    return subset


# ---------------------------------------------------------------------------
# T01 -- Dataset Inventory (DA-01)
# ---------------------------------------------------------------------------


def build_t01(command: str = "") -> Table:
    source = audit_dir() / "dataset_inventory.csv"
    frame = _read(source)
    spec = TableSpec(
        table_id="T01",
        title="Dataset Inventory",
        caption=(
            "The four public PCG corpora used by PV-MEPCG / PulseVision, as audited "
            "against the files on disk. 'Usable' counts the labeled recordings that "
            "enter modelling; 'Files' counts everything in the folder, including "
            "unlabelled records. Where these figures differ from the published "
            "description of a corpus, the audited figure is reported."
        ),
        sources=(_relative(source),),
        columns=(
            Column("dataset_source", header="ID", kind="text"),
            Column("dataset_name", header="Dataset", kind="text"),
            Column("version", header="Version", kind="text"),
            Column("folder", header="Folder", kind="text"),
            Column("total_files", header="Files", kind="count"),
            Column("usable_files", header="Usable", kind="count"),
            Column("n_classes", header="Classes", kind="count"),
            Column("n_subjects", header="Subjects", kind="count"),
            Column("subject_id_origin", header="Subject IDs", kind="text"),
            Column("original_fs", header="Native fs (Hz)", kind="count"),
            Column("target_fs", header="Target fs (Hz)", kind="count"),
            Column("total_hours", header="Hours (all files)", kind="seconds"),
            Column("role", header="Role", kind="text"),
        ),
        objective="O1 (corpus definition)",
        notes=(
            "Subject IDs are native for CirCor, derived for PhysioNet and PASCAL "
            "set_b, and unavailable for PASCAL set_a, which is grouped at record "
            "level only.",
            "Hours are over ALL files in the folder, including the unlabelled "
            "records counted under Files. The modelled subset is Usable.",
        ),
        command=command,
    )
    return build_table(spec, frame)


# ---------------------------------------------------------------------------
# T02 -- Class Distribution and Imbalance Ratio (DA-02)
# ---------------------------------------------------------------------------


def build_t02(command: str = "") -> Table:
    source = audit_dir() / "class_distribution.csv"
    frame = _supervised(_read(source), source)
    spec = TableSpec(
        table_id="T02",
        title="Class Distribution and Imbalance Ratio",
        caption=(
            "Class counts for the five separate label spaces, over the supervised "
            "(labeled) records only. The five tasks are never merged: binary, "
            "PASCAL A four-class, PASCAL B three-class, CirCor murmur and CirCor "
            "outcome each have their own target. Imbalance ratio is relative to "
            "the largest class within the same task."
        ),
        sources=(_relative(source),),
        columns=(
            Column("dataset_source", header="ID", kind="text"),
            Column("dataset_name", header="Dataset", kind="text"),
            Column("task", header="Task", kind="text"),
            Column("class", header="Class", kind="text"),
            Column("n_records", header="Records", kind="count"),
            Column("share", header="Share", kind="metric"),
            Column("imbalance_ratio", header="Imbalance", kind="metric"),
            Column("n_subjects", header="Subjects", kind="count"),
        ),
        objective="O1 (corpus definition)",
        notes=(
            "PASCAL's 'artifact' class is a recording-quality label, not a cardiac "
            "class. The four-class model is not a four-class cardiac classifier.",
        ),
        command=command,
    )
    return build_table(spec, frame)


# ---------------------------------------------------------------------------
# T03 -- Recording Duration and Sampling Summary (DA-03 + DA-04)
# ---------------------------------------------------------------------------


def build_t03(command: str = "") -> Table:
    duration_source = audit_dir() / "recording_duration_summary.csv"
    sampling_source = audit_dir() / "sampling_rate_summary.csv"

    durations = _supervised(_read(duration_source), duration_source)
    durations = durations[durations["class"] == "ALL"]
    if durations.empty:
        raise ValueError(str(duration_source) + " has no class=ALL summary rows")
    sampling = _read(sampling_source)

    merged = durations.merge(
        sampling[
            [
                "dataset_source",
                "original_fs",
                "converted_fs",
                "conversion",
                "factor",
                "n_supervised",
            ]
        ],
        on="dataset_source",
        how="inner",
        validate="one_to_one",
    )
    # An inner join that quietly drops a dataset would produce a table that is
    # correct about every row it shows and wrong about the corpus.
    if len(merged) != len(durations):
        lost = sorted(set(durations["dataset_source"]) - set(merged["dataset_source"]))
        raise ValueError(
            "T03: joining DA-03 to DA-04 lost dataset(s) "
            + ", ".join(lost)
            + " -- the two audit files disagree about which corpora exist"
        )
    # DA-03's supervised row count and DA-04's n_supervised are independently
    # derived. If they disagree, one of the two audit files is stale and the
    # table would report a duration distribution over a different set of
    # records than the one it names.
    disagree = merged[merged["n"] != merged["n_supervised"]]
    if len(disagree):
        raise ValueError(
            "T03: DA-03 and DA-04 disagree on the supervised record count for "
            + ", ".join(str(d) for d in disagree["dataset_source"])
        )
    merged = merged.drop(columns=["n_supervised"])

    spec = TableSpec(
        table_id="T03",
        title="Recording Duration and Sampling Summary",
        caption=(
            "Recording length in seconds over the supervised records of each "
            "corpus, and the sampling rate conversion applied to it. All four "
            "corpora are resampled to a common 2 kHz working rate; the durations "
            "are of the original recordings and are unchanged by resampling. "
            "Total corpus hours are in T01 and cover all files, not only the "
            "supervised subset summarised here."
        ),
        sources=(_relative(duration_source), _relative(sampling_source)),
        columns=(
            Column("dataset_source", header="ID", kind="text"),
            Column("dataset_name", header="Dataset", kind="text"),
            Column("n", header="Records", kind="count"),
            Column("min", header="Min (s)", kind="seconds"),
            Column("p25", header="P25 (s)", kind="seconds"),
            Column("median", header="Median (s)", kind="seconds"),
            Column("p75", header="P75 (s)", kind="seconds"),
            Column("max", header="Max (s)", kind="seconds"),
            Column("mean", header="Mean (s)", kind="seconds"),
            Column("sd", header="SD (s)", kind="seconds"),
            Column("original_fs", header="Native fs (Hz)", kind="count"),
            Column("converted_fs", header="Working fs (Hz)", kind="count"),
            Column("conversion", header="Conversion", kind="text"),
        ),
        objective="O1 (corpus definition)",
        notes=(
            "The duration extremes are real and both are handled by the extractor: "
            "PASCAL set_b reaches 0.76 s, too short for a five-level DWT, and "
            "PhysioNet reaches 122.0 s.",
        ),
        command=command,
    )
    return build_table(spec, merged)


# ---------------------------------------------------------------------------
# T04 -- Preprocessing Configuration (PP-07)
# ---------------------------------------------------------------------------


def build_t04(command: str = "") -> Table:
    source = outputs_dir("preprocessing") / "preprocessing_settings.csv"
    frame = _read(source)
    spec = TableSpec(
        table_id="T04",
        title="Preprocessing Configuration",
        caption=(
            "Every preprocessing setting applied before feature extraction, with "
            "the configuration key it is read from. Values are reproduced exactly "
            "as configured and are not rounded: a rounded filter cut-off is a "
            "different filter."
        ),
        sources=(_relative(source),),
        columns=(
            Column("stage", header="Stage", kind="text"),
            Column("setting", header="Setting", kind="text"),
            Column("value", header="Value", kind="text"),
            Column("unit", header="Unit", kind="text"),
            Column("config_key", header="Config key", kind="text"),
            Column("notes", header="Notes", kind="text"),
        ),
        objective="O3 (preprocessing pipeline)",
        command=command,
    )
    return build_table(spec, frame)


# ---------------------------------------------------------------------------
# T05 -- Feature Inventory and Counts (FE-01 + FE-02)
# ---------------------------------------------------------------------------


def build_t05(command: str = "") -> Table:
    inventory_source = outputs_dir("features") / "feature_inventory.csv"
    family_source = outputs_dir("features") / "feature_family_summary.csv"

    inventory = _read(inventory_source)
    families = _read(family_source)

    # Recomputed from the 138 rows rather than copied from FE-02, so that the
    # table is a check on the locked registry rather than a restatement of it.
    counted = (
        inventory.groupby("family", sort=False)
        .agg(counted_features=("name", "size"), first_index=("index", "min"))
        .reset_index()
    )
    declared = families[families["family"] != "TOTAL"]
    merged = declared.merge(counted, on="family", how="outer", validate="one_to_one")

    mismatched = merged[merged["n_features"] != merged["counted_features"]]
    if len(mismatched):
        raise ValueError(
            "T05: FE-02 disagrees with FE-01 on family size(s) "
            + ", ".join(str(f) for f in mismatched["family"])
            + " -- the feature registry and the extracted inventory are out of sync"
        )
    total = int(counted["counted_features"].sum())
    if total != 138:
        raise ValueError("T05: FE-01 holds " + str(total) + " features, not the locked 138")

    merged = merged[
        [
            "family",
            "extractor",
            "counted_features",
            "expected_count",
            "matches_expected",
            "first_index_x",
        ]
    ].rename(columns={"first_index_x": "first_index"})
    # The registry order (time, frequency, MFCC, chroma, DWT, envelope) is the
    # locked column order of the 138-vector. The merge above returns families
    # alphabetically, which reads as if the order were arbitrary.
    merged = merged.sort_values("first_index").reset_index(drop=True)

    spec = TableSpec(
        table_id="T05",
        title="Feature Inventory and Counts",
        caption=(
            "The locked 138-feature registry by family. 'Extracted' is counted "
            "from the feature inventory itself; 'Declared' is the count the "
            "registry states. The two are asserted equal at build time, so a "
            "silent drift between the registry and the extractor cannot reach a "
            "published table."
        ),
        sources=(_relative(inventory_source), _relative(family_source)),
        columns=(
            Column("family", header="Family", kind="text"),
            Column("extractor", header="Extractor", kind="text"),
            Column("counted_features", header="Extracted", kind="count"),
            Column("expected_count", header="Declared", kind="count"),
            Column("matches_expected", header="Agrees", kind="text"),
            Column("first_index", header="First index", kind="count"),
        ),
        objective="O4 (feature engineering)",
        notes=(
            "138 = time 24 + frequency 22 + MFCC 39 + chroma 24 + DWT 24 + "
            "envelope 5. Column order is a literal in "
            "src/feature_extraction/registry.py and is never derived; the full "
            "per-feature listing with equations is in "
            "outputs/03_features/feature_inventory.csv.",
        ),
        command=command,
    )
    return build_table(spec, merged)


# ---------------------------------------------------------------------------
# T06 -- Model Hyperparameter Configuration
# ---------------------------------------------------------------------------


def _search_space_text(rows: Any) -> str:
    """``C in log_uniform[0.001, 1000.0]; l1_ratio in {0.0, 0.5, 1.0}``."""
    import pandas as pd

    parts: list[str] = []
    for row in rows.itertuples(index=False):
        if str(row.kind) == "categorical":
            choices = str(row.choices).replace("|", ", ")
            parts.append(str(row.parameter) + " in {" + choices + "}")
        else:
            low = "" if pd.isna(row.low) else str(row.low)
            high = "" if pd.isna(row.high) else str(row.high)
            parts.append(
                str(row.parameter) + " in " + str(row.kind) + "[" + low + ", " + high + "]"
            )
    return "; ".join(parts)


def build_t06(command: str = "") -> Table:
    import pandas as pd

    registry_source = outputs_dir("models") / "model_registry.csv"
    spaces_source = outputs_dir("models") / "model_search_spaces.csv"

    registry = _read(registry_source).copy()
    spaces = _read(spaces_source)

    by_model = {
        str(model_id): _search_space_text(block)
        for model_id, block in spaces.groupby("model_id", sort=False)
    }
    registry["search_space"] = [
        by_model.get(str(model_id), "not searched") for model_id in registry["model_id"]
    ]
    registry["members"] = [
        "n/a" if pd.isna(members) else str(members).replace("|", " + ")
        for members in registry["members"]
    ]

    spec = TableSpec(
        table_id="T06",
        title="Model Hyperparameter Configuration",
        caption=(
            "The nine declared models, their estimators and the hyperparameter "
            "space each one is searched over. M9 (1D-CNN) is declared with "
            "in_scope=false rather than omitted, so the registry shows an explicit "
            "exclusion rather than a gap; no claim in this work compares "
            "PV-MEPCG / PulseVision against a convolutional network."
        ),
        sources=(_relative(registry_source), _relative(spaces_source)),
        columns=(
            Column("model_id", header="ID", kind="text"),
            Column("name", header="Model", kind="text"),
            Column("estimator", header="Estimator", kind="text"),
            Column("mandatory", header="Mandatory", kind="text"),
            Column("implemented", header="Implemented", kind="text"),
            Column("is_ensemble", header="Ensemble", kind="text"),
            Column("members", header="Members", kind="text"),
            Column("calibrated", header="Calibrated", kind="text"),
            Column("n_search_dimensions", header="Search dims", kind="count"),
            Column("search_space", header="Search space", kind="text"),
            Column("unavailable_reason", header="Exclusion reason", kind="text"),
        ),
        objective="O5 (model configuration)",
        command=command,
    )
    return build_table(spec, registry)


# ---------------------------------------------------------------------------
# T07 -- Search Space and Best Parameters
# ---------------------------------------------------------------------------


def build_t07(command: str = "") -> Table:
    source = outputs_dir("search_optimization") / "search_space_and_best_parameters.csv"
    frame = _read(source)
    spec = TableSpec(
        table_id="T07",
        title="Search Space and Best Parameters",
        caption=(
            "The hyperparameter space searched for each model and the value each "
            "search selected, on outer fold r0f0. SO-01 is randomised search and "
            "SO-02 is Bayesian optimisation; the final column records which search "
            "supplied the shipped value. Margins between the two are 0.1-0.5 points "
            "on n=3 pairs, so the claim is no significant difference at this "
            "budget, never that one method is better."
        ),
        sources=(_relative(source),),
        columns=(
            Column("model_id", header="ID", kind="text"),
            Column("parameter", header="Parameter", kind="text"),
            Column("distribution", header="Distribution", kind="text"),
            Column("range_or_choices", header="Range / choices", kind="text"),
            Column("so_01_selected", header="SO-01 value", kind="text"),
            Column("so_01_score", header="SO-01 score", kind="metric"),
            Column("so_02_selected", header="SO-02 value", kind="text"),
            Column("so_02_score", header="SO-02 score", kind="metric"),
            Column("final_selected", header="Shipped value", kind="text"),
            Column("final_source", header="From", kind="text"),
        ),
        objective="O5 (search optimisation)",
        notes=(
            "Hyperparameter values are reproduced at full precision and are NOT "
            "rounded to three decimals: the value is what reproduces the run "
            "(seed 42), and a rounded C or gamma cannot be pasted back into a "
            "config. Scores in this table are metrics and do follow the "
            "three-decimal rule.",
        ),
        command=command,
    )
    return build_table(spec, frame)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

_BUILDERS = {
    "T01": build_t01,
    "T02": build_t02,
    "T03": build_t03,
    "T04": build_t04,
    "T05": build_t05,
    "T06": build_t06,
    "T07": build_t07,
}


def build_setup_tables(
    table_ids: tuple[str, ...] = SETUP_TABLE_IDS, *, command: str = ""
) -> dict[str, Table]:
    """Build the requested setup tables without writing anything."""
    built: dict[str, Table] = {}
    for table_id in table_ids:
        if table_id not in _BUILDERS:
            raise KeyError("unknown setup table: " + table_id)
        built[table_id] = _BUILDERS[table_id](command)
        log.info("%s built (%d rows)", table_id, len(built[table_id].frame))
    return built
