"""Phase 101 gate: the Objective 2 literature review and its indicative comparison.

T101.7 asks three things, each checked here against the declared studies and the
tables as written:

* the review covers the ten synopsis references **and** the challenge
  benchmarks (both the 2016 and the 2022 Challenge);
* every row has a positioning entry saying how PV-MEPCG differs -- and none of
  them claims to beat anything;
* the comparison carries the caveat that split protocols differ and the
  comparison is indicative, not head-to-head.

It also checks what T101.6 implies: PV-MEPCG's rows in LIT-02 are the numbers in
its result files, not typed copies of them.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from src.reporting import literature as lit

ROOT = Path(__file__).resolve().parents[1]

#: T101.1's schema, field by field, mapped to the Study attribute holding it.
SCHEMA = {
    "author": "authors",
    "year": "year",
    "dataset": "dataset",
    "preprocessing": "preprocessing",
    "feature families": "features",
    "classifier": "classifier",
    "validation scheme": "validation",
    "reported metric": "reported_metric",
    "sample size": "sample_size",
    "stated limitation": "limitation",
}


def _of(group: str) -> list[lit.Study]:
    return [study for study in lit.STUDIES if study.group == group]


def test_the_schema_holds_every_t101_1_field() -> None:
    declared = {field.name for field in fields(lit.Study)}
    assert set(SCHEMA.values()) <= declared
    assert {"positioning", "verification", "source_url"} <= declared


def test_every_synopsis_reference_has_a_row() -> None:
    labels = [study.synopsis_label for study in _of("synopsis")]
    assert sorted(labels) == sorted(lit.SYNOPSIS_REFERENCES)
    assert len(lit.SYNOPSIS_REFERENCES) == 10


def test_both_challenges_are_benchmarked() -> None:
    assert len(_of("challenge_2016")) >= 3, "the 2016 leading entries"
    track_2022 = " ".join(study.title for study in _of("challenge_2022")).lower()
    assert "murmur" in track_2022 and "outcome" in track_2022


def test_recent_work_covers_ensembles_selection_and_deep_learning() -> None:
    corpus = " ".join(
        (study.classifier + " " + study.features + " " + study.title).lower()
        for study in lit.STUDIES
        if study.group in ("recent", "challenge_2016")
    )
    for topic in ("ensemble", "selection", "neural network", "domain"):
        assert topic in corpus, topic
    assert max(int(study.year) for study in _of("recent")) >= 2020


def test_every_row_is_complete_and_positioned() -> None:
    assert len({study.study_id for study in lit.STUDIES}) == len(lit.STUDIES)
    for study in lit.STUDIES:
        for name in (*SCHEMA.values(), "positioning", "verification", "title", "venue"):
            assert str(getattr(study, name)).strip(), study.study_id + " has no " + name
        assert "PV-MEPCG" in study.positioning, study.study_id
        assert study.source_url.startswith("https://"), study.study_id
        assert study.limitation_source in ("stated", "observed", "not available")


def test_no_positioning_entry_claims_superiority() -> None:
    for study in lit.STUDIES:
        text = study.positioning.lower()
        for claim in ("outperform", "better than", "beats", "superior", "state-of-the-art result"):
            assert claim not in text, study.study_id + ": " + claim


def test_every_published_value_belongs_to_a_sourced_study() -> None:
    by_id = {study.study_id: study for study in lit.STUDIES}
    for item in lit.PUBLISHED_2016:
        assert item.study_id in by_id
        assert item.source_url.startswith("https://")
        assert 0.0 < item.macc < 1.0


# ---------------------------------------------------------------------------
# the tables as written
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Path]]:
    for relative in (lit.AGGREGATE, lit.SELECTION, lit.LOSO):
        if not (ROOT / relative).is_file():
            pytest.skip(relative + " is not present in this checkout")
    target = tmp_path_factory.mktemp("literature")
    return lit.write_literature_tables(target, evidence_index=target / "evidence_index.csv")


def test_lit01_is_what_objective_2_looks_for(exported) -> None:
    csv_path = exported["LIT-01"]["csv"]
    assert "literature" in csv_path.name  # the glob T29 re-derives objective 2 from
    import pandas as pd

    frame = pd.read_csv(csv_path, keep_default_na=False)
    assert len(frame) == len(lit.STUDIES)
    assert (frame["positioning"].str.len() > 0).all()


def test_lit02_carries_the_split_protocol_caveat(exported) -> None:
    from docx import Document

    table = lit.build_lit02()
    notes = " ".join(table.spec.notes).lower()
    assert "split protocols differ" in notes and "indicative, not head-to-head" in notes
    assert "indicative, not head-to-head" in table.spec.caption.lower()
    written = exported["LIT-02"]
    for kind in ("md", "latex"):
        assert "head-to-head" in written[kind].read_text(encoding="utf-8").lower(), kind
    docx_text = "\n".join(p.text for p in Document(str(written["docx"])).paragraphs).lower()
    assert "head-to-head" in docx_text


def test_lit02_pv_mepcg_rows_are_the_result_files(exported) -> None:
    import pandas as pd

    final = str(pd.read_csv(ROOT / lit.SELECTION).sort_values("rank").iloc[0]["model_id"])
    aggregate = pd.read_csv(ROOT / lit.AGGREGATE).set_index("model_id").loc[final]
    loso = pd.read_csv(ROOT / lit.LOSO).set_index("model_id").loc[final]
    frame = lit.build_lit02().frame
    ours = frame[frame["origin"] == "this project, generated"].reset_index(drop=True)
    assert len(ours) == 2
    assert ours.loc[0, "sensitivity"] == pytest.approx(float(aggregate["sensitivity_mean"]))
    assert ours.loc[0, "macc"] == pytest.approx(float(aggregate["balanced_accuracy_mean"]))
    assert ours.loc[1, "macc"] == pytest.approx(float(loso["balanced_accuracy_holdout"]))
