"""Audio integrity scan gate (T16.7).

The ``needs_data`` tests here are marked ``slow`` as a group: the scan decodes
8,368 files and takes about three minutes cold. It caches, so a second run in
the same session is seconds -- but a cold CI-style run is not, and pretending
otherwise would make the default suite unusable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.data_loader import integrity as ig

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def catalog() -> Any:
    from src.data_loader.catalog import build_catalog

    return build_catalog()


@pytest.fixture(scope="module")
def scan(catalog: Any) -> Any:
    """The record scan -- D1-D4 only, without the Heartbeat_Sound helper files."""
    table, _ = ig.scan_corpus(catalog)
    return table[table["dataset_source"] != "heartbeat_sound"].reset_index(drop=True)


# ===========================================================================
# T16.7 -- the gate
# ===========================================================================


@pytest.mark.needs_data
@pytest.mark.slow
def test_scan_covers_every_record(catalog: Any, scan: Any) -> None:
    """All 7,536 records across D1-D4."""
    assert len(catalog) == 7536
    assert len(scan) == 7536
    assert list(scan["record_uid"]) == list(catalog["record_uid"])
    assert scan["dataset_source"].value_counts().to_dict() == {
        "D1": 3541,
        "D4": 3163,
        "D3": 656,
        "D2": 176,
    }


@pytest.mark.needs_data
@pytest.mark.slow
def test_zero_unreadable_and_zero_zero_length(scan: Any) -> None:
    """T16.2/T16.7 -- matching the 2026-08-22 audit, re-verified by decoding."""
    assert int((~scan["readable"]).sum()) == 0
    assert (scan["error"] == "").all()
    assert int(scan["is_zero_length"].sum()) == 0
    assert int(scan["is_truncated"].sum()) == 0

    # Every file decoded to at least one sample and every header told the truth
    # about how many. The second half is the part a header-only audit cannot say.
    assert (scan["n_frames"] > 0).all()
    assert (scan["n_frames"] == scan["n_frames_header"]).all()
    assert (scan["n_channels"] == 1).all()
    assert (scan["bit_depth"] == 16).all()


@pytest.mark.needs_data
@pytest.mark.slow
def test_one_near_silent_record_is_flagged(scan: Any) -> None:
    """T16.3 -- and the threshold is the project's own -60 dBFS line.

    Exactly one record in the corpus is dead: an unlabelled set_a file peaking at
    -71 dBFS across its full nine seconds. It is pinned by name because the
    finding is the record, not the count -- a threshold change that silently
    swept it up with others would be a different result wearing the same number.
    """
    silent = scan[scan["is_silent"]]
    assert len(silent) == 1
    assert list(silent["record_uid"]) == ["D2_set_a_Aunlabelledtest__201106120928"]
    assert float(silent["peak"].iloc[0]) < 1e-3

    # Nothing in the corpus is a constant-value file.
    assert int(scan["is_constant"].sum()) == 0

    # The threshold is -60 dBFS, the same silence line configs/signal.yaml
    # already draws for the preprocessed signal in quality.silence_threshold_db.
    from src.utils.config import load_config

    thresholds = ig.load_thresholds()
    quality_db = float(load_config("signal").require("quality.silence_threshold_db"))
    assert thresholds.silence_peak == pytest.approx(10 ** (quality_db / 20), rel=1e-9)


@pytest.mark.needs_data
@pytest.mark.slow
def test_clipping_is_flagged_and_discriminating(scan: Any) -> None:
    """T16.4 -- 63 records exceed 1% full-scale samples.

    2,929 files in the corpus touch full scale at least once. Only 63 spend more
    than 1% of their samples there. Both numbers are asserted: a flag that fires
    on every file that ever reaches 1.0 would be useless, and one that fires on
    nothing would be broken.
    """
    clipped = scan[scan["is_clipped"]]
    assert len(clipped) == 63
    assert clipped["dataset_source"].value_counts().to_dict() == {
        "D4": 27,
        "D1": 18,
        "D3": 18,
    }
    assert (clipped["clipped_fraction"] > 0.01).all()
    assert int((scan["peak"] >= 1.0).sum()) > 2000


@pytest.mark.needs_data
@pytest.mark.slow
def test_every_record_has_a_label_and_every_label_a_file(catalog: Any, scan: Any) -> None:
    """T16.5 -- both directions, per dataset, and both come back empty."""
    coverage = ig.check_label_coverage(catalog, scan)
    assert len(coverage) == 0
    assert list(coverage.columns) == [
        "record_uid",
        "dataset_source",
        "task",
        "problem",
        "detail",
        "file_path",
    ]


@pytest.mark.needs_data
@pytest.mark.slow
def test_missing_corrupt_report_is_written_even_when_nearly_empty(
    catalog: Any, scan: Any, tmp_path: Path
) -> None:
    """T16.6/T16.7 -- DA-05 exists whatever it finds."""
    import pandas as pd

    coverage = ig.check_label_coverage(catalog, scan)
    target = ig.write_missing_corrupt_report(scan, coverage, tmp_path)
    assert target.is_file()
    assert target.name == "missing_corrupt_files.csv"

    report = pd.read_csv(target)
    assert list(report.columns) == [
        "record_uid",
        "dataset_source",
        "task",
        "problem",
        "detail",
        "file_path",
    ]
    # Clipping and the one silent record are the only problems in this corpus.
    assert set(report["problem"]) == {"clipped", "all_silent"}
    assert int((report["problem"] == "clipped").sum()) == 63
    assert int((report["problem"] == "all_silent").sum()) == 1
    for absent in ("unreadable", "zero_length", "truncated", "constant_value"):
        assert absent not in set(report["problem"])


@pytest.mark.needs_data
@pytest.mark.slow
def test_an_empty_report_still_has_its_header(tmp_path: Path) -> None:
    """The distinction the file exists to preserve.

    An absent ``missing_corrupt_files.csv`` cannot be told apart from a scan that
    never ran. One with a header and no rows says the check happened.
    """
    import pandas as pd

    empty_scan = pd.DataFrame(
        columns=[*ig.AUDIO_SCAN_COLUMNS, *ig.FLAG_COLUMNS]
    ).astype(dict.fromkeys(ig.FLAG_COLUMNS, bool))
    empty_scan["readable"] = empty_scan["readable"].astype(bool)
    empty_coverage = pd.DataFrame(
        columns=["record_uid", "dataset_source", "task", "problem", "detail", "file_path"]
    )
    target = ig.write_missing_corrupt_report(empty_scan, empty_coverage, tmp_path)
    assert target.is_file()
    assert target.read_text(encoding="utf-8").strip().splitlines() == [
        "record_uid,dataset_source,task,problem,detail,file_path"
    ]


# ===========================================================================
# pure-function tests -- no dataset required
# ===========================================================================


def _write_wav(path: Path, samples: Any, fs: int = 4000) -> Path:
    import soundfile as sf

    sf.write(str(path), samples, fs, subtype="PCM_16")
    return path


def test_scan_a_normal_file(tmp_path: Path) -> None:
    import numpy as np

    thresholds = ig.load_thresholds()
    signal = 0.5 * np.sin(2 * np.pi * 50 * np.arange(8000) / 4000)
    path = _write_wav(tmp_path / "ok.wav", signal.astype(np.float32))

    row, envelope = ig.scan_audio_file(path, thresholds, record_uid="R1")
    assert row["readable"] is True
    assert row["fs"] == 4000
    assert row["n_frames"] == 8000
    assert row["bit_depth"] == 16
    assert row["duration_sec"] == pytest.approx(2.0)
    assert row["peak"] == pytest.approx(0.5, abs=1e-3)
    assert row["raw_sha256"] and row["content_sha256"]
    assert envelope.shape == (thresholds.envelope_points,)


def test_scan_an_unreadable_file_does_not_raise(tmp_path: Path) -> None:
    """One corrupt file must not abort a scan of 8,368."""
    path = tmp_path / "broken.wav"
    path.write_bytes(b"RIFF____WAVEnonsense")

    row, envelope = ig.scan_audio_file(path, ig.load_thresholds(), record_uid="BAD")
    assert row["readable"] is False
    assert row["error"]
    assert envelope.shape == (ig.load_thresholds().envelope_points,)


def test_flags_are_derived_not_baked_in(tmp_path: Path) -> None:
    """apply_flags turns measurements into flags, so a threshold change is free."""
    import numpy as np
    import pandas as pd

    thresholds = ig.load_thresholds()
    rows = []
    for name, signal in (
        ("silent", np.zeros(4000, dtype=np.float32)),
        ("loud", np.ones(4000, dtype=np.float32)),
        ("normal", (0.3 * np.sin(np.linspace(0, 60, 4000))).astype(np.float32)),
    ):
        row, _ = ig.scan_audio_file(
            _write_wav(tmp_path / (name + ".wav"), signal),
            thresholds,
            record_uid=name,
            with_hashes=False,
        )
        rows.append(row)

    flagged = ig.apply_flags(pd.DataFrame(rows), thresholds)
    by_uid = flagged.set_index("record_uid")
    assert bool(by_uid.at["silent", "is_silent"])
    assert bool(by_uid.at["silent", "is_constant"])
    assert bool(by_uid.at["loud", "is_clipped"])
    assert not bool(by_uid.at["normal", "is_silent"])
    assert not bool(by_uid.at["normal", "is_clipped"])


def test_apply_flags_rejects_a_stale_clipping_threshold(tmp_path: Path) -> None:
    """The one measurement that bakes a threshold in must not go unnoticed."""
    import pandas as pd

    thresholds = ig.load_thresholds()
    frame = pd.DataFrame(
        [
            {
                "record_uid": "R1",
                "readable": True,
                "n_frames": 100,
                "n_frames_header": 100,
                "file_bytes": 244,
                "peak": 0.5,
                "variance": 0.1,
                "clipped_fraction": 0.0,
                "clipping_threshold": 0.5,   # not what the config says
            }
        ]
    )
    with pytest.raises(ValueError, match="counted full-scale samples at threshold"):
        ig.apply_flags(frame, thresholds)


def test_envelope_is_scale_invariant_and_fixed_length() -> None:
    """Two recordings of one signal at different levels must look identical."""
    import numpy as np

    points = 64
    signal = np.abs(np.sin(np.linspace(0, 20, 5000))).astype(np.float32)
    quiet = ig._envelope(signal * 0.01, points)
    loud = ig._envelope(signal * 0.9, points)

    assert quiet.shape == (points,)
    assert float(np.dot(quiet, loud)) == pytest.approx(1.0, abs=1e-5)

    # A signal with no shape returns zeros rather than dividing by ~0.
    assert not ig._envelope(np.ones(1000, dtype=np.float32), points).any()


def test_envelope_handles_a_signal_shorter_than_the_bin_count() -> None:
    """PASCAL B's shortest record is 0.76 s; the envelope must still be 256 long."""
    import numpy as np

    envelope = ig._envelope(np.sin(np.linspace(0, 6, 40)).astype(np.float32), 256)
    assert envelope.shape == (256,)
    assert np.isfinite(envelope).all()


def test_subtype_bits_covers_the_pcm_forms() -> None:
    assert ig.SUBTYPE_BITS["PCM_16"] == 16
    assert ig.SUBTYPE_BITS["PCM_24"] == 24
    assert ig.SUBTYPE_BITS["FLOAT"] == 32
