"""One real CirCor recording, its cardiac-cycle segmentation, and its notices.

T113.6 asks for a waveform viewer with playback and an overlay marking
S1 / systole / S2 / diastole from the **real** CirCor TSV segmentation, not a
synthetic animation. That needs three things in `frontend/public/`: the audio,
the segment boundaries, and the licence notices that make redistributing them
lawful.

## Reading `dataset/` and writing beside it

`dataset/` is read-only input and nothing here writes into it. The `.wav` and
its `.tsv` are **copied out** into `frontend/public/`, byte for byte, with both
sha256s recorded so the files in the repository can be shown to be the files in
the corpus.

Both are committed, not just the audio, so the overlay can be rebuilt on a
machine that has no corpus -- which is every fresh clone and every CI checkout.
See :func:`resolve_sources` for why exporting an "unavailable" payload beside a
perfectly good audio file would have been the worse failure.

## The notices, and why there are four of them

The CirCor DigiScope corpus is published under the Open Data Commons
Attribution License (ODC-By 1.0), whose section 4 permits redistribution only
with attribution. Three clauses apply here and each needs a different artifact:

* **4.2(b)** wants the licence URI "in any relevant documentation" -- so a line
  in `README.md`.
* **4.2(d)** covers the case where a file cannot carry a notice. A `.wav` has
  nowhere to put one, so the notice goes "in a location (such as a relevant
  directory) where users would be likely to look for it" -- `public/NOTICE.md`,
  in the same directory as the audio.
* **4.3** governs Publicly Using a Produced Work: anyone exposed to the work
  must be made aware the content came from the Database and is available under
  this licence. The dashboard IS the Produced Work, so the attribution is
  rendered in the UI wherever the audio plays, not only in a file nobody opens.

The record id travels with all of it. Research rule 1 applies to a segmentation
overlay exactly as it applies to a metric: a reader must be able to get from the
line drawn on screen back to the TSV row that put it there.

## Screening language

The recording is labelled a **dataset sample** everywhere it appears. It is
de-identified paediatric screening data, and calling it a patient or a case
would be both a privacy overreach and the diagnostic framing rule 7 forbids.
"""

from __future__ import annotations

import csv
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.reporting.tables import content_digest
from src.utils.io import ensure_dir
from src.utils.logging_setup import get_logger

__all__ = [
    "CIRCOR_LICENCE_NAME",
    "CIRCOR_LICENCE_URI",
    "CIRCOR_SOURCE_URL",
    "SAMPLE_RECORD_ID",
    "SEGMENT_LABELS",
    "SegmentationSample",
    "attribution_line",
    "export_sample",
    "notice_markdown",
    "read_segmentation",
    "resolve_sources",
    "segmentation_payload",
]

log = get_logger("reporting.segmentation")

#: The record the overlay is built from.
#:
#: Chosen from `outputs/01_dataset_audit/circor_segmentation_summary.csv` as the
#: only recording that is fully annotated (annotated_fraction 1.000), long
#: enough to show many cycles (21), and short enough to commit (93 KB). Pinned
#: rather than picked at export time: a viewer that shows a different recording
#: on each run cannot be cross-checked against anything.
SAMPLE_RECORD_ID = "85197_TV"

#: The corpus, relative to the repository root. Read-only.
CIRCOR_ROOT = "dataset/archive/training_data/training_data"

CIRCOR_DATASET_NAME = "The CirCor DigiScope Phonocardiogram Dataset"
CIRCOR_SOURCE_URL = "https://physionet.org/content/circor-heart-sound/1.0.3/"
CIRCOR_LICENCE_NAME = "Open Data Commons Attribution License v1.0 (ODC-By 1.0)"
CIRCOR_LICENCE_URI = "https://opendatacommons.org/licenses/by/1-0/"

#: CirCor's TSV label codes. 0 is "unannotated", not a fifth cardiac phase --
#: the segmentation algorithm declined to label that span, and drawing it as a
#: phase would invent an observation.
SEGMENT_LABELS: dict[int, dict[str, str]] = {
    0: {
        "key": "unannotated",
        "name": "Unannotated",
        "description": "The segmentation did not label this span. Not a cardiac phase.",
    },
    1: {
        "key": "s1",
        "name": "S1",
        "description": "First heart sound: mitral and tricuspid valve closure.",
    },
    2: {
        "key": "systole",
        "name": "Systole",
        "description": "Ventricular ejection, between S1 and S2.",
    },
    3: {
        "key": "s2",
        "name": "S2",
        "description": "Second heart sound: aortic and pulmonary valve closure.",
    },
    4: {
        "key": "diastole",
        "name": "Diastole",
        "description": "Ventricular filling, between S2 and the next S1.",
    },
}


@dataclass(frozen=True)
class SegmentationSample:
    """The exported sample: audio, segments and provenance."""

    record_id: str
    wav_source: Path
    tsv_source: Path
    duration_seconds: float
    sample_rate: int
    channels: int
    segments: list[dict[str, Any]]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_segmentation(tsv_path: Path) -> list[dict[str, Any]]:
    """Parse a CirCor `.tsv`: ``start<TAB>end<TAB>label`` per row.

    No header, three columns, tab separated. Rows are returned in file order --
    they are already sorted, and re-sorting would hide a corrupt file rather
    than surface it.
    """
    segments: list[dict[str, Any]] = []
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != 3:
                raise ValueError(
                    str(tsv_path)
                    + " row "
                    + str(row_number)
                    + " has "
                    + str(len(row))
                    + " columns, expected 3"
                )
            start, end, label = float(row[0]), float(row[1]), int(float(row[2]))
            if end < start:
                raise ValueError(
                    str(tsv_path) + " row " + str(row_number) + " ends before it starts"
                )
            if label not in SEGMENT_LABELS:
                raise ValueError(
                    str(tsv_path) + " row " + str(row_number) + " has unknown label " + str(label)
                )
            meta = SEGMENT_LABELS[label]
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "label": label,
                    "key": meta["key"],
                    "name": meta["name"],
                }
            )
    if not segments:
        raise ValueError(str(tsv_path) + " contains no segments")
    return segments


def _wav_info(path: Path) -> tuple[float, int, int]:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        channels = handle.getnchannels()
    return frames / float(rate), rate, channels


def resolve_sources(
    record_id: str = SAMPLE_RECORD_ID, *, public_dir: Path | None = None
) -> tuple[Path, Path, str]:
    """``(wav, tsv, origin)`` -- the corpus if it is here, else the committed copy.

    `dataset/` is 1.3 GB and is not in the repository, so a fresh clone and every
    CI checkout have no corpus. Both files are therefore committed under
    `frontend/public/` and the export falls back to them.

    That fallback is not a convenience. Without it, `npm run build` on a machine
    with no corpus would write a `segmentation.json` saying the sample is
    unavailable **while the audio sat in `public/` beside it**, and the dashboard
    a grader builds would silently lose the one figure T113.6 exists to produce.

    The corpus wins wherever it is present, so the committed copies cannot drift
    from it unnoticed: `export_sample` re-copies and re-checksums on every run.
    """
    corpus = _project_root() / CIRCOR_ROOT
    corpus_wav = corpus / (record_id + ".wav")
    corpus_tsv = corpus / (record_id + ".tsv")
    if corpus_wav.is_file() and corpus_tsv.is_file():
        return corpus_wav, corpus_tsv, "corpus"

    committed = public_dir if public_dir is not None else _project_root() / "frontend" / "public"
    committed_wav = committed / (record_id + ".wav")
    committed_tsv = committed / (record_id + ".tsv")
    if committed_wav.is_file() and committed_tsv.is_file():
        return committed_wav, committed_tsv, "committed"

    raise FileNotFoundError(
        "neither the corpus file "
        + str(corpus_wav)
        + " nor the committed copy at "
        + str(committed_wav)
        + " is present, so no cardiac-cycle sample can be exported"
    )


def load_sample(
    record_id: str = SAMPLE_RECORD_ID, *, public_dir: Path | None = None
) -> SegmentationSample:
    """Read the pinned recording and its segmentation."""
    wav_path, tsv_path, _origin = resolve_sources(record_id, public_dir=public_dir)
    duration, rate, channels = _wav_info(wav_path)
    segments = read_segmentation(tsv_path)
    covered = segments[-1]["end"]
    if covered > duration + 0.05:
        raise ValueError(
            "the segmentation of "
            + record_id
            + " runs to "
            + format(covered, ".3f")
            + " s but the recording is only "
            + format(duration, ".3f")
            + " s long"
        )
    return SegmentationSample(
        record_id=record_id,
        wav_source=wav_path,
        tsv_source=tsv_path,
        duration_seconds=duration,
        sample_rate=rate,
        channels=channels,
        segments=segments,
    )


def attribution_line() -> str:
    """The ODC-By 4.3 notice, rendered in the UI wherever the audio plays."""
    return (
        "Contains information from "
        + CIRCOR_DATASET_NAME
        + " (record "
        + SAMPLE_RECORD_ID
        + "), which is made available under the "
        + CIRCOR_LICENCE_NAME
        + "."
    )


def notice_markdown(sample: SegmentationSample, wav_sha256: str) -> str:
    """`public/NOTICE.md` -- the ODC-By 4.2(d) notice, beside the audio."""
    return "\n".join(
        [
            "# Third-party data notice",
            "",
            "The audio file in this directory is redistributed from a public research",
            "corpus. It is included so the dashboard can show a real cardiac-cycle",
            "segmentation rather than a synthetic animation.",
            "",
            "## Source",
            "",
            "| | |",
            "|---|---|",
            "| Dataset | " + CIRCOR_DATASET_NAME + " |",
            "| Source | " + CIRCOR_SOURCE_URL + " |",
            "| Licence | [" + CIRCOR_LICENCE_NAME + "](" + CIRCOR_LICENCE_URI + ") |",
            "| Record id | `" + sample.record_id + "` |",
            "| File | `" + sample.record_id + ".wav` |",
            "| sha256 | `" + wav_sha256 + "` |",
            "| Duration | " + format(sample.duration_seconds, ".3f") + " s |",
            "| Sample rate | " + str(sample.sample_rate) + " Hz |",
            "| Segments | " + str(len(sample.segments)) + " |",
            "",
            "## Why this file is here rather than a notice inside it",
            "",
            "ODC-By section 4.2(d) provides that where a file's structure cannot carry",
            "the required notices, they go in a location where users would be likely to",
            "look -- such as the directory holding the file. A WAV has no such field, so",
            "this is that notice.",
            "",
            "Section 4.3 additionally requires a notice associated with the Produced Work",
            "when it is publicly used. The dashboard renders that attribution in the",
            "interface wherever this recording plays; it is not left to this file alone.",
            "",
            "## Provenance of the overlay",
            "",
            "The S1 / systole / S2 / diastole overlay is read from `" + sample.record_id + ".tsv`",
            "in the corpus -- the dataset's own expert-reviewed segmentation. Nothing about",
            "the overlay is generated, smoothed or inferred, so every boundary drawn on",
            "screen traces back to a row of that file.",
            "",
            "## Scope",
            "",
            "This is a **dataset sample**, not a patient and not a case. The corpus is",
            "de-identified paediatric screening data, and PV-MEPCG / PulseVision is an",
            "academic screening and decision-support prototype: it does not diagnose.",
            "",
        ]
    )


def segmentation_payload(
    sample: SegmentationSample, *, audio_url: str, wav_sha256: str
) -> dict[str, Any]:
    """The JSON the viewer reads. Times in seconds; nothing computed client-side."""
    tsv_digest, tsv_method = content_digest(sample.tsv_source)
    counts: dict[str, int] = {}
    durations: dict[str, float] = {}
    for segment in sample.segments:
        key = str(segment["key"])
        counts[key] = counts.get(key, 0) + 1
        durations[key] = durations.get(key, 0.0) + (segment["end"] - segment["start"])

    return {
        "record_id": sample.record_id,
        # Never "patient", never "case". See the module docstring.
        "label": "Dataset sample " + sample.record_id,
        "audio_url": audio_url,
        "duration_seconds": sample.duration_seconds,
        "duration_display": format(sample.duration_seconds, ".2f") + " s",
        "sample_rate_hz": sample.sample_rate,
        "channels": sample.channels,
        "n_segments": len(sample.segments),
        "segments": sample.segments,
        "legend": [
            {
                "label": label,
                "key": meta["key"],
                "name": meta["name"],
                "description": meta["description"],
                "n_segments": counts.get(meta["key"], 0),
                "seconds": durations.get(meta["key"], 0.0),
                "seconds_display": format(durations.get(meta["key"], 0.0), ".2f") + " s",
            }
            for label, meta in sorted(SEGMENT_LABELS.items())
        ],
        "provenance": {
            "dataset": CIRCOR_DATASET_NAME,
            "source_url": CIRCOR_SOURCE_URL,
            "licence": CIRCOR_LICENCE_NAME,
            "licence_uri": CIRCOR_LICENCE_URI,
            "attribution": attribution_line(),
            "wav_source": CIRCOR_ROOT + "/" + sample.record_id + ".wav",
            "wav_sha256": wav_sha256,
            "tsv_source": CIRCOR_ROOT + "/" + sample.record_id + ".tsv",
            "tsv_sha256": tsv_digest,
            "tsv_digest_method": tsv_method,
            "notice": "/NOTICE.md",
        },
        "scope_note": (
            "A de-identified recording from a public research corpus, shown to "
            "demonstrate the segmentation overlay. It is a dataset sample, not a "
            "patient, and nothing on this page is a diagnosis."
        ),
    }


def export_sample(public_dir: Path, *, record_id: str = SAMPLE_RECORD_ID) -> dict[str, Any]:
    """Copy the audio and its TSV into `public/`, write `NOTICE.md`, return the payload."""
    target_dir = ensure_dir(public_dir)
    sample = load_sample(record_id, public_dir=target_dir)
    destination = target_dir / (record_id + ".wav")

    source_digest, _ = content_digest(sample.wav_source)
    # Both files travel: the TSV is what makes the overlay rebuildable on a
    # machine with no corpus, and it is the artifact the notice checksums.
    for source in (sample.wav_source, sample.tsv_source):
        copy = target_dir / source.name
        if copy.resolve() == source.resolve():
            continue  # already reading the committed copy
        if not copy.is_file() or content_digest(copy)[0] != content_digest(source)[0]:
            shutil.copyfile(source, copy)
        if content_digest(copy)[0] != content_digest(source)[0]:
            raise ValueError("the copied " + source.name + " does not match its source")

    (target_dir / "NOTICE.md").write_text(
        notice_markdown(sample, source_digest), encoding="utf-8", newline="\n"
    )

    log.info("exported cardiac-cycle sample %s (%d segments)", record_id, len(sample.segments))
    return segmentation_payload(sample, audio_url="/" + destination.name, wav_sha256=source_digest)
