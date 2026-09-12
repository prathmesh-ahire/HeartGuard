# Third-party data notice

The audio file in this directory is redistributed from a public research
corpus. It is included so the dashboard can show a real cardiac-cycle
segmentation rather than a synthetic animation.

## Source

| | |
|---|---|
| Dataset | The CirCor DigiScope Phonocardiogram Dataset |
| Source | https://physionet.org/content/circor-heart-sound/1.0.3/ |
| Licence | [Open Data Commons Attribution License v1.0 (ODC-By 1.0)](https://opendatacommons.org/licenses/by/1-0/) |
| Record id | `85197_TV` |
| File | `85197_TV.wav` |
| sha256 | `c4d3ef2a01c5fb919edb6ef6699e146467e5177106b16f391485c42987c59f79` |
| Duration | 11.648 s |
| Sample rate | 4000 Hz |
| Segments | 81 |

## Why this file is here rather than a notice inside it

ODC-By section 4.2(d) provides that where a file's structure cannot carry
the required notices, they go in a location where users would be likely to
look -- such as the directory holding the file. A WAV has no such field, so
this is that notice.

Section 4.3 additionally requires a notice associated with the Produced Work
when it is publicly used. The dashboard renders that attribution in the
interface wherever this recording plays; it is not left to this file alone.

## Provenance of the overlay

The S1 / systole / S2 / diastole overlay is read from `85197_TV.tsv`
in the corpus -- the dataset's own expert-reviewed segmentation. Nothing about
the overlay is generated, smoothed or inferred, so every boundary drawn on
screen traces back to a row of that file.

## Scope

This is a **dataset sample**, not a patient and not a case. The corpus is
de-identified paediatric screening data, and PV-MEPCG / PulseVision is an
academic screening and decision-support prototype: it does not diagnose.
