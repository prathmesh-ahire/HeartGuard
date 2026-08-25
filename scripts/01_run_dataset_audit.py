"""Run the whole Part II dataset audit from one command (T22.3).

Executes Phases 08 through 21 in order and writes DA-01 .. DA-09 into
``outputs/01_dataset_audit/``:

    08-11  PhysioNet loader, appendix enrichment, subject derivation
    12-13  PASCAL set_a / set_b loader and subject derivation
    14-15  CirCor patient labels, recordings and segmentation
    16     audio integrity scan            -> DA-05
    17     duplicate detection             -> DA-06
    18     duration / sampling / class      -> DA-03, DA-04, DA-02
    19     master metadata assembly        -> DA-08
    20     split map generation            -> DA-07
    21     inventory and narrative report  -> DA-01, DA-09

T22.3 names Phases 08-20. Phase 21 is included because DA-01 and DA-09 are part
of the same audit and the T21.7 gate requires them on disk; running them
separately would leave "the audit" meaning two different things.

Usage
-----
    python scripts/01_run_dataset_audit.py
    python scripts/01_run_dataset_audit.py --limit 20     # smoke run
    python scripts/01_run_dataset_audit.py --force        # ignore every cache

The run is idempotent: a second invocation over the same dataset reproduces
byte-identical CSVs, because every step is seeded at 42 and nothing sampled at
random. ``--limit`` is a smoke path only -- it loads and scans a small slice to
prove the pipeline runs, and deliberately writes no DA artifact, because a
partial DA file on disk reads exactly like a complete one.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/01_run_dataset_audit.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger
from src.utils.timing import format_duration, timer

log = get_logger("audit")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="01_run_dataset_audit",
        description="Run the Part II dataset audit end to end (Phases 08-21).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="smoke run: load at most N records per dataset and write no DA files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="ignore the metadata and audio-scan caches and rebuild everything",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="do not read or write the parsed-metadata cache at all",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="write the DA artifacts somewhere other than outputs/01_dataset_audit",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="skip the CirCor SHA-256 pass over 585 MB (about 40 s)",
    )
    return parser.parse_args(argv)


def run_audit(args: argparse.Namespace) -> dict[str, object]:
    """The audit itself. Returns a summary dict; raises on any failure."""
    from src.data_loader.catalog import build_catalog
    from src.data_loader.circor import load_circor
    from src.data_loader.duplicates import (
        heartbeat_extra_files,
        run_duplicate_detection,
    )
    from src.data_loader.integrity import (
        check_label_coverage,
        scan_corpus,
        write_missing_corrupt_report,
    )
    from src.data_loader.inventory import run_inventory
    from src.data_loader.master import run_master_assembly
    from src.data_loader.pascal import load_pascal
    from src.data_loader.physionet import load_physionet
    from src.data_loader.splits import run_split_generation
    from src.data_loader.summaries import run_summaries
    from src.utils.run_manifest import start_run

    smoke = args.limit is not None
    use_cache = not args.no_cache
    out_dir = args.out_dir

    run = start_run("dataset_audit" + ("_smoke" if smoke else ""))
    run.set("audit_mode", "smoke" if smoke else "full")
    run.set("audit_limit", args.limit)
    started = time.perf_counter()

    # ---- Phases 08-15: the four loaders --------------------------------
    load_kwargs = {"limit": args.limit, "use_cache": use_cache}
    if args.force:
        load_kwargs["use_cache"] = False

    with timer("audit:phase_08_11_physionet"):
        physionet = load_physionet(
            write_outputs=not smoke, out_dir=out_dir, **load_kwargs
        )
    with timer("audit:phase_12_13_pascal"):
        pascal = load_pascal(write_outputs=not smoke, out_dir=out_dir, **load_kwargs)
    with timer("audit:phase_14_15_circor"):
        circor = load_circor(
            with_segmentation=not smoke,
            write_outputs=not smoke,
            out_dir=out_dir,
            verify=not smoke and not args.skip_verify,
            **load_kwargs,
        )

    with timer("audit:catalog"):
        catalog = build_catalog(physionet=physionet, pascal=pascal, circor=circor)

    if smoke:
        # The smoke path stops before anything is written. It has proved the
        # four loaders parse, agree on a schema and concatenate; that is what a
        # smoke run is for. Writing a 60-row DA-08 would not be.
        with timer("audit:smoke_scan"):
            scan, _ = scan_corpus(catalog, force=args.force, with_hashes=True)
        elapsed = time.perf_counter() - started
        run.set("audit_records", len(catalog))
        run.set("audit_wall_seconds", round(elapsed, 3))
        run.finish()
        log.info(
            "SMOKE RUN complete in %s: %d records, %d scanned, no DA files written",
            format_duration(elapsed),
            len(catalog),
            len(scan),
        )
        return {"mode": "smoke", "n_records": len(catalog), "seconds": elapsed}

    # ---- Phase 16: integrity scan (DA-05) -------------------------------
    # One decode pass covers both phases. Phase 17 additionally has to hash the
    # 832 Heartbeat_Sound files to prove they duplicate set_a + set_b; they are
    # not records, so they ride along as extras rather than entering the
    # catalog.
    with timer("audit:phase_16_integrity"):
        heartbeat = heartbeat_extra_files()
        scan_all, envelopes_all = scan_corpus(
            catalog, force=args.force, extra_files=heartbeat
        )
        scan = scan_all[scan_all["dataset_source"] != "heartbeat_sound"]
        coverage = check_label_coverage(catalog, scan)
        write_missing_corrupt_report(scan, coverage, out_dir)

    with timer("audit:phase_17_duplicates"):
        duplicates = run_duplicate_detection(
            scan_all, envelopes_all, write_outputs=True, out_dir=out_dir
        )

    # ---- Phase 18: summaries (DA-02, DA-03, DA-04) ----------------------
    with timer("audit:phase_18_summaries"):
        summaries = run_summaries(
            catalog, scan, write_outputs=True, out_dir=out_dir
        )

    # ---- Phase 19: master metadata (DA-08) ------------------------------
    with timer("audit:phase_19_master"):
        master_result = run_master_assembly(
            catalog=catalog,
            scan=scan,
            duplicates=duplicates,
            physionet=physionet,
            pascal=pascal,
            circor=circor,
            out_dir=out_dir,
        )
        master = master_result["master"]

    # ---- Phase 20: split maps (DA-07) -----------------------------------
    with timer("audit:phase_20_splits"):
        splits = run_split_generation(master, out_dir=out_dir)

    # ---- Phase 21: inventory and report (DA-01, DA-09) ------------------
    with timer("audit:phase_21_inventory"):
        inventory = run_inventory(master, out_dir=out_dir)

    elapsed = time.perf_counter() - started

    # T22.5 -- the wall time is itself a deliverable (it feeds T25).
    run.set("audit_wall_seconds", round(elapsed, 3))
    run.set("audit_records", len(master))
    run.set("audit_supervised", int(master["use_in_supervised"].sum()))
    run.set("audit_files_scanned", len(scan_all))
    run.set("audit_fold_assignments", len(splits["split_map"]))
    run.finish()

    log.info(
        "AUDIT COMPLETE in %s -- %d records, %d supervised, %d files scanned, "
        "%d fold assignments",
        format_duration(elapsed),
        len(master),
        int(master["use_in_supervised"].sum()),
        len(scan_all),
        len(splits["split_map"]),
    )
    _ = (envelopes_all, summaries, inventory)  # kept for readability of the flow
    return {
        "mode": "full",
        "n_records": len(master),
        "n_supervised": int(master["use_in_supervised"].sum()),
        "n_scanned": len(scan_all),
        "n_assignments": len(splits["split_map"]),
        "seconds": elapsed,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_audit(args)
    except Exception as error:
        log.error("audit failed: %s", error, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
