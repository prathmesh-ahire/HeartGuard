"""Final delivery: checklist, counts, handover, ZIP, and the ZIP's own check (Phase 126).

Usage
-----
    python scripts/51_package_delivery.py --check        # T126.1/T126.2 only
    python scripts/51_package_delivery.py --handover     # write HANDOVER.md
    python scripts/51_package_delivery.py                # everything, then build and verify
    python scripts/51_package_delivery.py --verify-only  # re-open an existing archive

The archive lands in `dist/`, which is gitignored: a 200 MB build product does
not belong in version control, and rebuilding it is one command.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/51_package_delivery.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="51_package_delivery",
        description="Verify the completion checklist and build the delivery archive.",
    )
    parser.add_argument("--check", action="store_true", help="checklist and counts only")
    parser.add_argument("--handover", action="store_true", help="write HANDOVER.md and stop")
    parser.add_argument("--verify-only", action="store_true", help="check an existing archive")
    parser.add_argument("--out", default=None, help="where to write the archive")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from src.reporting.delivery import (
        DeliveryError,
        asset_inventory,
        build_zip,
        checklist,
        verify_zip,
        write_handover,
    )
    from src.reporting.evidence_pack import assemble
    from src.utils.evidence import register_evidence
    from src.utils.run_manifest import start_run

    destination = Path(args.out) if args.out else None

    if args.verify_only:
        try:
            print(verify_zip(destination))
        except DeliveryError as error:
            print("FAILED: " + str(error))
            return 1
        return 0

    run = start_run("package_delivery")
    try:
        # T126.2 -- the counts.
        counts = asset_inventory()
        print("\nAsset counts (T126.2)")
        problems = 0
        for count in counts:
            status = "ok" if count.ok else "GAP"
            print(f"  {status:4s} {count.kind:12s} {count.found:3d} / {count.target}")
            if not count.ok:
                problems += 1
                print(f"       missing: {', '.join(count.missing)}")

        # T126.1 -- the section-19 checklist.
        spec = checklist()
        print("\nCompletion checklist (T126.1)")
        print(
            f"  {spec['present']} present, {spec['not_produced']} not produced (with reason), "
            f"{spec['failed']} failed, of {spec['total']}"
        )
        for row in spec["items"]:
            if row["status"] != "present":
                reason = str(row.get("detail", ""))[:100]
                print(f"    {row['status']:14s} {row['item_id']:22s} {reason}")
        if spec["unexplained"]:
            print(f"  UNEXPLAINED (no reason recorded): {spec['unexplained']}")
            problems += 1
        if spec["failed"]:
            problems += 1

        handover = write_handover()
        print(f"\nwrote {handover.name}")
        register_evidence(
            "DOC-HANDOVER",
            handover,
            metric_or_asset="handover summary: produced, not produced, next steps (T126.6)",
            command="python scripts/51_package_delivery.py",
        )

        if args.handover:
            run.finish(status="ok")
            return 0

        if args.check:
            run.finish(status="ok" if not problems else "failed")
            return 1 if problems else 0

        # Re-assemble the evidence index BEFORE packaging, so the archive
        # carries an index and a workbook that agree with each other and with
        # the report. This stage runs after `evidence_index_final`, so without
        # this the handover's row would be in the CSV and absent from the
        # workbook -- the exact disagreement T102.1 tests for.
        assemble()

        # T126.3 / T126.4 -- build it, then open it.
        archive = build_zip(destination)
        try:
            result = verify_zip(archive)
        except DeliveryError as error:
            print("FAILED: " + str(error))
            run.finish(status="failed")
            return 1

        print(
            f"\n{archive.name}: {result['megabytes']} MB, {result['entries']} entries, "
            f"sha256 {result['sha256'][:16]}, Node required: {result['node_required']}"
        )
        # Deliberately NOT registered as evidence. The index is inside the
        # archive, so a row describing the archive could never be in the copy
        # the archive carries -- and `dist/` is gitignored, so the committed
        # index would name a file no other checkout has. The archive's own
        # record is DELIVERY_MANIFEST.json inside it and the .sha256 beside it.
    except Exception:
        run.finish(status="failed")
        raise

    run.finish(status="ok" if not problems else "failed")
    if problems:
        print("\nFAILED: the delivery has unexplained gaps")
        return 1
    print("\ndelivery packaged and verified")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
