"""Chain every mandatory experiment in dependency order (T63.6).

The order is **computed from the config**, not typed here: every experiment's
``depends_on`` list is topologically sorted, with the declared phase number as
the tie-break so two independent runs still come out in the order the project
builds them. Typing the order by hand would let it drift from the declarations
the moment an experiment gains a dependency.

Experiments this runner cannot execute are listed, never silently dropped:

* ``analysis_only`` runs (EXP-C3, EXP-E1, EXP-E2) re-slice predictions other
  experiments produced; they have their own phase scripts.
* ``holdout_external`` (EXP-D1) trains on one dataset and tests on another,
  which is not a fold map and needs Phase 71's own runner.
* Everything whose phase has not been built yet is reported with that phase
  number, so a short run is visibly short rather than looking complete.

Usage
-----
    python scripts/12_run_all_experiments.py --plan
    python scripts/12_run_all_experiments.py
    python scripts/12_run_all_experiments.py --through EXP-A2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/12_run_all_experiments.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("run_all_experiments")

#: What this runner can execute today. An experiment is added here by the phase
#: that builds it, alongside the arguments that phase settled on. Anything
#: absent is reported as pending with its phase number rather than attempted.
IMPLEMENTED: dict[str, list[str]] = {
    "EXP-A1": ["--planner", "default"],
    "EXP-A2": ["--planner", "nested"],
}

RUNNER = "scripts/11_run_experiment.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="12_run_all_experiments",
        description="Run every runnable experiment in dependency order.",
    )
    parser.add_argument(
        "--plan", action="store_true",
        help="print the resolved order and what each entry would do; run nothing",
    )
    parser.add_argument(
        "--through", default=None, metavar="EXP-ID",
        help="stop after this experiment (inclusive)",
    )
    parser.add_argument(
        "--only", nargs="+", default=None, metavar="EXP-ID",
        help="run just these, still in dependency order",
    )
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="arguments appended to every underlying run")
    return parser.parse_args(argv)


def dependency_order() -> list[str]:
    """Topological order over ``depends_on``, phase number as the tie-break."""
    from src.utils.config import load_config

    table = load_config("experiments").require("experiments")
    phase = {exp: int(spec.get("phase", 0)) for exp, spec in table.items()}
    pending = {
        exp: {dep for dep in spec.get("depends_on", []) if dep in table}
        for exp, spec in table.items()
    }

    ordered: list[str] = []
    while pending:
        ready = sorted(
            (exp for exp, deps in pending.items() if not deps),
            key=lambda exp: (phase[exp], exp),
        )
        if not ready:
            raise RuntimeError(
                "configs/experiments.yaml has a dependency cycle among "
                + ", ".join(sorted(pending))
            )
        for exp in ready:
            ordered.append(exp)
            del pending[exp]
        for deps in pending.values():
            deps.difference_update(ready)
    return ordered


def classify(exp_id: str) -> tuple[str, str]:
    """``(status, reason)`` for one experiment: runnable, analysis, external, pending."""
    from src.utils.config import load_config

    spec = dict(load_config("experiments").require("experiments")[exp_id])
    phase = spec.get("phase", "?")
    if spec.get("analysis_only", False):
        return "analysis", "analysis-only; Phase " + str(phase) + " re-slices existing predictions"
    if str(spec.get("cv", "")) == "holdout_external":
        return "external", "train-on-A/test-on-B; Phase " + str(phase) + " has its own runner"
    if exp_id in IMPLEMENTED:
        return "runnable", ""
    return "pending", "built in Phase " + str(phase)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    order = dependency_order()

    if args.only:
        unknown = [exp for exp in args.only if exp not in order]
        if unknown:
            log.error("unknown experiment id(s): %s", ", ".join(unknown))
            return 1
        order = [exp for exp in order if exp in set(args.only)]
    if args.through:
        if args.through not in order:
            log.error("--through %s is not in the resolved order", args.through)
            return 1
        order = order[: order.index(args.through) + 1]

    rows = [(exp, *classify(exp)) for exp in order]
    width = max(len(exp) for exp, _, _ in rows)
    print("resolved dependency order (" + str(len(rows)) + " experiment(s)):")
    for exp, status, reason in rows:
        print("  " + exp.ljust(width) + "  " + status.ljust(9) + "  " + reason)
    print()

    if args.plan:
        return 0

    failures: list[str] = []
    for exp, status, reason in rows:
        if status != "runnable":
            log.info("%s SKIPPED -- %s", exp, reason)
            continue
        command = [
            sys.executable, RUNNER, "--exp", exp, *IMPLEMENTED[exp], *args.extra
        ]
        log.info("running %s: %s", exp, " ".join(command))
        started = time.perf_counter()
        completed = subprocess.run(command, check=False)
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            log.error("%s FAILED (exit %d) after %.1f s", exp, completed.returncode, elapsed)
            failures.append(exp)
            break  # a downstream run would consume a result that does not exist
        log.info("%s completed in %.1f s", exp, elapsed)

    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
