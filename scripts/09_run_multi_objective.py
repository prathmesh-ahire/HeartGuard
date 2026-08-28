"""SO-06: the performance-versus-complexity Pareto front (Phase 61).

Emits, into ``outputs/05_search_optimization/SO-06/``:

    pareto_configurations.csv  every (ranker, k) with its three objectives and
                               whether it is non-dominated (T61.4, T61.6)
    pareto_front.csv           the non-dominated set alone
    weighting_sweep.csv        which configuration each (alpha, beta, gamma)
                               selects, raw and under the T57.5 guard (T61.4)
    operating_point.json       where the shipped subset sits on the front (T61.5)
    j_definition.json          the objective exactly as implemented, its weights
                               and the measured cost model behind its time term
                               (T61.1, T61.2, T61.3)

Reads the SO-04 sweep and computes; it fits nothing, so it costs seconds.

The operating point is the subset SO-04 already ships. SO-06 places that point on
the front and reports how much of weight space agrees with it; it does not make a
second, competing choice of feature subset.

Usage
-----
    python scripts/09_run_multi_objective.py
    python scripts/09_run_multi_objective.py --resolution 0.02
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/09_run_multi_objective.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging_setup import get_logger

log = get_logger("run_multi_objective")

EXP = "SO-06"
CONFIGURATIONS_FILENAME = "pareto_configurations.csv"
FRONT_FILENAME = "pareto_front.csv"
SWEEP_FILENAME = "weighting_sweep.csv"
OPERATING_FILENAME = "operating_point.json"
DEFINITION_FILENAME = "j_definition.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="09_run_multi_objective",
        description="Trace the SO-06 Pareto front from the SO-04 sweep.",
    )
    parser.add_argument(
        "--resolution", type=float, default=0.05,
        help="lattice spacing for the (alpha, beta, gamma) sweep",
    )
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args(argv)


def _section_root(out_dir: str | None) -> Path:
    from src.utils.config import load_config

    return (
        Path(out_dir)
        if out_dir
        else Path(load_config("paths").require("outputs.search_optimization"))
    )


def main(argv: list[str] | None = None) -> int:
    import json

    import pandas as pd

    from src.optimization import multi_objective as mo
    from src.optimization import pareto as pt
    from src.utils.evidence import register_evidence
    from src.utils.io import ensure_dir, save_csv, save_json
    from src.utils.run_manifest import start_run
    from src.utils.seed import GLOBAL_SEED, set_global_seed

    args = parse_args(argv)
    set_global_seed()

    root = _section_root(args.out_dir)
    sweep_path = root / "SO-04" / "feature_count_sweep.csv"
    settings_path = root / "SO-04" / "so04_settings.json"
    if not sweep_path.is_file():
        raise SystemExit(
            "SO-06 needs " + str(sweep_path) + "; run scripts/06_run_feature_selection.py first"
        )
    if not settings_path.is_file():
        raise SystemExit("SO-06 needs " + str(settings_path) + " to place the operating point")

    command = "python scripts/09_run_multi_objective.py"
    run = start_run("multi_objective")
    run.set("exp", EXP)

    started = time.perf_counter()
    sweep = pd.read_csv(sweep_path)
    so04 = json.loads(settings_path.read_text(encoding="utf-8"))
    weights = mo.load_weights()
    cost_model = mo.load_cost_model()

    # --- T61.4: the front ----------------------------------------------------
    configurations = pt.pareto_front(pt.configuration_frame(sweep))
    front = configurations[configurations.on_front].sort_values(["n_selected", "ranker"])
    log.info(
        "%d configurations, %d on the Pareto front over (macro-F1, feature count, "
        "inference time)",
        len(configurations), len(front),
    )

    # --- T61.4: the weighting sweep, raw and guarded -------------------------
    grid = pt.weighting_grid(args.resolution)
    guard = so04.get("settings", {}).get("n_standard_errors")
    raw = pt.sweep_weightings(configurations, grid)
    frames = [raw]
    if guard is not None:
        frames.append(
            pt.sweep_weightings(configurations, grid, n_standard_errors=float(guard))
        )
    weighting_sweep = pd.concat(frames, ignore_index=True)

    for guarded, block in weighting_sweep.groupby("guarded"):
        top = (
            block.groupby(["selected_ranker", "selected_k"])
            .size()
            .sort_values(ascending=False)
            .head(3)
        )
        log.info(
            "  %s J: %s",
            "guarded" if guarded else "raw   ",
            "; ".join(
                str(name[0]) + " k=" + str(name[1]) + " -> "
                + str(round(100.0 * count / len(block), 1)) + "% of weight space"
                for name, count in top.items()
            ),
        )

    # --- T61.5: where the shipped subset sits --------------------------------
    chosen = so04["chosen"]
    guarded_sweep = (
        weighting_sweep[weighting_sweep.guarded] if guard is not None else weighting_sweep
    )
    operating = pt.operating_point(
        configurations, str(chosen["ranker"]), int(chosen["k"]), sweep=guarded_sweep
    )
    operating["selection_rule"] = str(chosen.get("selection_rule", ""))
    operating["configured_weights"] = weights.as_dict()
    # J is computed from the three terms directly rather than through
    # `mo.score_j`, which takes feature NAMES so it can derive the family set.
    # Here the family set is already known -- it is the union across folds that
    # `configuration_frame` charged -- so re-deriving it from names would be
    # asking a second question and hoping for the same answer.
    operating["j_at_configured_weights"] = float(
        weights.alpha * (1.0 - operating["macro_f1"])
        + weights.beta * operating["n_selected"] / weights.n_features_total
        + weights.gamma * operating["normalized_inference_time"]
    )

    # --- the emitted files ---------------------------------------------------
    directory = Path(ensure_dir(root / EXP))
    configurations_path = save_csv(configurations, directory / CONFIGURATIONS_FILENAME)
    front_path = save_csv(front, directory / FRONT_FILENAME)
    sweep_out_path = save_csv(weighting_sweep, directory / SWEEP_FILENAME)
    operating_path = save_json(operating, directory / OPERATING_FILENAME)

    time_span = float(
        configurations.normalized_inference_time.max()
        - configurations.normalized_inference_time.min()
    )
    definition_path = save_json(
        {
            "experiment": EXP,
            "formula": (
                "J = alpha*(1 - MacroF1) + beta*(SelectedFeatures/138) "
                "+ gamma*NormalizedInferenceTime"
            ),
            "minimised": True,
            "implemented_in": "src/optimization/multi_objective.py",
            "weights": weights.as_dict(),
            "cost_model": cost_model.as_dict(),
            "inference_time_basis": (
                "per-family extraction seconds; a subset pays a family in full if it "
                "keeps any feature from it. Normalised against all families, so the "
                "term is bounded in [0, 1] by construction (T61.3)."
            ),
            "inference_time_span_across_configurations": time_span,
            "inference_time_is_effectively_inert": bool(time_span < 0.05),
            "n_configurations": len(configurations),
            "n_on_front": len(front),
            "weighting_lattice_resolution": float(args.resolution),
            "n_weightings_sampled": len(grid),
            "seed": int(GLOBAL_SEED),
            "seconds": time.perf_counter() - started,
        },
        directory / DEFINITION_FILENAME,
    )

    for path in (configurations_path, front_path, sweep_out_path,
                 operating_path, definition_path):
        run.record_artifact(path)
    run.record_timing("so06_total", time.perf_counter() - started)

    register_evidence(
        EXP, front_path,
        metric_or_asset="SO-06 Pareto front over macro-F1, feature count and inference time",
        dataset="D1",
        source_data="outputs/05_search_optimization/SO-04/feature_count_sweep.csv",
        command=command,
    )
    register_evidence(
        EXP + "-SWEEP", sweep_out_path,
        metric_or_asset=(
            "SO-06 weighting sweep: the configuration each (alpha, beta, gamma) selects"
        ),
        dataset="D1",
        command=command,
    )
    register_evidence(
        EXP + "-POINT", operating_path,
        metric_or_asset="SO-06 final operating point and its share of weight space",
        dataset="D1",
        command=command,
    )
    register_evidence(
        EXP + "-J", definition_path,
        metric_or_asset=(
            "SO-06 multi-objective score J as implemented, with its weights and cost model"
        ),
        dataset="D1",
        command=command,
    )

    print(
        front[
            ["ranker", "k", "macro_f1", "macro_f1_se", "n_selected",
             "normalized_inference_time"]
        ].to_string(index=False)
    )
    print()
    print(
        "operating point: " + operating["ranker"] + " k=" + str(operating["k"])
        + ", on front: " + str(operating["on_pareto_front"])
        + ", chosen by "
        + str(round(100.0 * operating.get("share_of_weight_space", float("nan")), 1))
        + "% of weight space under the guard"
    )
    if time_span < 0.05:
        log.warning(
            "the inference-time term spans only %.4f across all %d configurations, so "
            "gamma cannot change any selection on this sweep; recorded in %s",
            time_span, len(configurations), definition_path.name,
        )
    log.info("%s finished in %.1fs", EXP, time.perf_counter() - started)
    run.finish(status="ok")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
