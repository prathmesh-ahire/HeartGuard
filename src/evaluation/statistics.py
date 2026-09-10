"""Statistical validation of the model comparisons (Phase 82).

Every test here answers "is this difference bigger than the noise?", and every
one of them is easy to run in a way that produces a number meaning nothing. The
three traps this module is built around:

## 1. Every p-value states its n, always

The blueprint's statistical plan is underpowered as written: Wilcoxon over 5 CV
folds is n=5, whose two-sided p-value cannot go below 0.0625 **no matter how
large the effect**. That is why the primary binary track is repeated 5x5 and why
:data:`N_DESCRIPTION` travels on every emitted row. A p-value of 0.0625 from five
folds and a p-value of 0.0625 from twenty-five mean different things, and a table
that prints only the number cannot tell them apart.

An underpowered test that fails to reject is **not** evidence of no difference.
:func:`minimum_achievable_p` computes, for each paired test, the smallest
p-value its sample size could possibly produce, so "we could not have detected
this" is a fact on the row rather than a caveat someone has to remember.

## 2. A repeated map counts every record five times

The 5x5 map holds every record out once per repeat, so pooling all 25 folds'
predictions gives a set five times the size of the corpus with each record in it
five times. A McNemar test on that pool would have five times the discordant
count it is entitled to and would reject almost anything.

So record-level tests (McNemar, the bootstrap) run on **repeat 0 alone**, which
is a complete partition: every record exactly once. Fold-level tests (Wilcoxon,
paired t, Friedman) use all 25 folds, because there the unit of observation is
the fold, not the record.

## 3. An effect size beside every p-value

Cohen's d for the paired t-test, the matched-pairs rank-biserial correlation for
Wilcoxon, Kendall's W for Friedman, and the discordant odds ratio for McNemar.
A significant p-value on 25 folds can sit on a 0.001 difference in balanced
accuracy, and the effect size is what stops that being written up as a result.

## The normality check is recorded, not just performed

T82.4 says to choose between Wilcoxon and the paired t-test "per a documented
normality check". Shapiro-Wilk on the paired differences, alpha 0.05, and the
outcome is written into the row along with **both** tests' results -- so the
choice is auditable and a reader who disagrees with it can see the other number.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_setup import get_logger

__all__ = [
    "ALPHA",
    "HEADLINE_METRICS",
    "PARTITION_REPEAT",
    "BOOTSTRAP_RESAMPLES",
    "N_DESCRIPTION",
    "RUNS",
    "StatsError",
    "StatRun",
    "load_fold_metrics",
    "foldwise_summary",
    "partition_predictions",
    "bootstrap_auc",
    "mcnemar_matrix",
    "paired_fold_tests",
    "friedman_test",
    "nemenyi_critical_difference",
    "minimum_achievable_p",
    "correct_p_values",
]

log = get_logger("evaluation.statistics")

ALPHA = 0.05

#: The metrics every paired comparison is run on. Sensitivity and balanced
#: accuracy lead because research rule 6 makes them the selection criteria.
HEADLINE_METRICS: tuple[str, ...] = (
    "sensitivity",
    "balanced_accuracy",
    "specificity",
    "f1",
    "roc_auc",
    "accuracy",
    "macro_f1",
)

#: Record-level tests use this repeat alone -- a complete partition of the corpus.
PARTITION_REPEAT = 0

BOOTSTRAP_RESAMPLES = 2000

N_DESCRIPTION = {
    "fold": "n is the number of cross-validation folds, one paired observation each",
    "record": "n is the number of records, each appearing exactly once",
}


class StatsError(RuntimeError):
    """A statistical test cannot be run as asked, or would not mean what it says."""


@dataclass(frozen=True)
class StatRun:
    """One stored run the tests are applied to."""

    run: str
    directory: str
    task: str
    note: str = ""


RUNS: tuple[StatRun, ...] = (
    StatRun(
        "EXP-A1",
        "outputs/06_binary_results/EXP-A1",
        "binary",
        "config defaults over the repeated 5x5 map",
    ),
    StatRun(
        "EXP-A2",
        "outputs/06_binary_results/EXP-A2",
        "binary",
        "nested search over the repeated 5x5 map -- the headline binary run",
    ),
    StatRun(
        "EXP-B1",
        "outputs/07_multiclass_results/EXP-B1",
        "pascal_a",
        "PASCAL A, 10 folds (5x2). n=124 records; every test here is underpowered",
    ),
    StatRun(
        "EXP-B2",
        "outputs/07_multiclass_results/EXP-B2",
        "pascal_b",
        "PASCAL B, 5 folds and NOT repeated -- n=5, the underpowered case",
    ),
    StatRun(
        "EXP-C1-three_class",
        "outputs/08_circor_external_validation/EXP-C1-three_class",
        "circor_murmur",
        "CirCor murmur, 5 patient-grouped folds",
    ),
    StatRun(
        "EXP-C1-two_class",
        "outputs/08_circor_external_validation/EXP-C1-two_class",
        "circor_murmur",
        "CirCor murmur over the 874 known-murmur patients, 5 folds",
    ),
    StatRun(
        "EXP-C2",
        "outputs/08_circor_external_validation/EXP-C2",
        "circor_outcome",
        "CirCor clinical outcome, 5 patient-grouped folds",
    ),
)


def load_fold_metrics(spec: StatRun, *, root: Path | None = None) -> Any:
    """One run's ``per_fold_metrics.csv``, or ``None`` if it has not been run."""
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    path = (root or PROJECT_ROOT) / spec.directory / "per_fold_metrics.csv"
    if not path.is_file():
        log.warning("%s: no per_fold_metrics.csv", spec.run)
        return None
    frame = pd.read_csv(path)
    frame.insert(0, "run", spec.run)
    return frame


# ---------------------------------------------------------------------------
# T82.1 -- fold-wise mean and SD
# ---------------------------------------------------------------------------


def foldwise_summary(per_fold: Any, metrics: tuple[str, ...] = HEADLINE_METRICS) -> Any:
    """mean, SD and the 95% t-interval of the MEAN, per (run, model, metric).

    The interval is over folds and is an interval on the mean of a
    cross-validated estimate. It is not a confidence interval on a single
    record-level metric -- that is what the bootstrap in :func:`bootstrap_auc`
    is for, and the two are different quantities with different widths.
    """
    import pandas as pd
    from scipy import stats

    rows: list[dict[str, Any]] = []
    for (run, model_id), block in per_fold.groupby(["run", "model_id"], sort=True):
        for metric in metrics:
            if metric not in block.columns:
                continue
            values = np.asarray(block[metric], dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            n = int(values.size)
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if n > 1 else float("nan")
            half = (
                float(stats.t.ppf(1 - ALPHA / 2, n - 1) * sd / np.sqrt(n))
                if n > 1 and np.isfinite(sd)
                else float("nan")
            )
            rows.append(
                {
                    "run": str(run),
                    "model_id": str(model_id),
                    "metric": metric,
                    "n_folds": n,
                    "mean": mean,
                    "sd": sd,
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "ci_low": mean - half,
                    "ci_high": mean + half,
                    "ci_kind": "95% t-interval of the fold mean",
                    "formatted": format(mean, ".4f") + " +/- " + format(sd, ".4f"),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# record-level tests: the partition, the bootstrap, McNemar
# ---------------------------------------------------------------------------


def partition_predictions(spec: StatRun, *, root: Path | None = None) -> Any:
    """The run's out-of-fold predictions restricted to one complete partition.

    On a repeated map that is repeat 0's folds -- every record exactly once. On
    a plain k-fold map it is the whole thing. Either way the result holds each
    record once, which is the precondition for any record-level test.
    """
    import pandas as pd

    from src.utils.evidence import PROJECT_ROOT

    path = (root or PROJECT_ROOT) / spec.directory / "predictions.parquet"
    if not path.is_file():
        log.warning("%s: no predictions.parquet", spec.run)
        return None
    frame = pd.read_parquet(path)
    if "repeat" in frame.columns and frame["repeat"].nunique() > 1:
        frame = frame[frame["repeat"] == PARTITION_REPEAT]

    duplicated = frame.groupby("model_id")["record_uid"].apply(
        lambda s: int(s.duplicated().sum())
    )
    if int(duplicated.max()) > 0:
        raise StatsError(
            spec.run
            + ": the chosen partition still holds duplicate records (max "
            + str(int(duplicated.max()))
            + " per model); a record-level test on it would count them twice"
        )
    return frame


def bootstrap_auc(
    predictions: Any,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 42,
    alpha: float = ALPHA,
) -> Any:
    """T82.2 -- percentile bootstrap CI for ROC-AUC, resampling RECORDS.

    Runs on the partition, so each record enters the resample once and the
    interval has the width the corpus supports rather than the width five copies
    of it would suggest. Seeded, so the published interval does not move on
    re-run.

    **Binary runs only, and the multiclass ones are recorded as skipped rather
    than dropped.** T82.2 asks for the CI "on the key binary results", and a
    multiclass AUC needs a one-vs-rest or one-vs-one scheme that is a different
    quantity with a different meaning. Passing a 3-class target with a single
    positive-class column would either raise or, worse, silently score one class
    against the rest and label it "ROC-AUC". A row with ``status='skipped'`` and
    the reason keeps the absence visible.
    """
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    rows: list[dict[str, Any]] = []
    for (run, model_id), block in predictions.groupby(["run", "model_id"], sort=True):
        truth = block["y_true"].to_numpy(dtype=int)
        classes = np.unique(truth)
        if "proba_1" not in block.columns or classes.size != 2:
            rows.append(
                {
                    "run": str(run),
                    "model_id": str(model_id),
                    "metric": "roc_auc",
                    "n": int(truth.size),
                    "n_description": N_DESCRIPTION["record"],
                    "status": "skipped",
                    "reason": (
                        "the target has "
                        + str(classes.size)
                        + " classes; a binary ROC-AUC is undefined for it, and a "
                        "one-vs-rest AUC is a different quantity that must not "
                        "be labelled the same way"
                        if classes.size != 2
                        else "no positive-class probability column"
                    ),
                    "point": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "ci_width": float("nan"),
                    "n_resamples": 0,
                    "n_valid_resamples": 0,
                    "n_skipped_single_class": 0,
                    "seed": seed,
                    "ci_kind": "n/a",
                }
            )
            continue
        score = block["proba_1"].to_numpy(dtype=float)
        rng = np.random.default_rng(seed)
        point = float(roc_auc_score(truth, score))
        samples: list[float] = []
        skipped = 0
        for _ in range(n_resamples):
            index = rng.integers(0, truth.size, size=truth.size)
            drawn = truth[index]
            if len(np.unique(drawn)) < 2:
                # A resample with one class has no AUC. Counted and excluded --
                # never replaced by a substitute value.
                skipped += 1
                continue
            samples.append(float(roc_auc_score(drawn, score[index])))
        values = np.asarray(samples, dtype=float)
        low, high = (
            np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            if values.size
            else (float("nan"), float("nan"))
        )
        rows.append(
            {
                "run": str(run),
                "model_id": str(model_id),
                "metric": "roc_auc",
                "n": int(truth.size),
                "n_description": N_DESCRIPTION["record"],
                "status": "computed",
                "reason": "",
                "point": point,
                "ci_low": float(low),
                "ci_high": float(high),
                "ci_width": float(high - low),
                "n_resamples": int(n_resamples),
                "n_valid_resamples": int(values.size),
                "n_skipped_single_class": skipped,
                "seed": seed,
                "ci_kind": "percentile bootstrap over records, "
                + str(int((1 - alpha) * 100))
                + "%",
            }
        )
    return pd.DataFrame(rows)


def mcnemar_matrix(predictions: Any) -> Any:
    """T82.3 -- McNemar over paired predictions, every model pair, per run.

    The exact binomial form when the discordant total is small (<= 25) and the
    chi-square form with continuity correction otherwise; the row says which was
    used. ``n`` is the discordant count, which is the test's real sample size --
    a pair agreeing on 3,000 records and disagreeing on 4 has n=4, and printing
    3,240 beside that p-value would be misleading.
    """
    import pandas as pd
    from statsmodels.stats.contingency_tables import mcnemar

    rows: list[dict[str, Any]] = []
    for run, block in predictions.groupby("run", sort=True):
        wide = block.pivot_table(
            index="record_uid", columns="model_id", values="y_pred", aggfunc="first"
        )
        truth = (
            block.drop_duplicates("record_uid")
            .set_index("record_uid")["y_true"]
            .reindex(wide.index)
            .to_numpy(dtype=int)
        )
        correct = {
            model: (wide[model].to_numpy(dtype=int) == truth) for model in wide.columns
        }
        for first, second in combinations(sorted(wide.columns), 2):
            a, b = correct[first], correct[second]
            only_first = int(np.sum(a & ~b))
            only_second = int(np.sum(~a & b))
            discordant = only_first + only_second
            table = [[int(np.sum(a & b)), only_first], [only_second, int(np.sum(~a & ~b))]]
            exact = discordant <= 25
            if discordant == 0:
                # Two models that agreed on EVERY record. There is nothing to
                # test: McNemar's statistic is a function of the discordant
                # cells alone, and a p-value of 1.0 here would read as "tested,
                # found no difference" when the truth is "no test exists". NaN,
                # with the reason on the row. M6 and M7 hit this on the CirCor
                # runs, where they are the same fitted ensemble.
                statistic, p_value = float("nan"), float("nan")
                note = (
                    "not tested: the two models agreed on all "
                    + str(len(wide))
                    + " records, so McNemar has no discordant pairs to work with"
                )
            else:
                result = mcnemar(table, exact=exact, correction=not exact)
                statistic, p_value = float(result.statistic), float(result.pvalue)
                note = ""
            rows.append(
                {
                    "run": str(run),
                    "test": "McNemar (exact binomial)" if exact else "McNemar (chi-square, cc)",
                    "note": note,
                    "model_a": first,
                    "model_b": second,
                    "n": discordant,
                    "n_description": (
                        "n is the DISCORDANT count -- records the two models "
                        "disagreed about. Agreements carry no information for "
                        "this test."
                    ),
                    "n_records_compared": len(wide),
                    "n_correct_a_only": only_first,
                    "n_correct_b_only": only_second,
                    "n_both_correct": table[0][0],
                    "n_both_wrong": table[1][1],
                    "statistic": statistic,
                    "p_value": p_value,
                    "effect_size_name": "discordant odds ratio (a-only / b-only)",
                    "effect_size": (
                        float(only_first) / only_second
                        if only_second
                        else float("inf")
                        if only_first
                        else float("nan")
                    ),
                    "favours": (
                        first
                        if only_first > only_second
                        else second
                        if only_second > only_first
                        else "tie"
                    ),
                    "minimum_achievable_p": minimum_achievable_p("mcnemar", discordant),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# fold-level tests
# ---------------------------------------------------------------------------


def minimum_achievable_p(test: str, n: int) -> float:
    """The smallest two-sided p-value this sample size could ever produce.

    The point of the whole module in one number. Wilcoxon on n=5 bottoms out at
    0.0625, so a non-significant result there is a statement about the design,
    not about the models. Returns NaN where the floor is effectively zero.
    """
    if n <= 0:
        return float("nan")
    if test in ("wilcoxon", "sign", "mcnemar"):
        # Every pair falling the same way: 2 * 0.5**n, capped at 1.
        return float(min(1.0, 2.0 * (0.5**n)))
    return float("nan")


def paired_fold_tests(
    per_fold: Any, metrics: tuple[str, ...] = HEADLINE_METRICS
) -> Any:
    """T82.4 / T82.6 -- Wilcoxon and paired t on fold-level pairs, with effect sizes.

    Both tests are run and both are reported; the documented Shapiro-Wilk check
    on the paired differences decides which one the ``chosen_test`` column names.
    Running only the chosen one would make the choice unauditable.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for (run, metric), block in _metric_blocks(per_fold, metrics):
        wide = block.pivot_table(index="fold_label", columns="model_id", values=metric)
        wide = wide.dropna()
        if len(wide) < 3:
            continue
        n = len(wide)
        for first, second in combinations(sorted(wide.columns), 2):
            a = wide[first].to_numpy(dtype=float)
            b = wide[second].to_numpy(dtype=float)
            differences = a - b
            rows.append(
                {
                    "run": str(run),
                    "metric": metric,
                    "model_a": first,
                    "model_b": second,
                    "n": n,
                    "n_description": N_DESCRIPTION["fold"],
                    "mean_a": float(a.mean()),
                    "mean_b": float(b.mean()),
                    "mean_difference": float(differences.mean()),
                    "n_folds_a_wins": int(np.sum(differences > 0)),
                    "n_folds_b_wins": int(np.sum(differences < 0)),
                    "n_folds_tied": int(np.sum(differences == 0)),
                    **_normality(differences),
                    **_wilcoxon(differences, n),
                    **_paired_t(differences, n),
                }
            )

    frame = pd.DataFrame(rows)
    if not len(frame):
        return frame
    frame["chosen_test"] = np.where(
        frame["differences_are_normal"], "paired t-test", "Wilcoxon signed-rank"
    )
    frame["p_value"] = np.where(
        frame["differences_are_normal"], frame["t_p_value"], frame["wilcoxon_p_value"]
    )
    frame["statistic"] = np.where(
        frame["differences_are_normal"], frame["t_statistic"], frame["wilcoxon_statistic"]
    )
    frame["effect_size_name"] = np.where(
        frame["differences_are_normal"], "Cohen d (paired)", "rank-biserial correlation"
    )
    frame["effect_size"] = np.where(
        frame["differences_are_normal"], frame["cohen_d"], frame["rank_biserial"]
    )
    frame["minimum_achievable_p"] = [
        minimum_achievable_p("wilcoxon", int(n)) for n in frame["n"]
    ]
    frame["underpowered"] = frame["minimum_achievable_p"] >= ALPHA
    return frame


def _metric_blocks(per_fold: Any, metrics: tuple[str, ...]) -> Any:
    for run, block in per_fold.groupby("run", sort=True):
        for metric in metrics:
            if metric in block.columns and block[metric].notna().any():
                yield (run, metric), block


def _normality(differences: np.ndarray) -> dict[str, Any]:
    """Shapiro-Wilk on the paired differences, recorded rather than just used."""
    from scipy import stats

    if differences.size < 3 or np.allclose(differences, differences[0]):
        return {
            "shapiro_statistic": float("nan"),
            "shapiro_p_value": float("nan"),
            "differences_are_normal": False,
            "normality_check": (
                "Shapiro-Wilk not applicable (constant or fewer than three "
                "differences); the rank test is used"
            ),
        }
    statistic, p_value = stats.shapiro(differences)
    normal = bool(p_value > ALPHA)
    return {
        "shapiro_statistic": float(statistic),
        "shapiro_p_value": float(p_value),
        "differences_are_normal": normal,
        "normality_check": (
            "Shapiro-Wilk p="
            + format(float(p_value), ".4f")
            + (
                " > 0.05, so the paired t-test is used"
                if normal
                else " <= 0.05, so Wilcoxon signed-rank is used"
            )
        ),
    }


def _wilcoxon(differences: np.ndarray, n: int) -> dict[str, Any]:
    from scipy import stats

    if np.allclose(differences, 0.0):
        return {
            "wilcoxon_statistic": float("nan"),
            "wilcoxon_p_value": 1.0,
            "rank_biserial": 0.0,
        }
    try:
        statistic, p_value = stats.wilcoxon(differences, zero_method="wilcox")
    except ValueError:  # pragma: no cover - all-zero handled above
        return {
            "wilcoxon_statistic": float("nan"),
            "wilcoxon_p_value": float("nan"),
            "rank_biserial": float("nan"),
        }
    # Matched-pairs rank-biserial: the signed-rank sums scaled to [-1, 1].
    nonzero = differences[differences != 0]
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    total = positive + negative
    return {
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p_value": float(p_value),
        "rank_biserial": float((positive - negative) / total) if total else 0.0,
    }


def _paired_t(differences: np.ndarray, n: int) -> dict[str, Any]:
    from scipy import stats

    sd = float(differences.std(ddof=1)) if n > 1 else 0.0
    if sd == 0.0:
        return {
            "t_statistic": float("nan"),
            "t_p_value": 1.0 if np.allclose(differences, 0.0) else float("nan"),
            "cohen_d": 0.0 if np.allclose(differences, 0.0) else float("inf"),
        }
    statistic, p_value = stats.ttest_rel(differences, np.zeros_like(differences))
    return {
        "t_statistic": float(statistic),
        "t_p_value": float(p_value),
        "cohen_d": float(differences.mean() / sd),
    }


def friedman_test(
    per_fold: Any, metrics: tuple[str, ...] = HEADLINE_METRICS
) -> tuple[Any, Any]:
    """T82.5 -- Friedman across all models, with a Nemenyi post-hoc where warranted.

    Returns ``(omnibus, posthoc)``. The post-hoc is computed only where the
    omnibus rejects: running it regardless is the classic way to manufacture a
    significant pair out of a non-significant comparison.

    Kendall's W accompanies the statistic as the effect size -- the agreement
    among folds about the ranking, from 0 (no agreement) to 1 (identical
    ranking every fold).
    """
    import pandas as pd
    from scipy import stats

    omnibus_rows: list[dict[str, Any]] = []
    posthoc_rows: list[dict[str, Any]] = []

    for (run, metric), block in _metric_blocks(per_fold, metrics):
        wide = block.pivot_table(index="fold_label", columns="model_id", values=metric).dropna()
        k = int(wide.shape[1])
        n = int(wide.shape[0])
        if k < 3 or n < 3:
            continue
        statistic, p_value = stats.friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
        # Ranked per fold, best = rank 1, so a low average rank is a good model.
        ranks = wide.rank(axis=1, ascending=False)
        average = ranks.mean(axis=0)
        kendall_w = float(statistic) / (n * (k - 1))
        significant = bool(p_value < ALPHA)
        cd = nemenyi_critical_difference(k, n)
        omnibus_rows.append(
            {
                "run": str(run),
                "metric": metric,
                "test": "Friedman",
                "n": n,
                "n_description": N_DESCRIPTION["fold"],
                "k_models": k,
                "models": "; ".join(str(c) for c in wide.columns),
                "statistic": float(statistic),
                "p_value": float(p_value),
                "effect_size_name": "Kendall W (fold agreement on the ranking)",
                "effect_size": kendall_w,
                "significant": significant,
                "nemenyi_critical_difference": cd,
                "average_ranks": "; ".join(
                    str(model) + "=" + format(float(value), ".3f")
                    for model, value in average.sort_values().items()
                ),
                "posthoc_run": significant,
                "posthoc_note": (
                    "Nemenyi post-hoc computed."
                    if significant
                    else "The omnibus did not reject at alpha="
                    + str(ALPHA)
                    + ", so no post-hoc was run."
                ),
            }
        )
        if not significant:
            continue
        for first, second in combinations(sorted(wide.columns), 2):
            gap = float(abs(average[first] - average[second]))
            posthoc_rows.append(
                {
                    "run": str(run),
                    "metric": metric,
                    "test": "Nemenyi (post-hoc to Friedman)",
                    "model_a": first,
                    "model_b": second,
                    "n": n,
                    "n_description": N_DESCRIPTION["fold"],
                    "k_models": k,
                    "average_rank_a": float(average[first]),
                    "average_rank_b": float(average[second]),
                    "rank_difference": gap,
                    "critical_difference": cd,
                    "significant": bool(gap > cd),
                    "note": (
                        "Nemenyi compares average ranks against one critical "
                        "difference; it reports a decision, not a p-value."
                    ),
                }
            )

    return pd.DataFrame(omnibus_rows), pd.DataFrame(posthoc_rows)


def nemenyi_critical_difference(k: int, n: int, alpha: float = ALPHA) -> float:
    """CD = q_alpha * sqrt(k(k+1) / (6n)), with q from the studentized range.

    ``q_alpha`` is the studentized range quantile at infinite degrees of freedom
    divided by sqrt(2), which is the Nemenyi critical value. Computed rather than
    looked up in a table, so it is right for any k rather than for the handful a
    printed table covers.
    """
    from scipy import stats

    if k < 2 or n < 1:
        return float("nan")
    q = float(stats.studentized_range.ppf(1 - alpha, k, np.inf)) / np.sqrt(2.0)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


# ---------------------------------------------------------------------------
# T83.4 -- multiple-comparison correction
# ---------------------------------------------------------------------------


def correct_p_values(frame: Any, *, group_by: tuple[str, ...] = ("run", "metric")) -> Any:
    """Holm and Benjamini-Hochberg, applied within each family of comparisons.

    Both are reported. Holm controls the family-wise error rate -- the chance of
    **any** false positive -- and is the conservative choice for a claim like
    "the ensemble beats this model". Benjamini-Hochberg controls the false
    discovery rate and is the right one for screening many pairs to decide which
    are worth looking at. They answer different questions and a table that
    printed one of them unlabelled would be choosing for the reader.

    The family is a (run, metric) group: correcting across metrics would treat
    sensitivity and specificity on the same folds as independent tests, which
    they are emphatically not.
    """
    import pandas as pd
    from statsmodels.stats.multitest import multipletests

    if not len(frame):
        return frame
    keys = [k for k in group_by if k in frame.columns]
    blocks: list[pd.DataFrame] = []
    for values, block in frame.groupby(keys, sort=False):
        part = block.copy()
        valid = part["p_value"].notna()
        part["n_comparisons_in_family"] = int(valid.sum())
        part["correction_family"] = " / ".join(str(v) for v in np.atleast_1d(values))
        for method, column in (("holm", "p_holm"), ("fdr_bh", "p_bh")):
            part[column] = np.nan
            if valid.any():
                part.loc[valid, column] = multipletests(
                    part.loc[valid, "p_value"].to_numpy(dtype=float), alpha=ALPHA, method=method
                )[1]
        part["significant_raw"] = part["p_value"] < ALPHA
        part["significant_holm"] = part["p_holm"] < ALPHA
        part["significant_bh"] = part["p_bh"] < ALPHA
        blocks.append(part)
    return pd.concat(blocks, ignore_index=True)
