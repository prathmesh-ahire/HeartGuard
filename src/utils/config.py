"""YAML configuration loader (Phase 03, task T03.6).

Three things this module provides:

**Dotted-key access.** ``cfg["filter.low_hz"]`` and ``cfg.get("filter.low_hz", 20)``
instead of ``cfg["filter"]["low_hz"]``, so a missing intermediate level raises a
message naming the full path rather than a bare ``KeyError: 'filter'``.

**Environment-variable override.** Any leaf can be overridden without editing a
file, using ``HEARTGUARD__<FILE>__<KEY__PATH>``::

    HEARTGUARD__PATHS__PROJECT_ROOT=E:/HeartGuard
    HEARTGUARD__SIGNAL__RESAMPLE__TARGET_FS=4000
    HEARTGUARD__MODELS__GLOBAL__N_JOBS=4

Values are parsed as YAML scalars, so ``4000`` arrives as an int and ``true`` as
a bool. Every override that fires is recorded in :attr:`Config.overrides` so it
can be written into the run manifest -- an override that changes a result and
leaves no trace would break rule 5.

**Schema validation.** Not a generic validator: the checks encode the specific
things that must not drift in this project. The feature counts must still sum to
138. The filter passband must stay below Nyquist at the target sampling rate.
Band-power edges must stay inside the passband. An unknown top-level key is an
error rather than a silently ignored typo, because ``target_fs_hz: 2000`` next
to a defaulted ``target_fs`` is exactly the kind of mistake that produces a
plausible wrong number.

Depends only on PyYAML (core requirements). Not on pydantic, which is API-only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "Config",
    "ConfigError",
    "ConfigValidationError",
    "CONFIG_NAMES",
    "load_config",
    "load_all",
    "clear_cache",
]

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
ENV_PREFIX = "HEARTGUARD__"

CONFIG_NAMES = ("paths", "signal", "features", "models", "experiments")

# The locked 138. Docs/note.md, 2026-08-22.
EXPECTED_FEATURE_TOTAL = 138
EXPECTED_FAMILY_COUNTS = {
    "time": 24,
    "frequency": 22,
    "mfcc": 39,
    "chroma": 24,
    "dwt": 24,
    "envelope": 5,
}


class ConfigError(Exception):
    """Raised when a configuration file cannot be loaded or a key is missing."""


class ConfigValidationError(ConfigError):
    """Raised when a configuration file loads but violates its schema."""

    def __init__(self, name: str, problems: list[str]) -> None:
        self.name = name
        self.problems = problems
        detail = "\n".join("  - " + p for p in problems)
        super().__init__(
            "configs/" + name + ".yaml failed validation ("
            + str(len(problems))
            + " problem(s)):\n"
            + detail
        )


# ---------------------------------------------------------------------------
# dotted-key helpers
# ---------------------------------------------------------------------------

_MISSING = object()


def _dig(data: Any, dotted: str) -> Any:
    """Walk a dotted path, returning ``_MISSING`` if any level is absent."""
    node = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _match_key(node: dict, part: str) -> str:
    """Resolve one path segment against an existing mapping, case-insensitively.

    Environment variable names are conventionally upper case, but YAML keys here
    are not uniformly so -- experiment ids are ``EXP-A1``, model ids are ``M3``.
    Lower-casing the whole path would make ``HEARTGUARD__EXPERIMENTS__
    EXPERIMENTS__EXP-D1__RETUNING_ALLOWED`` create a *new* ``exp-d1`` entry
    beside the real ``EXP-D1`` rather than overriding it -- an override that
    appears to work and changes nothing.
    """
    if part in node:
        return part
    lowered = part.lower()
    for key in node:
        if isinstance(key, str) and key.lower() == lowered:
            return key
    return part


def _plant(data: dict, dotted: str, value: Any) -> None:
    """Set a dotted path, creating intermediate dicts as needed.

    Segments resolve against existing keys case-insensitively, so an override
    targets the real key rather than shadowing it.
    """
    parts = dotted.split(".")
    node = data
    for part in parts[:-1]:
        key = _match_key(node, part)
        nxt = node.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            node[key] = nxt
        node = nxt
    node[_match_key(node, parts[-1])] = value


class Config:
    """A loaded configuration file with dotted-key access."""

    def __init__(self, name: str, data: dict, overrides: dict[str, Any] | None = None) -> None:
        self.name = name
        self._data = data
        self.overrides = overrides or {}

    # -- access -------------------------------------------------------------

    def get(self, dotted: str, default: Any = None) -> Any:
        """Return the value at ``dotted``, or ``default`` if it is absent."""
        found = _dig(self._data, dotted)
        return default if found is _MISSING else found

    def require(self, dotted: str) -> Any:
        """Return the value at ``dotted``, raising if it is absent."""
        found = _dig(self._data, dotted)
        if found is _MISSING:
            raise ConfigError(
                "configs/" + self.name + ".yaml has no key " + repr(dotted)
            )
        return found

    def __getitem__(self, dotted: str) -> Any:
        return self.require(dotted)

    def __contains__(self, dotted: str) -> bool:
        return _dig(self._data, dotted) is not _MISSING

    def as_dict(self) -> dict:
        """The raw nested dict. Treat as read-only."""
        return self._data

    def keys(self) -> list[str]:
        """Top-level keys."""
        return list(self._data.keys())

    def leaf_paths(self) -> list[str]:
        """Every dotted path that resolves to a non-dict value."""
        out: list[str] = []

        def walk(node: Any, prefix: str) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value, prefix + "." + str(key) if prefix else str(key))
            else:
                out.append(prefix)

        walk(self._data, "")
        return out

    def __repr__(self) -> str:
        return "Config(" + repr(self.name) + ", " + str(len(self._data)) + " top-level keys)"


# ---------------------------------------------------------------------------
# environment overrides
# ---------------------------------------------------------------------------


def _apply_env_overrides(name: str, data: dict) -> dict[str, Any]:
    """Apply ``HEARTGUARD__<NAME>__<PATH>`` overrides in place.

    Returns the overrides that fired, keyed by dotted path.
    """
    applied: dict[str, Any] = {}
    prefix = ENV_PREFIX + name.upper() + "__"
    for env_key, raw in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        dotted = env_key[len(prefix) :].lower().replace("__", ".")
        if not dotted:
            continue
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        _plant(data, dotted, value)
        applied[dotted] = value
    return applied


# ---------------------------------------------------------------------------
# path resolution
# ---------------------------------------------------------------------------


# Keys inside paths.yaml whose string values are NOT paths and must be left
# alone. Declared explicitly rather than sniffed: a "does this look like a
# path?" heuristic gets it wrong in both directions here -- it misses bare
# roots like "outputs" and "cache", and it wrongly absolutises bare filenames
# like reference_filename: "REFERENCE.csv", which is joined per-subset at load
# time and is not a path relative to the project root at all.
NON_PATH_KEYS = frozenset(
    {
        "subsets",
        "classes",
        "reference_filename",
        "reference_sqi_filename",
        "is_training_source",
    }
)

# Top-level sections of paths.yaml whose string leaves are paths.
PATH_SECTIONS = frozenset({"dataset", "cache", "outputs", "models_saved", "frontend"})


def _resolve_paths(data: dict) -> None:
    """Make every path in paths.yaml absolute, relative to ``project_root``.

    T03.1 requires downstream code to receive absolute paths; declaring them
    relative in the file keeps a machine move to a single edit.
    """
    root_raw = data.get("project_root")
    if not isinstance(root_raw, str):
        return
    root = Path(root_raw).expanduser().resolve()
    data["project_root"] = str(root)

    def walk(node: Any, key: str | None) -> Any:
        if key in NON_PATH_KEYS:
            return node
        if isinstance(node, dict):
            return {k: walk(v, str(k)) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v, key) for v in node]
        if isinstance(node, str):
            candidate = Path(node)
            if not candidate.is_absolute():
                candidate = root / candidate
            return str(candidate)
        return node

    for key in list(data.keys()):
        if key in PATH_SECTIONS:
            data[key] = walk(data[key], key)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

# Allowed top-level keys per file. Anything else is a typo, not a feature.
ALLOWED_TOP_LEVEL: dict[str, set[str]] = {
    "paths": {"project_root", "dataset", "cache", "outputs", "models_saved", "frontend"},
    "signal": {"resample", "filter", "normalization", "quality", "framing", "cache"},
    "features": {"expected_total", "failure_policy", "families", "extraction"},
    "models": {"global", "pipeline", "calibration", "models"},
    "experiments": {"defaults", "cv_schemes", "experiments"},
}

# (dotted path, python type) pairs that must exist with that type.
REQUIRED: dict[str, list[tuple[str, type | tuple[type, ...]]]] = {
    "paths": [
        ("project_root", str),
        ("dataset.d1_physionet.root", str),
        ("dataset.d2_pascal_a.root", str),
        ("dataset.d3_pascal_b.root", str),
        ("dataset.d4_circor.root", str),
        ("cache.root", str),
        ("outputs.root", str),
        ("outputs.missing_outputs_report", str),
        ("models_saved", str),
    ],
    "signal": [
        ("resample.target_fs", int),
        ("resample.method", str),
        ("filter.enabled", bool),
        ("filter.type", str),
        ("filter.order", int),
        ("filter.low_hz", (int, float)),
        ("filter.high_hz", (int, float)),
        ("normalization.enabled", bool),
        ("normalization.method", str),
        ("framing.frame_length", int),
        ("framing.hop_length", int),
    ],
    "features": [
        ("expected_total", int),
        ("failure_policy.on_error", str),
        ("families.time.count", int),
        ("families.frequency.count", int),
        ("families.mfcc.count", int),
        ("families.chroma.count", int),
        ("families.dwt.count", int),
        ("families.envelope.count", int),
        ("families.mfcc.n_mfcc", int),
        ("families.mfcc.fmax", (int, float)),
        ("families.chroma.n_chroma", int),
        ("families.dwt.wavelet", str),
        ("families.dwt.level", int),
    ],
    "models": [
        ("global.random_state", int),
        ("pipeline.imputer.strategy", str),
        ("pipeline.scaler.kind", str),
        ("models.M1.estimator", str),
        ("models.M3.estimator", str),
        ("models.M4.estimator", str),
        ("models.M5.estimator", str),
    ],
    "experiments": [
        ("defaults.seed", int),
        ("cv_schemes.repeated_5x5_grouped.n_splits", int),
        ("cv_schemes.repeated_5x5_grouped.n_repeats", int),
        ("experiments.EXP-A1.dataset", str),
        ("experiments.EXP-A1.task", str),
        ("experiments.EXP-A2.task", str),
        ("experiments.EXP-B1.task", str),
        ("experiments.EXP-B2.task", str),
        ("experiments.EXP-C1.task", str),
        ("experiments.EXP-C2.task", str),
        ("experiments.EXP-D1.retuning_allowed", bool),
    ],
}

# The five label spaces. Rule 4: they are never merged.
VALID_TASKS = {
    "binary",
    "pascal_a",
    "pascal_b",
    "circor_murmur",
    "circor_outcome",
    "circor_murmur_and_outcome",
    "diagnosis_multiclass",
}


def _type_name(expected: type | tuple[type, ...]) -> str:
    if isinstance(expected, tuple):
        return " or ".join(t.__name__ for t in expected)
    return expected.__name__


def _validate(name: str, data: dict) -> list[str]:
    problems: list[str] = []

    # -- unknown top-level keys ------------------------------------------
    allowed = ALLOWED_TOP_LEVEL.get(name)
    if allowed is not None:
        for key in data:
            if key not in allowed:
                problems.append(
                    "unknown top-level key " + repr(key) + " (expected one of: "
                    + ", ".join(sorted(allowed)) + ")"
                )

    # -- required keys and their types -----------------------------------
    for dotted, expected in REQUIRED.get(name, []):
        found = _dig(data, dotted)
        if found is _MISSING:
            problems.append("missing required key " + repr(dotted))
            continue
        # bool is a subclass of int; reject it where an int is wanted.
        if expected is int and isinstance(found, bool):
            problems.append(dotted + ": expected int, got bool")
            continue
        if not isinstance(found, expected):
            problems.append(
                dotted + ": expected " + _type_name(expected)
                + ", got " + type(found).__name__
            )

    # -- file-specific invariants ----------------------------------------
    if name == "signal":
        problems += _validate_signal(data)
    elif name == "features":
        problems += _validate_features(data)
    elif name == "models":
        problems += _validate_models(data)
    elif name == "experiments":
        problems += _validate_experiments(data)

    return problems


def _validate_signal(data: dict) -> list[str]:
    problems: list[str] = []
    fs = _dig(data, "resample.target_fs")
    low = _dig(data, "filter.low_hz")
    high = _dig(data, "filter.high_hz")
    order = _dig(data, "filter.order")

    if isinstance(fs, int) and fs <= 0:
        problems.append("resample.target_fs must be positive, got " + str(fs))

    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        if low >= high:
            problems.append(
                "filter.low_hz (" + str(low) + ") must be below filter.high_hz (" + str(high) + ")"
            )
        if isinstance(fs, int) and fs > 0:
            nyquist = fs / 2
            if high >= nyquist:
                problems.append(
                    "filter.high_hz (" + str(high) + ") must be below Nyquist ("
                    + str(nyquist) + " at target_fs " + str(fs) + ")"
                )
        if low <= 0:
            problems.append("filter.low_hz must be positive, got " + str(low))

    if isinstance(order, int) and not 1 <= order <= 10:
        problems.append("filter.order " + str(order) + " outside the sane range 1-10")

    method = _dig(data, "normalization.method")
    if isinstance(method, str) and method not in {"zscore", "peak", "none"}:
        problems.append(
            "normalization.method " + repr(method) + " must be zscore, peak or none"
        )

    frame = _dig(data, "framing.frame_length")
    hop = _dig(data, "framing.hop_length")
    if isinstance(frame, int) and isinstance(hop, int) and hop > frame:
        problems.append(
            "framing.hop_length (" + str(hop) + ") exceeds frame_length (" + str(frame) + ")"
        )
    return problems


def _validate_features(data: dict) -> list[str]:
    problems: list[str] = []
    total = _dig(data, "expected_total")
    if isinstance(total, int) and total != EXPECTED_FEATURE_TOTAL:
        problems.append(
            "expected_total is " + str(total) + " but the locked composition is "
            + str(EXPECTED_FEATURE_TOTAL) + " -- changing it invalidates every cached "
            "matrix, trained model and ablation result"
        )

    running = 0
    for family, expected_count in EXPECTED_FAMILY_COUNTS.items():
        count = _dig(data, "families." + family + ".count")
        if count is _MISSING:
            problems.append("missing families." + family + ".count")
            continue
        if count != expected_count:
            problems.append(
                "families." + family + ".count is " + str(count)
                + " but the locked count is " + str(expected_count)
            )
        if isinstance(count, int):
            running += count
    if running and running != EXPECTED_FEATURE_TOTAL:
        problems.append(
            "family counts sum to " + str(running) + ", not " + str(EXPECTED_FEATURE_TOTAL)
        )

    # Every family's declared composition must add up to its own count.
    families = _dig(data, "families")
    if isinstance(families, dict):
        for family, spec in families.items():
            if not isinstance(spec, dict):
                continue
            comp = spec.get("composition")
            count = spec.get("count")
            if isinstance(comp, dict) and isinstance(count, int):
                parts = [v for v in comp.values() if isinstance(v, int)]
                if len(parts) == len(comp) and sum(parts) != count:
                    problems.append(
                        "families." + family + ".composition sums to " + str(sum(parts))
                        + " but count is " + str(count)
                    )

    # The mel filterbank must stay below Nyquist. librosa's defaults assume
    # 22 kHz speech and would place most filters above Nyquist at 2 kHz.
    fmax = _dig(data, "families.mfcc.fmax")
    fmin = _dig(data, "families.mfcc.fmin")
    if isinstance(fmax, (int, float)) and fmax >= 1000:
        problems.append(
            "families.mfcc.fmax (" + str(fmax) + ") is at or above Nyquist (1000 Hz at "
            "the 2 kHz target rate)"
        )
    if isinstance(fmin, (int, float)) and isinstance(fmax, (int, float)) and fmin >= fmax:
        problems.append(
            "families.mfcc.fmin (" + str(fmin) + ") must be below fmax (" + str(fmax) + ")"
        )

    # Band-power edges must be contiguous and inside the passband.
    bands = _dig(data, "families.frequency.bands_hz")
    n_bands = _dig(data, "families.frequency.composition.band_powers")
    if isinstance(bands, list):
        if isinstance(n_bands, int) and len(bands) != n_bands:
            problems.append(
                "families.frequency.bands_hz has " + str(len(bands)) + " bands but "
                "composition.band_powers declares " + str(n_bands)
            )
        for band in bands:
            if not (isinstance(band, list) and len(band) == 2):
                problems.append("malformed band in families.frequency.bands_hz: " + repr(band))
                continue
            lo, hi = band
            if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)) or lo >= hi:
                problems.append("band " + repr(band) + " is not an increasing [low, high] pair")

    level = _dig(data, "families.dwt.level")
    subbands = _dig(data, "families.dwt.subbands")
    if isinstance(level, int) and isinstance(subbands, list) and len(subbands) != level + 1:
        problems.append(
            "families.dwt.level " + str(level) + " implies " + str(level + 1)
            + " sub-bands but " + str(len(subbands)) + " are listed"
        )

    policy = _dig(data, "failure_policy.on_error")
    if isinstance(policy, str) and policy != "nan":
        problems.append(
            "failure_policy.on_error must be 'nan' -- extractors return NaN on "
            "failure and never raise (T31.4)"
        )
    return problems


def _validate_models(data: dict) -> list[str]:
    problems: list[str] = []
    seed = _dig(data, "global.random_state")
    if seed != 42:
        problems.append(
            "global.random_state is " + repr(seed) + ", not 42 -- the seed is fixed "
            "project-wide (rule 5)"
        )

    models = _dig(data, "models")
    if isinstance(models, dict):
        for model_id, spec in models.items():
            if not isinstance(spec, dict):
                problems.append("models." + str(model_id) + " is not a mapping")
                continue
            for field in ("name", "mandatory"):
                if field not in spec:
                    problems.append("models." + str(model_id) + " is missing " + repr(field))
            state = spec.get("defaults", {})
            if isinstance(state, dict) and "random_state" in state and state["random_state"] != 42:
                problems.append(
                    "models." + str(model_id) + ".defaults.random_state is "
                    + repr(state["random_state"]) + ", not 42"
                )

        # M6/M7 fuse calibrated base models; their members must exist.
        for ens in ("M6", "M7"):
            spec = models.get(ens)
            if isinstance(spec, dict):
                for member in spec.get("members", []) or []:
                    if member not in models:
                        problems.append(
                            "models." + ens + " lists member " + repr(member)
                            + " which is not defined"
                        )

    if _dig(data, "calibration.fit_inside_fold") is False:
        problems.append(
            "calibration.fit_inside_fold is false -- the calibrator must fit on the "
            "training fold only (rule 2)"
        )
    return problems


def _validate_experiments(data: dict) -> list[str]:
    problems: list[str] = []
    schemes = _dig(data, "cv_schemes")
    experiments = _dig(data, "experiments")

    if isinstance(schemes, dict):
        for scheme_id, spec in schemes.items():
            if not isinstance(spec, dict):
                continue
            splits = spec.get("n_splits")
            repeats = spec.get("n_repeats", 1)
            total = spec.get("total_folds")
            if (
                isinstance(splits, int)
                and isinstance(repeats, int)
                and isinstance(total, int)
                and splits * repeats != total
            ):
                problems.append(
                    "cv_schemes." + str(scheme_id) + ": n_splits x n_repeats = "
                    + str(splits * repeats) + " but total_folds is " + str(total)
                )

    if isinstance(experiments, dict):
        for exp_id, spec in experiments.items():
            if not isinstance(spec, dict):
                problems.append("experiments." + str(exp_id) + " is not a mapping")
                continue

            task = spec.get("task")
            if task is not None and task not in VALID_TASKS:
                problems.append(
                    "experiments." + str(exp_id) + ".task " + repr(task)
                    + " is not one of the defined label spaces: "
                    + ", ".join(sorted(VALID_TASKS))
                )

            cv = spec.get("cv")
            if cv is not None and isinstance(schemes, dict) and cv not in schemes:
                problems.append(
                    "experiments." + str(exp_id) + ".cv " + repr(cv)
                    + " is not a defined cv_scheme"
                )

            for dep in spec.get("depends_on", []) or []:
                if isinstance(experiments, dict) and dep not in experiments:
                    problems.append(
                        "experiments." + str(exp_id) + " depends on " + repr(dep)
                        + " which is not defined"
                    )

    # EXP-D1 must never retune on CirCor, and must carry its population note
    # before any metric exists.
    d1 = _dig(data, "experiments.EXP-D1")
    if isinstance(d1, dict):
        if d1.get("retuning_allowed") is not False:
            problems.append(
                "experiments.EXP-D1.retuning_allowed must be false -- no retuning of "
                "any kind on CirCor"
            )
        note = d1.get("population_note")
        if not isinstance(note, str) or "paediatric" not in note.lower():
            problems.append(
                "experiments.EXP-D1.population_note must record the adult-to-paediatric "
                "population mismatch before any metric exists (T71.1)"
            )
    return problems


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

_CACHE: dict[str, Config] = {}


def load_config(name: str, *, validate: bool = True, use_cache: bool = True) -> Config:
    """Load ``configs/<name>.yaml``.

    Applies environment overrides, resolves paths to absolute (for ``paths``),
    and validates unless ``validate=False``.
    """
    if use_cache and name in _CACHE:
        return _CACHE[name]

    path = CONFIG_DIR / (name + ".yaml")
    if not path.is_file():
        raise ConfigError("no such config file: " + str(path))

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError("configs/" + name + ".yaml is not valid YAML: " + str(exc)) from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(
            "configs/" + name + ".yaml must be a mapping at the top level, got "
            + type(raw).__name__
        )

    overrides = _apply_env_overrides(name, raw)

    if name == "paths":
        _resolve_paths(raw)

    if validate:
        problems = _validate(name, raw)
        if problems:
            raise ConfigValidationError(name, problems)

    cfg = Config(name, raw, overrides)
    if use_cache:
        _CACHE[name] = cfg
    return cfg


def load_all(*, validate: bool = True, use_cache: bool = True) -> dict[str, Config]:
    """Load every configuration file, keyed by name."""
    return {n: load_config(n, validate=validate, use_cache=use_cache) for n in CONFIG_NAMES}


def clear_cache() -> None:
    """Drop cached configs so the next load re-reads from disk."""
    _CACHE.clear()
