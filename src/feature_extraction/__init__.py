"""The 138 engineered features: time, frequency, MFCC, chroma, DWT, envelope.

Import the registry from here; it is the only thing in this package that is safe
to touch before a family's own module exists::

    from src.feature_extraction import FEATURE_NAMES, get_extractor

The family modules are imported on demand by
:func:`src.feature_extraction.registry.get_extractor`, so this package imports
cleanly while Phases 32-37 are still being built.
"""

from src.feature_extraction.base import (
    BaseFeatureExtractor,
    FamilyResult,
    FeatureExtractor,
    reset_timings,
    timing_table,
)
from src.feature_extraction.registry import (
    EXPECTED_FAMILY_COUNTS,
    EXPECTED_TOTAL,
    FAMILY_ORDER,
    FEATURE_NAMES,
    FEATURE_SPECS,
    FeatureSpec,
    family_counts,
    feature_names,
    get_extractor,
    registry_fingerprint,
)

__all__ = [
    "BaseFeatureExtractor",
    "FamilyResult",
    "FeatureExtractor",
    "FeatureSpec",
    "EXPECTED_FAMILY_COUNTS",
    "EXPECTED_TOTAL",
    "FAMILY_ORDER",
    "FEATURE_NAMES",
    "FEATURE_SPECS",
    "family_counts",
    "feature_names",
    "get_extractor",
    "registry_fingerprint",
    "reset_timings",
    "timing_table",
]
