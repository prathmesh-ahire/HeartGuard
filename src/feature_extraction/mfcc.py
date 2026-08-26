"""MFCC features: the 39 of Phase 34.

Thirteen coefficient means, thirteen standard deviations and thirteen
first-order delta means, in that order -- blocked by statistic rather than
interleaved by coefficient, so a contiguous column slice is one statistic across
all thirteen coefficients.

**The filterbank had to be rebuilt for this signal, not borrowed.** librosa's
defaults assume 22 kHz speech: ``fmin=0``, ``fmax=sr/2``, 128 mel bands. Applied
to a 2 kHz PCG that places almost every mel filter above the 400 Hz where heart
sounds actually live, and the resulting coefficients would describe an empty
band. The bank here spans exactly the passband the preprocessing produced --
``fmin=20``, ``fmax=400``, 40 bands -- and ``fmax`` is checked against Nyquist on
every call rather than assumed (T34.1, T34.7).

Two facts make that configuration work at this sampling rate, and both were
verified rather than hoped for: at ``n_fft=512`` and 2 kHz the FFT resolution is
3.906 Hz, so each of the 40 bands covers 4-5 bins and **none of them is empty**;
and with ``htk=false`` the Slaney mel scale is *linear* below 1 kHz, so 20-400 Hz
is a linear 40-band split rather than the log-spaced one the name suggests. An
empty filter would take the log of zero and put a floor constant into a
coefficient for every record -- silent, and indistinguishable from a real value.

**``top_db`` is disabled, deliberately.** ``librosa.power_to_db`` clips by
default at 80 dB below the *maximum of the spectrogram it is given*, which is a
per-record quantity. Two recordings with identical spectral shape and different
peak levels would then produce different MFCCs. The floor here is the absolute
``amin`` instead, so the mapping from spectrum to coefficients is the same
function for every record in the corpus -- which is what rule 5 requires and
what makes two records comparable at all.

**``mfcc_01`` is c0, the log-energy term**, not the first "shape" coefficient.
It is kept because the locked count is 39 and both source documents list 13
coefficients, but it carries overall band energy rather than spectral shape, and
the signal is z-normalized before it is computed. Read it as a normalizer
artifact unless Phase 41 says otherwise.

**Short recordings degrade, they never raise (T34.6).** The delta needs at least
``delta_width`` frames; a 0.76 s record yields six. The configured policy is
``shrink_delta_width_then_pad``: shrink to the largest odd width the frame count
allows, and only if that is still impossible (fewer than three frames) replicate
the edge frames to reach three. Every reduction is flagged on the record.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from src.feature_extraction.base import BaseFeatureExtractor
from src.feature_extraction.registry import feature_names, register_extractor

__all__ = [
    "MFCCExtractor",
    "MFCCSettings",
    "extract_mfcc_features",
    "mfcc_matrix",
    "delta_matrix",
    "resolve_delta_width",
]

FAMILY = "mfcc"

#: Below this many samples there is no spectrum worth estimating. The shortest
#: real record is 1,520 samples; this path exists for the inference API, where an
#: upload can be any length.
MIN_MFCC_SAMPLES = 32

#: The smallest usable Savitzky-Golay window librosa will accept.
MIN_DELTA_WIDTH = 3


class MFCCSettings:
    """The filterbank and framing parameters, resolved from config once.

    A small class rather than a dict so the Nyquist check has an obvious home and
    runs before any of the values are used.
    """

    __slots__ = (
        "n_mfcc",
        "n_mels",
        "fmin",
        "fmax",
        "n_fft",
        "hop_length",
        "htk",
        "delta_width",
        "window",
        "center",
    )

    def __init__(self, features_cfg: Any, signal_cfg: Any) -> None:
        self.n_mfcc = int(features_cfg.get("families.mfcc.n_mfcc", 13))
        self.n_mels = int(features_cfg.get("families.mfcc.n_mels", 40))
        self.fmin = float(features_cfg.get("families.mfcc.fmin", 20))
        self.fmax = float(features_cfg.get("families.mfcc.fmax", 400))
        # n_fft and hop come from the mfcc section, not from `framing`: the
        # family is allowed its own analysis window. window and center are not
        # declared there, so they follow the project-wide framing settings.
        self.n_fft = int(features_cfg.get("families.mfcc.n_fft", 512))
        self.hop_length = int(features_cfg.get("families.mfcc.hop_length", 256))
        self.htk = bool(features_cfg.get("families.mfcc.htk", False))
        self.delta_width = int(features_cfg.get("families.mfcc.delta_width", 9))
        self.window = str(signal_cfg.get("framing.window", "hann"))
        self.center = bool(signal_cfg.get("framing.center", True))

    def check_nyquist(self, fs: int) -> None:
        """T34.1 -- ``fmax`` must stay below Nyquist, checked, not assumed.

        Above Nyquist the top mel filters address FFT bins that do not exist;
        librosa clamps them and the affected coefficients quietly become a
        function of the clamp rather than of the recording.
        """
        nyquist = fs / 2.0
        if not self.fmax < nyquist:
            raise ValueError(
                "mfcc fmax=" + str(self.fmax) + " Hz is not below Nyquist ("
                + str(nyquist) + " Hz at fs=" + str(fs) + ")"
            )
        if not 0.0 <= self.fmin < self.fmax:
            raise ValueError(
                "mfcc fmin=" + str(self.fmin) + " must be in [0, fmax=" + str(self.fmax) + ")"
            )


# ---------------------------------------------------------------------------
# the two matrices, exposed for testing
# ---------------------------------------------------------------------------


def mfcc_matrix(
    signal: np.ndarray,
    fs: int,
    settings: MFCCSettings,
    flags: list[str] | None = None,
) -> np.ndarray:
    """``(n_mfcc, n_frames)`` MFCCs over a 20-400 Hz mel filterbank (T34.1, T34.2).

    Returns an empty array when the signal is too short to frame at all; the
    caller turns that into 39 flagged NaN rather than a fabricated zero.
    """
    import librosa

    settings.check_nyquist(fs)
    values = np.asarray(signal, dtype=np.float64)

    if values.size < MIN_MFCC_SAMPLES:
        if flags is not None:
            flags.append("mfcc_signal_too_short")
        return np.empty((settings.n_mfcc, 0), dtype=np.float64)

    n_fft = settings.n_fft
    hop_length = settings.hop_length
    n_mels = settings.n_mels

    if values.size < n_fft:
        n_fft = int(2 ** np.floor(np.log2(values.size)))
        hop_length = max(1, n_fft // 2)
        if flags is not None:
            flags.append("mfcc_n_fft_" + str(n_fft))
        # A shorter window is a coarser spectrum, and 40 mel bands over 20-400 Hz
        # would then contain empty filters whose log is a floor constant. Cap the
        # bank at the number of FFT bins actually inside the passband, never
        # below n_mfcc -- fewer bands than coefficients makes the DCT degenerate.
        bins_in_band = int(
            np.count_nonzero(
                (np.fft.rfftfreq(n_fft, 1.0 / fs) >= settings.fmin)
                & (np.fft.rfftfreq(n_fft, 1.0 / fs) <= settings.fmax)
            )
        )
        capped = max(settings.n_mfcc, min(n_mels, bins_in_band))
        if capped != n_mels:
            n_mels = capped
            if flags is not None:
                flags.append("mfcc_n_mels_" + str(n_mels))

    mel = librosa.feature.melspectrogram(
        y=values,
        sr=fs,
        n_fft=n_fft,
        hop_length=hop_length,
        window=settings.window,
        center=settings.center,
        n_mels=n_mels,
        fmin=settings.fmin,
        fmax=settings.fmax,
        htk=settings.htk,
        power=2.0,
    )
    # ref=1.0 and top_db=None: an absolute, record-independent dB mapping. See
    # the module docstring -- the librosa default would make the floor depend on
    # each recording's own peak.
    log_mel = librosa.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)
    coefficients = librosa.feature.mfcc(
        S=log_mel, n_mfcc=settings.n_mfcc, dct_type=2, norm="ortho"
    )
    return np.asarray(coefficients, dtype=np.float64)


def resolve_delta_width(
    configured: int, n_frames: int, flags: list[str] | None = None
) -> int:
    """Largest usable odd delta width for ``n_frames`` frames (T34.6).

    Returns 0 when even the minimum window does not fit, which tells the caller
    to pad instead.
    """
    if n_frames >= configured:
        return configured

    usable = n_frames if n_frames % 2 == 1 else n_frames - 1
    if usable < MIN_DELTA_WIDTH:
        return 0

    if flags is not None:
        flags.append("mfcc_delta_width_" + str(usable))
    return usable


def delta_matrix(
    coefficients: np.ndarray,
    configured_width: int,
    flags: list[str] | None = None,
) -> np.ndarray:
    """First-order delta MFCCs, degrading rather than raising (T34.5, T34.6).

    ``librosa.feature.delta`` refuses a window wider than the frame count. The
    configured policy is ``shrink_delta_width_then_pad``: shrink first, and pad
    by edge replication only when three frames are not available. Padding by
    replication means the delta over the padded region is exactly zero -- for a
    recording with one or two frames that is the honest answer, because nothing
    changed between frames that do not exist.
    """
    import librosa

    if coefficients.size == 0 or coefficients.shape[1] == 0:
        return np.empty_like(coefficients)

    n_frames = int(coefficients.shape[1])
    width = resolve_delta_width(configured_width, n_frames, flags)

    if width == 0:
        pad = MIN_DELTA_WIDTH - n_frames
        left = pad // 2
        right = pad - left
        padded = np.pad(coefficients, ((0, 0), (left, right)), mode="edge")
        if flags is not None:
            flags.append("mfcc_delta_padded_" + str(n_frames) + "to" + str(MIN_DELTA_WIDTH))
        deltas = librosa.feature.delta(padded, width=MIN_DELTA_WIDTH)
        return np.asarray(deltas[:, left : left + n_frames], dtype=np.float64)

    return np.asarray(librosa.feature.delta(coefficients, width=width), dtype=np.float64)


# ---------------------------------------------------------------------------
# the family
# ---------------------------------------------------------------------------


class MFCCExtractor(BaseFeatureExtractor):
    """The 39 MFCC features (T34.1-T34.6)."""

    family = FAMILY
    name = "mfcc"

    def __init__(self, cfg: Any | None = None, signal_cfg: Any | None = None) -> None:
        super().__init__(cfg)
        self._signal_cfg = signal_cfg
        self._settings: MFCCSettings | None = None

    @property
    def _features(self) -> Any:
        if self._cfg is None:
            from src.utils.config import load_config

            self._cfg = load_config("features")
        return self._cfg

    @property
    def _signal(self) -> Any:
        if self._signal_cfg is None:
            from src.utils.config import load_config

            self._signal_cfg = load_config("signal")
        return self._signal_cfg

    def settings(self) -> MFCCSettings:
        if self._settings is None:
            self._settings = MFCCSettings(self._features, self._signal)
        return self._settings

    def feature_names(self) -> tuple[str, ...]:
        return feature_names(FAMILY)

    def _compute(
        self, signal: np.ndarray, fs: int, flags: list[str]
    ) -> Mapping[str, float]:
        settings = self.settings()
        names = self.feature_names()

        coefficients = mfcc_matrix(signal, fs, settings, flags)
        if coefficients.shape[1] == 0:
            return dict.fromkeys(names, float("nan"))

        deltas = delta_matrix(coefficients, settings.delta_width, flags)

        values: dict[str, float] = {}
        for index in range(settings.n_mfcc):
            key = "mfcc_" + str(index + 1).zfill(2)
            row = coefficients[index]
            values[key + "_mean"] = float(np.mean(row))
            # ddof=0 throughout the project; a single frame has zero spread, not
            # an undefined one.
            values[key + "_std"] = float(np.std(row)) if row.size > 1 else 0.0
            values[key + "_delta_mean"] = float(np.mean(deltas[index]))

        return values


def extract_mfcc_features(
    signal: np.ndarray, fs: int, *, record_uid: str | None = None, cfg: Any | None = None
):
    """Convenience wrapper returning a :class:`FamilyResult`."""
    return MFCCExtractor(cfg).extract(signal, fs, record_uid=record_uid)


register_extractor(MFCCExtractor())
