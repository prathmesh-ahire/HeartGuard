"""The one plotting style for every figure in this project (T28.6).

Thirty-five graphs, six preprocessing figures and twenty diagrams end up side by
side in a thesis and a paper. If half of them are matplotlib blue on a white
grid and half are seaborn pastel, the reader notices the inconsistency before
they notice the result. This module is imported by every figure-producing
module in the project -- Phase 28 here, Phase 42's feature plots, Phase 90's
``src/reporting/graphs.py`` -- so a style decision is made once.

**Serif, because the documents are.** The thesis and the Q1 paper are set in a
serif face; a sans-serif axis label in the middle of a serif page reads as a
screenshot from somewhere else. DejaVu Serif is the default here because it
ships *with matplotlib*: naming Times New Roman first would render correctly on
this machine and silently fall back to sans-serif with a warning on the grader's.

**Okabe-Ito, because the reader may be colourblind.** The eight-colour palette
below is Okabe & Ito's, designed to stay distinguishable under protanopia,
deuteranopia and tritanopia -- about 1 man in 12 has one of them. It is used in
a fixed order so that "the first series" is the same colour in every figure. For
continuous data the colormap is ``viridis``, which is perceptually uniform and
survives greyscale printing; ``jet`` and ``rainbow`` do neither and are never
used in this project.

**300 dpi, because print.** Journals reject 72 dpi raster figures.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

__all__ = [
    "DPI",
    "OKABE_ITO",
    "SEQUENTIAL_CMAP",
    "DIVERGING_CMAP",
    "FIGSIZE",
    "RC_PARAMS",
    "apply_style",
    "styled",
    "figure",
    "save_figure",
    "annotate_source",
    "class_color",
]

DPI = 300

# Okabe & Ito (2008), "Color Universal Design". Order is fixed and meaningful:
# index 0 is the first/reference series in every figure that has one.
OKABE_ITO: tuple[str, ...] = (
    "#0072B2",  # blue          -- series 1 / normal
    "#D55E00",  # vermillion    -- series 2 / abnormal
    "#009E73",  # bluish green  -- series 3
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)

SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP = "RdBu_r"

# Widths chosen for the printed page rather than the screen: "single" is one
# journal column, "double" spans both, "wide" is a full-width thesis figure.
FIGSIZE: dict[str, tuple[float, float]] = {
    "single": (3.5, 2.6),
    "double": (7.2, 3.2),
    "wide": (10.0, 4.0),
    "tall": (7.2, 6.0),
    "square": (5.0, 5.0),
}

RC_PARAMS: dict[str, Any] = {
    # -- fonts ---------------------------------------------------------
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Liberation Serif", "Times New Roman", "serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
    "mathtext.fontset": "dejavuserif",
    # -- output --------------------------------------------------------
    "figure.dpi": 150,          # on-screen; save_figure overrides to 300
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    # -- axes ----------------------------------------------------------
    "axes.grid": True,
    "axes.grid.axis": "both",
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.axisbelow": True,      # grid behind the data, never over it
    "axes.titlelocation": "left",
    "axes.titleweight": "normal",
    # -- lines and images ----------------------------------------------
    "lines.linewidth": 1.0,
    "lines.solid_capstyle": "round",
    "image.cmap": SEQUENTIAL_CMAP,
    "legend.frameon": False,
    "errorbar.capsize": 2.0,
}


def apply_style() -> None:
    """Install the project style globally, on the Agg backend.

    Agg is forced because every figure in this project is written to a file and
    never shown. Without it, importing a plotting module on a machine with a GUI
    backend configured can block on a window that nobody is looking at.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(RC_PARAMS)  # type: ignore[arg-type]
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=list(OKABE_ITO))


@contextmanager
def styled(**overrides: Any) -> Iterator[None]:
    """Apply the style for one figure only, restoring the caller's settings after.

    For the rare figure that needs a local deviation -- a larger font for a
    poster, no grid for a waveform -- without leaking that deviation into every
    later figure the process draws.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    params = {**RC_PARAMS, **overrides}
    with plt.rc_context(params):  # type: ignore[arg-type]
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=list(OKABE_ITO))
        yield


def figure(size: str | tuple[float, float] = "double", **kwargs: Any) -> Any:
    """``plt.subplots`` with a named page size. Returns ``(fig, ax)``."""
    apply_style()
    import matplotlib.pyplot as plt

    figsize = FIGSIZE[size] if isinstance(size, str) else size
    return plt.subplots(figsize=figsize, **kwargs)


def class_color(index: int) -> str:
    """A palette colour by position, wrapping if there are more classes than colours."""
    return OKABE_ITO[index % len(OKABE_ITO)]


def annotate_source(fig: Any, text: str) -> None:
    """Stamp the data source along the bottom of a figure.

    Rule 1 in visual form: a figure that cannot be traced back to the record or
    CSV it was drawn from is an untraceable number with axes on it. Every figure
    this project emits carries its source.

    Two mechanics matter, both learned by getting them wrong in Phase 28. The
    layout is squeezed upward first, because a stamp placed at the bottom of an
    unreserved figure lands on top of the x-axis label. And the caller passes the
    text in short lines separated by newlines, because ``savefig`` with
    ``bbox_inches="tight"`` expands the *canvas* to fit a wide text box -- a
    single 180-character stamp silently turned a 7.2-inch figure into a 10-inch
    one, changing the aspect ratio of the plot it was supposed to annotate.
    """
    reserved = 0.045 * len(text.splitlines())
    try:
        fig.tight_layout(rect=(0.0, reserved, 1.0, 1.0))
    except (ValueError, AttributeError):  # some layouts (colorbars) refuse; not fatal
        fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, reserved + 0.08))

    fig.text(
        0.005,
        0.004,
        text,
        ha="left",
        va="bottom",
        fontsize=6,
        color="#555555",
        linespacing=1.4,
    )


def save_figure(
    fig: Any, path: str | Path, *, source: str | None = None, close: bool = True
) -> Path:
    """Save at 300 dpi, atomically, with an optional source stamp."""
    from src.utils.io import save_png

    if source:
        annotate_source(fig, source)
    return save_png(fig, path, dpi=DPI, close=close)
