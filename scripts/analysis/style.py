"""Shared figure style for every paper figure.

Import this module (``from . import style``) before drawing and call
``style.apply()`` once per figure. It (1) registers the vendored TeX Gyre Heros
TrueType fonts under ``scripts/analysis/fonts/`` so that every machine renders
the same typeface and embeds it as a plain TrueType font in the PDF, (2) sets
matplotlib's rcParams to one consistent look (sizes, line widths, spines), and
(3) exposes the small set of layout constants the figure scripts share.

Figures are drawn at the physical size they are printed at (``TEXT_WIDTH_IN``
is the NeurIPS text width) with ``\\includegraphics[width=\\linewidth]``, so the
point sizes below are the point sizes the reader sees. Colours live in
``spec.py`` (section 8) so that the one palette is defined in one place.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

from .spec import (
    LIGHT_ALPHA,
    NEUTRAL_LINE,
    PALETTE,
    Q1_COLOR,
    Q2_COLOR,
    SIG_ALPHA,
)

# ---------------------------------------------------------------------------
# Physical sizes (inches) and type sizes (points, as printed)
# ---------------------------------------------------------------------------
TEXT_WIDTH_IN = 5.5          # NeurIPS \textwidth
FONT_FAMILY = "TeX Gyre Heros TT"   # vendored; falls back to a system sans-serif
FS_BASE = 7.5                # tick labels, legend entries
FS_LABEL = 8.0               # axis labels
FS_TITLE = 8.5               # panel titles
FS_SMALL = 6.5               # in-plot annotations
LW_AXIS = 0.6                # spines and tick marks
LW_CI = 1.2                  # confidence-interval bars
LW_REF = 0.8                 # reference lines
MARKER_SIZE = 9              # scatter ``s`` (points^2) for estimate markers
GRID_COLOR = "#e4e4e1"
BAND_COLOR = "#f1f1ee"       # alternating background band
POOLED_BAND_COLOR = "#e2e2de"
SEPARATOR_COLOR = "#b8b8b4"

_FONT_DIR = Path(__file__).resolve().parent / "fonts"
_fonts_registered = False


def register_fonts() -> None:
    """Register the vendored TrueType fonts with matplotlib (idempotent)."""
    global _fonts_registered
    if _fonts_registered:
        return
    for p in sorted(_FONT_DIR.glob("*.ttf")):
        try:
            font_manager.fontManager.addfont(str(p))
        except Exception:  # pragma: no cover - a bad font file should not stop a run
            pass
    _fonts_registered = True


def apply() -> None:
    """Set the shared rcParams. Call once per figure (cheap; idempotent)."""
    register_fonts()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": FS_BASE,
        "axes.titlesize": FS_TITLE,
        "axes.titleweight": "bold",
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_BASE,
        "ytick.labelsize": FS_BASE,
        "legend.fontsize": FS_BASE,
        "legend.title_fontsize": FS_BASE,
        "legend.frameon": False,
        "axes.linewidth": LW_AXIS,
        "axes.edgecolor": "#444444",
        "axes.labelcolor": PALETTE["ink"],
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "xtick.labelcolor": PALETTE["ink"],
        "ytick.labelcolor": PALETTE["ink"],
        "xtick.major.width": LW_AXIS,
        "ytick.major.width": LW_AXIS,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.5,
        "lines.linewidth": LW_CI,
        "text.color": PALETTE["ink"],
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,      # embed as TrueType, not Type 3
        "ps.fonttype": 42,
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT_FAMILY,
        "mathtext.it": f"{FONT_FAMILY}:italic",
        "mathtext.bf": f"{FONT_FAMILY}:bold",
        "mathtext.default": "regular",
    })


def tidy_axes(ax, *, grid_axis: str | None = "x") -> None:
    """Recessive grid, clean spines."""
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID_COLOR, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def save(fig, out_path: Path, *, png: bool = True, extra=None) -> None:
    """Write ``out_path`` as PDF (and PNG). ``out_path`` may carry any suffix."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # NOTE: no ``bbox_extra_artists`` — passing it makes matplotlib ignore the
    # default artist list, which drops free-floating axis text (panel titles).
    # Figure legends are figure artists and are included automatically.
    kw = {"bbox_inches": "tight", "pad_inches": 0.02}
    fig.savefig(out_path.with_suffix(".pdf"), **kw)
    if png:
        fig.savefig(out_path.with_suffix(".png"), dpi=300, **kw)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Estimate markers and the shared legend for the forest-style figures
# ---------------------------------------------------------------------------


def plot_estimate(ax, ypos: float, beta: float, ci_low: float, ci_high: float,
                  *, color: str, significant: bool, faded: bool = False,
                  zorder: int = 3) -> None:
    """One point estimate with its CI: filled marker if significant, open otherwise.

    A faded (below-quota) estimate is drawn in a lighter, *opaque* tint rather
    than with alpha, so that the white fill of an open marker still hides the
    CI line behind it.
    """
    c = fade(color) if faded else color
    ax.hlines(ypos, ci_low, ci_high, color=c, lw=LW_CI, zorder=zorder - 1, capstyle="butt")
    if significant:
        ax.scatter([beta], [ypos], s=MARKER_SIZE, marker="o", facecolor=c,
                   edgecolor=c, linewidth=0.6, zorder=zorder)
    else:
        ax.scatter([beta], [ypos], s=MARKER_SIZE, marker="o", facecolor="white",
                   edgecolor=c, linewidth=0.8, zorder=zorder)


def fade(color: str, amount: float = 1 - LIGHT_ALPHA) -> tuple:
    """Blend ``color`` towards white (opaque equivalent of alpha = LIGHT_ALPHA)."""
    from matplotlib.colors import to_rgb
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def effect_ticks(ax, *, label_every: float | None = None, mark_every: float = 0.25) -> None:
    """Tick marks every 0.25 on an effect axis, with a grid at each.

    Labels go every 0.5, or every 0.25 when the axis spans 0.7 or less (so a
    narrow panel still gets more than one label). Call after setting xlim.
    """
    from matplotlib.ticker import MultipleLocator
    if label_every is None:
        lo, hi = ax.get_xlim()
        label_every = 0.25 if (hi - lo) <= 0.7 else 0.5
    ax.xaxis.set_major_locator(MultipleLocator(label_every))
    ax.xaxis.set_minor_locator(MultipleLocator(mark_every))
    ax.tick_params(axis="x", which="minor", length=1.8, width=LW_AXIS)
    ax.grid(axis="x", which="major", color=GRID_COLOR, lw=0.5, zorder=0)
    ax.grid(axis="x", which="minor", color=GRID_COLOR, lw=0.35, zorder=0)


def forest_legend_handles(*, q1: bool = True, q2: bool = True,
                          faded: bool = True, pooled_line: bool = True) -> list[Line2D]:
    """Legend entries shared by the forest-style figures (Figs 2, 4, 5, 11-13)."""
    ms = 3.8
    handles: list[Line2D] = []
    if q1:
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=Q1_COLOR,
                              markeredgecolor=Q1_COLOR, markersize=ms, label="Q1 (post-filter)"))
    if q2:
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=Q2_COLOR,
                              markeredgecolor=Q2_COLOR, markersize=ms, label="Q2 (full sample)"))
    handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="#6b6b6b",
                          markeredgecolor="#6b6b6b", markersize=ms,
                          label=f"Bonferroni-corrected p < {SIG_ALPHA:g}"))
    handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                          markeredgecolor="#6b6b6b", markeredgewidth=0.8, markersize=ms,
                          label=f"p ≥ {SIG_ALPHA:g}"))
    if faded:
        handles.append(Line2D([0], [0], color=fade("#6b6b6b"), lw=LW_CI,
                              label="Below sample-size quota"))
    if pooled_line:
        handles.append(Line2D([0], [0], color="#6b6b6b", lw=LW_REF, ls=(0, (1, 1.5)),
                              label="Pooled estimate"))
    return handles


def panel_title(ax, display: str, *, pad_pt: float = 3.0) -> None:
    """Two-line panel title: the environment code in bold, the name beneath.

    ``display`` is an ENVIRONMENT_DISPLAY value such as "E3. Methodology Review".
    Splitting it keeps long names inside narrow panels.
    """
    if ". " in display:
        code, name = display.split(". ", 1)
    else:
        code, name = display, ""
    from matplotlib import transforms
    fig = ax.figure
    t1 = transforms.offset_copy(ax.transAxes, fig=fig, x=0, y=pad_pt, units="points")
    t2 = transforms.offset_copy(ax.transAxes, fig=fig, x=0,
                                y=pad_pt + FS_BASE * 1.35, units="points")
    ax.text(0.5, 1.0, name, transform=t1, ha="center", va="bottom",
            fontsize=FS_BASE, clip_on=False)
    ax.text(0.5, 1.0, code, transform=t2, ha="center", va="bottom",
            fontsize=FS_TITLE, fontweight="bold", clip_on=False)


def zero_line(ax) -> None:
    ax.axvline(0, color=NEUTRAL_LINE, lw=LW_REF, ls="-", alpha=0.8, zorder=1)


def reference_line(ax, x: float, color: str) -> None:
    """Dotted vertical line at a pooled estimate."""
    ax.axvline(x, color=color, lw=LW_REF, ls=(0, (1, 1.5)), alpha=0.9, zorder=1)
