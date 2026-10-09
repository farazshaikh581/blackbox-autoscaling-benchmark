"""Shared matplotlib style for every plot the repo produces.

One place for the Agg backend, the palette, the method identities (color and
marker per algorithm), the figure sizes and the save conventions, so every
figure (calibration fits, measurements, single runs, the cross-method result
figures) looks like it belongs to the same
repo and paper.

    from viz import new_fig, style_axes, savefig, COLOR, METHOD

Palette: an 8-slot categorical set, assigned in fixed order (never cycled).
Adjacent pairs clear the colorblind (CVD) separation target (worst 9.1, OKLab
x100, Machado 2009) and the normal-vision floor (worst 19.6). With more than
three series in one scatter, colors alone are not enough, so every method also
has its own marker, and multi-method scatters are faceted instead of overlaid.
Slots 3 to 5 sit below 3:1 contrast on white, so plots that use them always
carry markers and a legend or direct labels.

Sizes follow the IEEE two-column layout: SINGLE (3.5 in) and DOUBLE (7.16 in).
`savefig` writes a 300 dpi PNG for the docs and a vector PDF next to it for
the paper.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
           "#4a3aa7", "#e34948"]

# Ink and surface roles. Text never takes a series color.
INK = "#1d1d1b"
INK_2 = "#52514e"
INK_3 = "#8a8984"
GRID = "#e6e5e1"
NEUTRAL = "#b4b3ad"      # context marks (e.g. "all fronts" backdrop)
NEUTRAL_LIGHT = "#dcdbd6"
BAND = "#f0efec"          # shaded windows (disruption, admissible range)
STATUS_CRITICAL = "#d03b3b"  # reserved for failures, never a series color

SINGLE = 3.5
DOUBLE = 7.16

# One identity per method, fixed across every figure in the repo.
METHOD = {
    "nsga2":   dict(label="NSGA-II", color=PALETTE[0], marker="o"),
    "moead":   dict(label="MOEA/D",  color=PALETTE[1], marker="s"),
    "morl_bf": dict(label="MORL-BF", color=PALETTE[2], marker="^"),
    "morl":    dict(label="MORL",    color=PALETTE[3], marker="D"),
    "random":  dict(label="Random",  color=PALETTE[4], marker="v"),
    "hpa":     dict(label="HPA",     color=PALETTE[5], marker="P"),
}

# Roles for calibration plots: measured points vs. fitted model, plus a gray
# for a prior (guessed) default or a reference line.
COLOR = {
    "measured": PALETTE[0],
    "fit": PALETTE[1],
    "prior": INK_3,
    **{k: v["color"] for k, v in METHOD.items()},
}

OBJ_LABEL = {"latency_ms": "latency (ms)", "cost": "cost", "energy_W": "energy (W)"}

_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 6,
    "axes.labelsize": 8,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": INK_3,
    "axes.linewidth": 0.6,
    "axes.facecolor": "white",
    "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_3,
    "ytick.color": INK_3,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "lines.markeredgewidth": 0.6,
    "patch.linewidth": 0,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.3,
    "legend.labelspacing": 0.35,
    "legend.columnspacing": 1.2,
    "figure.facecolor": "white",
    "figure.dpi": 100,
    "figure.titlesize": 9,
    "figure.titleweight": "bold",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
    "mathtext.fontset": "dejavusans",
    "text.color": INK,
}
plt.rcParams.update(_RC)


def new_fig(width=SINGLE, height=None, ncols=1, nrows=1, **kw):
    """`width` is the whole figure's width (SINGLE or DOUBLE column, or inches);
    `height` defaults to a per-row height that suits the width."""
    if height is None:
        per_row = 2.3 if width <= SINGLE + 0.01 else 2.2
        height = per_row * nrows
    kw.setdefault("layout", "constrained")
    return plt.subplots(nrows, ncols, figsize=(width, height), **kw)


def style_axes(ax, grid="y") -> None:
    """Recessive grid on `grid` ('y', 'x', 'both' or None), no top/right spines."""
    if grid:
        ax.grid(True, axis=grid, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def method_style(algo: str) -> dict:
    return METHOD.get(algo, dict(label=algo, color=INK_3, marker="o"))


def label_end(ax, x, y, text, color=INK_2, dx=4, dy=0, **kw):
    """Direct label at the right end of a line, in ink (not the series color)."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                va="center", ha="left", fontsize=7.5, color=color, **kw)


def plain_log_ticks(axis) -> None:
    """Log axis with plain numbers (200, 500, 1,000) instead of 2x10^2."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def note(ax, text, loc="upper left", **kw):
    """Small secondary-ink annotation inside the axes (fit constants, n, rho)."""
    xy = {"upper left, below legend": (0.02, 0.76, "left", "top"),
          "lower right, raised": (0.98, 0.12, "right", "bottom"),"upper left": (0.02, 0.98, "left", "top"),
          "upper right": (0.98, 0.98, "right", "top"),
          "lower right": (0.98, 0.03, "right", "bottom"),
          "lower left": (0.02, 0.03, "left", "bottom")}[loc]
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=xy[2], va=xy[3],
            fontsize=7, color=INK_2, linespacing=1.35, **kw)


def savefig(fig, path: str, pdf: bool = True) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=300)
    if pdf and path.endswith(".png"):
        fig.savefig(path[:-4] + ".pdf")
    plt.close(fig)
    print(f"wrote -> {path}")


def rank_scatter(ax, sim, meas, rho=None) -> None:
    """Rank-vs-rank agreement plot (simulator vs. measurement), with the
    perfect-agreement diagonal and the Spearman rho in a corner note."""
    import numpy as np
    sim, meas = np.asarray(sim, float), np.asarray(meas, float)
    rs, rm = sim.argsort().argsort() + 1, meas.argsort().argsort() + 1
    n = len(sim)
    ax.plot([0.5, n + 0.5], [0.5, n + 0.5], color=INK_3, lw=0.8, ls="--")
    ax.scatter(rs, rm, s=22, color=COLOR["measured"], edgecolor="white",
               linewidth=0.5, zorder=3)
    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(0.5, n + 0.5)
    step = 1 if n <= 10 else 5
    ticks = [1] + list(range(step, n + 1, step)) if step > 1 else list(range(1, n + 1))
    ax.set_xticks(sorted(set(ticks)))
    ax.set_yticks(sorted(set(ticks)))
    if rho is not None:
        note(ax, f"Spearman ρ = {rho:.3f}\nn = {n}", loc="upper left")
    style_axes(ax, grid="both")


def label_ends(ax, items, min_gap=0.075, dx=4) -> None:
    """Direct labels just right of the axes for several line ends, nudged
    apart so they never overlap. `items` is a list of (x, y, text). Call after
    the axis limits and scales are final."""
    if not items:
        return
    to_axes = ax.transAxes.inverted()
    pts = sorted((to_axes.transform(ax.transData.transform((x, y)))[1], text)
                 for x, y, text in items)
    ys = [p[0] for p in pts]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    over = ys[-1] - 1.0
    if over > 0:
        ys = [v - over for v in ys]
    for (_, text), ty in zip(pts, ys):
        ax.annotate(text, xy=(1.0, ty), xycoords="axes fraction", xytext=(dx, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=7.5, color=INK_2, annotation_clip=False)


def plot_hv_convergence(curves: dict, path: str, title="Anytime hypervolume",
                        width=SINGLE, height=2.5) -> None:
    """Mean anytime HV per method with a +-1 std band. `curves` maps an algo key
    to a list of per-run (evals, hv) arrays. Each method is drawn only over the
    evaluations its own runs cover (no extrapolation). Methods whose histories
    are identical (MORL-BF reads out the same training archive as MORL) are
    drawn once, with a joint label."""
    import numpy as np
    curves = dict(curves)
    labels = {a: method_style(a)["label"] for a in curves}
    keys = list(curves)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a in curves and b in curves and len(curves[a]) == len(curves[b]) and all(
                    np.array_equal(na, nb) and np.allclose(ha, hb)
                    for (na, ha), (nb, hb) in zip(curves[a], curves[b])):
                del curves[b]
                labels[a] = f"{labels[a]}, {labels[b]}"
    fig, ax = new_fig(width, height)
    for a, runs in curves.items():
        st = method_style(a)
        lo = max(ns.min() for ns, _ in runs)
        hi = min(ns.max() for ns, _ in runs)
        grid = np.linspace(lo, hi, 60)
        C = np.vstack([np.interp(grid, ns, hv) for ns, hv in runs])
        m, s = C.mean(0), C.std(0)
        ax.fill_between(grid, m - s, m + s, color=st["color"], alpha=0.10, lw=0)
        ax.plot(grid, m, color=st["color"], marker=st["marker"], markevery=[-1],
                markeredgecolor="white", label=labels[a])
    ax.set_xlabel("evaluations")
    ax.set_ylabel("hypervolume (normalized)")
    ax.set_title(title)
    ax.set_xlim(left=0)
    style_axes(ax)
    # The line ends finish within ~0.01 HV of each other, too close for
    # direct labels, so a legend goes in the empty lower-right corner.
    ax.legend(loc="lower right", frameon=False, labelcolor=INK_2)
    savefig(fig, path)


def _log_if_wide(ax, values, axis):
    import numpy as np
    v = np.asarray(values, float)
    v = v[v > 0]
    if len(v) and v.max() / v.min() > 30:
        (ax.set_xscale if axis == "x" else ax.set_yscale)("log")


OBJ_PAIRS = [(0, 1), (0, 2), (1, 2)]
OBJ_NAMES = ["latency (ms)", "cost", "energy (W)"]


def plot_fronts_faceted(fronts: dict, path: str, title=None) -> None:
    """Small multiples: one row per method, one column per objective pair.
    Each panel shows that method's front points over the union of all
    methods' fronts in light gray, so shape and coverage compare at a glance
    without overlaying more than one colored series."""
    import numpy as np
    algos = list(fronts)
    union = np.vstack([fronts[a] for a in algos])
    n = len(algos)
    fig, axes = new_fig(DOUBLE, 1.45 * n + 0.3, ncols=3, nrows=n, squeeze=False)
    for r, a in enumerate(algos):
        st = method_style(a)
        F = fronts[a]
        for c, (i, j) in enumerate(OBJ_PAIRS):
            ax = axes[r, c]
            ax.scatter(union[:, i], union[:, j], s=5, color=NEUTRAL_LIGHT, lw=0,
                       zorder=1)
            ax.scatter(F[:, i], F[:, j], s=11, color=st["color"], marker=st["marker"],
                       edgecolor="white", linewidth=0.3, alpha=0.9, zorder=2)
            _log_if_wide(ax, union[:, i], "x")
            _log_if_wide(ax, union[:, j], "y")
            style_axes(ax, grid="both")
            if r == n - 1:
                ax.set_xlabel(OBJ_NAMES[i])
            else:
                ax.tick_params(labelbottom=False)
            ax.set_ylabel(OBJ_NAMES[j], fontsize=7)
        axes[r, 0].set_title(st["label"], loc="left", color=INK)
    for c in range(3):
        col = axes[:, c]
        for ax in col[1:]:
            ax.sharex(col[0])
            ax.sharey(col[0])
    if title:
        fig.suptitle(title, x=0.01, ha="left")
    savefig(fig, path)


def plot_single_front(F, algo: str, path: str, label=None) -> None:
    """One method's final front in the three objective-pair projections."""
    import numpy as np
    F = np.atleast_2d(F)
    st = method_style(algo)
    fig, axes = new_fig(DOUBLE, 2.2, ncols=3)
    for ax, (i, j) in zip(axes, OBJ_PAIRS):
        ax.scatter(F[:, i], F[:, j], s=18, color=st["color"], marker=st["marker"],
                   edgecolor="white", linewidth=0.4)
        _log_if_wide(ax, F[:, i], "x")
        _log_if_wide(ax, F[:, j], "y")
        ax.set_xlabel(OBJ_NAMES[i])
        ax.set_ylabel(OBJ_NAMES[j])
        style_axes(ax, grid="both")
    axes[0].set_title(f"{label or st['label']}: final front, {len(F)} points")
    savefig(fig, path)
