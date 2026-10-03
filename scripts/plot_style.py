"""Shared matplotlib style for all paper figures: one palette, one set of
rcParams, used by every figure-generation script in scripts/ and
real_pusht/ so the whole paper looks like it came from one hand instead of
a dozen quick-and-dirty debug plots.

Built on top of SciencePlots (github.com/garrettj403/SciencePlots)'s
"science" style, which gives the classic journal look: inward ticks on
all four sides, minor ticks, a thin frame, and true LaTeX (Computer
Modern) text via matplotlib's usetex renderer. Falls back to a
LaTeX-free approximation if SciencePlots or a working LaTeX install is
not available, so figure generation never hard-fails on a missing
dependency.

Import this before creating any figure:
    from plot_style import COLORS, setup_style
    setup_style()
"""
import matplotlib


COLORS = {
    "gt": "#222222",              # ground truth: near-black
    "baseline_k1": "#D64550",     # memoryless baseline: warm red (the "fails" arm)
    "baseline_k3": "#E8A33D",     # history-window baseline: amber
    "baseline_rnn": "#8456B0",    # recurrent aggregator baseline: purple
    "tijepa": "#1F6FB2",          # TI-JEPA: deep blue (the "hero" arm)
    "oracle": "#3E9B5C",          # oracle / positive reference: green
    "neutral": "#8A8A8A",         # grey for de-emphasized reference lines
}

LABELS = {
    "gt": "ground truth",
    "baseline_k1": r"baseline$_{k=1}$ (memoryless)",
    "baseline_k3": r"baseline$_{k=3}$ (history)",
    "baseline_rnn": "recurrent baseline",
    "tijepa": "TI-JEPA",
    "oracle": "oracle",
}


def _try_scienceplots(use_latex: bool) -> bool:
    """Apply SciencePlots' base style. Returns True on success."""
    import matplotlib.pyplot as plt
    try:
        import scienceplots  # noqa: F401
    except ImportError:
        return False
    try:
        if use_latex:
            plt.style.use(["science"])
        else:
            plt.style.use(["science", "no-latex"])
        return True
    except Exception:
        return False


def setup_style(use_latex: bool = True):
    """Set up the shared paper style.

    use_latex=True (default) renders text with real LaTeX (Computer
    Modern), matching the look of the SciencePlots gallery. This requires
    a working `latex` binary plus the `cm-super`/`type1cm` packages; if
    that combination is not available, we transparently fall back to
    SciencePlots' no-latex mode, then to a plain approximation, so every
    figure script keeps working on any machine.
    """
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = _try_scienceplots(use_latex=use_latex)
    if not ok and use_latex:
        ok = _try_scienceplots(use_latex=False)

    if not ok:
        # Minimal hand-rolled approximation of the science style so figures
        # still look reasonable even with no SciencePlots installed.
        plt.rcParams.update({
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
        })

    # Our own overrides, applied on top of whichever base style loaded
    # above: consistent colors, sizes, and a touch more breathing room
    # than SciencePlots' tight defaults (which target 3.5in single-column
    # IEEE figures, not our wider multi-panel ones).
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.5,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "savefig.dpi": 220,
        "figure.dpi": 140,
        "lines.linewidth": 2.0,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })
