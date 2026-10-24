"""Generate the conceptual figures for lesson 4.

    python Figures/make_figures.py

Data figures come from the notebooks. These are the diagrams that carry an idea
and have no data behind them: the shape of the sigmoid and what it does to the
log-odds, and the picture of which cells of the confusion matrix each metric
divides by, which is the one thing students most reliably get backwards.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent

# Course palette, matching Course/template.pptx and lessons 1-3
BLUE = "#1F4E79"
TEAL = "#2E7D8A"
RUST = "#9C4221"
GOLD = "#F0BE50"
INK = "#1A1A1A"
SLATE = "#4A5568"

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": SLATE,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {name}")


def sigmoid_and_odds():
    """The curve, and the straight line hiding inside it."""
    z = np.linspace(-8, 8, 500)
    p = 1 / (1 + np.exp(-z))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    axes[0].plot(z, p, lw=2.8, color=BLUE)
    axes[0].axhline(0.5, color=SLATE, ls=":", lw=1)
    axes[0].axvline(0, color=SLATE, ls=":", lw=1)
    axes[0].scatter([0], [0.5], s=70, color=GOLD, edgecolor=INK, zorder=5)
    axes[0].annotate("steepest here:\nslope 1/4", xy=(0, 0.5),
                     xytext=(1.6, 0.24), fontsize=10, color=INK,
                     arrowprops=dict(arrowstyle="->", color=INK))
    axes[0].annotate("flat: more evidence\nbarely moves it", xy=(5.6, 0.996),
                     xytext=(-1.2, 0.86), fontsize=10, color=SLATE,
                     arrowprops=dict(arrowstyle="->", color=SLATE))
    axes[0].set_xlabel("log-odds  z = wᵀx + b")
    axes[0].set_ylabel("probability  σ(z)")
    axes[0].set_title("The sigmoid squashes ℝ into (0, 1)", fontsize=11)
    axes[0].set_ylim(-0.04, 1.09)

    # The same relationship the other way round: log-odds are linear.
    grid = np.linspace(0.002, 0.998, 500)
    axes[1].plot(grid, np.log(grid / (1 - grid)), lw=2.8, color=TEAL)
    axes[1].axhline(0, color=SLATE, ls=":", lw=1)
    axes[1].set_xlabel("probability  p")
    axes[1].set_ylabel("log-odds  log(p / (1 − p))")
    axes[1].set_title("...because the log-odds run over all of ℝ", fontsize=11)
    axes[1].set_ylim(-6.5, 6.5)

    fig.tight_layout()
    save(fig, "sigmoid_and_odds.png")


def metric_denominators():
    """Which cells each metric divides by. The reliable source of confusion."""
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))

    cells = {  # (column, row) with row 0 at the top
        "TN": (0, 0), "FP": (1, 0),
        "FN": (0, 1), "TP": (1, 1),
    }
    counts = {"TN": "1911", "FP": "13", "FN": "33", "TP": "43"}

    for ax, title, highlight, colour in (
            (axes[0], "Precision = TP / (TP + FP)\n“of the alarms we raised”",
             ("TP", "FP"), TEAL),
            (axes[1], "Recall = TP / (TP + FN)\n“of the drives that failed”",
             ("TP", "FN"), RUST)):
        for name, (column, row) in cells.items():
            filled = name in highlight
            ax.add_patch(Rectangle((column, 1 - row), 1, 1,
                                   facecolor=colour if filled else "white",
                                   alpha=.30 if filled else 1,
                                   edgecolor=SLATE, linewidth=1.4))
            weight = "bold" if filled else "normal"
            ax.text(column + .5, 1 - row + .62, name, ha="center",
                    fontsize=13, fontweight=weight,
                    color=INK if filled else SLATE)
            ax.text(column + .5, 1 - row + .28, counts[name], ha="center",
                    fontsize=12, color=INK if filled else SLATE)

        ax.text(-0.09, 1.5, "actually\nhealthy", ha="right", va="center",
                fontsize=10, color=SLATE)
        ax.text(-0.09, 0.5, "actually\nfailed", ha="right", va="center",
                fontsize=10, color=SLATE)
        ax.text(0.5, 2.12, "predicted\nhealthy", ha="center", fontsize=10,
                color=SLATE)
        ax.text(1.5, 2.12, "predicted\nfailing", ha="center", fontsize=10,
                color=SLATE)

        ax.set_title(title, fontsize=11, pad=30)
        ax.set_xlim(-0.85, 2.15)
        ax.set_ylim(-0.15, 2.5)
        ax.axis("off")

    fig.tight_layout()
    save(fig, "metric_denominators.png")


def evidence_on_the_sigmoid():
    """The same evidence (odds x 6) on three drives: a fixed step in log-odds,
    a large move in probability in the middle and a small one at the ends."""
    z = np.linspace(-8, 7, 500)
    sig = lambda v: 1 / (1 + np.exp(-v))
    step = np.log(6)

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.plot(z, sig(z), lw=2.8, color=BLUE, zorder=2)

    # Starting points: an average drive (intercept -6.20), 50%, 95%.
    for z0, start, label_xy in ((-6.20, "0.2%", (-6.9, 0.17)),
                                (0.0, "50%", (-2.9, 0.62)),
                                (np.log(19), "95%", (3.4, 0.80))):
        z1 = z0 + step
        p0, p1 = sig(z0), sig(z1)
        ax.annotate("", xy=(z1, p0), xytext=(z0, p0),
                    arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2.4,
                                    mutation_scale=16), zorder=3)
        # A rise too small for an arrowhead is drawn as a plain segment.
        if p1 - p0 > 0.1:
            ax.annotate("", xy=(z1, p1), xytext=(z1, p0),
                        arrowprops=dict(arrowstyle="-|>", color=RUST, lw=2.4,
                                        mutation_scale=16), zorder=3)
        else:
            ax.plot([z1, z1], [p0, p1], color=RUST, lw=2.4, zorder=3)
        ax.scatter([z0], [p0], s=60, color=GOLD, edgecolor=INK, zorder=4)
        ax.scatter([z1], [p1], s=34, color=RUST, zorder=4)
        points = round(100 * (p1 - p0))
        noun = "point" if points == 1 else "points"
        ax.text(*label_xy, f"drive at {start}:\n+{points} {noun}",
                fontsize=11, color=INK, ha="left", va="center")

    ax.plot([], [], color=TEAL, lw=2.4,
            label="the same reading on every drive: odds × 6")
    ax.plot([], [], color=RUST, lw=2.4, label="what it does to the risk")
    ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    ax.set_xlabel("log-odds  log(p ÷ (1 − p))")
    ax.set_ylabel("probability of failing  p")
    ax.set_ylim(-0.04, 1.06)
    fig.tight_layout()
    save(fig, "evidence_on_the_sigmoid.png")


if __name__ == "__main__":
    print("figure concettuali della lezione 4:")
    sigmoid_and_odds()
    metric_denominators()
    evidence_on_the_sigmoid()
