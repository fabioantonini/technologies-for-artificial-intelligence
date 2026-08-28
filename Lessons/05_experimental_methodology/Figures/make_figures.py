"""Generate the conceptual figures for lesson 5.

    python Figures/make_figures.py

Data figures come from the notebooks. These two are diagrams of a *procedure*
rather than plots of numbers: how k-fold rotates the test block, and how nested
cross-validation puts a second loop around the first. Both are far easier to
understand as a picture than as a paragraph, and both are the kind of thing
that otherwise gets drawn badly on a whiteboard.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent

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


def kfold_diagram(k=5):
    """The test block moving along, one row per fold."""
    fig, ax = plt.subplots(figsize=(8.6, 3.8))

    for fold in range(k):
        y = k - fold - 1
        for block in range(k):
            is_test = block == fold
            ax.add_patch(Rectangle(
                (block, y), 0.94, 0.78,
                facecolor=GOLD if is_test else "#DCE6F0",
                edgecolor=SLATE, linewidth=1.1))
            ax.text(block + 0.47, y + 0.39, "test" if is_test else "train",
                    ha="center", va="center", fontsize=9.5,
                    color=INK if is_test else SLATE,
                    fontweight="bold" if is_test else "normal")
        ax.text(-0.25, y + 0.39, f"fold {fold + 1}", ha="right", va="center",
                fontsize=10, color=SLATE)

    ax.text(k / 2, k + 0.15, "the dataset, cut into 5 equal parts",
            ha="center", fontsize=11, color=INK)
    ax.annotate("", xy=(k - 0.06, k - 0.12), xytext=(0, k - 0.12),
                arrowprops=dict(arrowstyle="<->", color=SLATE, lw=1.2))

    ax.set_xlim(-1.7, k + 0.1)
    ax.set_ylim(-0.35, k + 0.45)
    ax.axis("off")
    fig.tight_layout()
    save(fig, "kfold_diagram.png")


def nested_cv_diagram():
    """Why there have to be two loops, and which one you may look at."""
    fig, ax = plt.subplots(figsize=(9.2, 4.2))

    # Outer loop: one row, a held-out block on the right.
    ax.add_patch(Rectangle((0, 3.0), 6.4, 0.8, facecolor="#DCE6F0",
                           edgecolor=SLATE, linewidth=1.2))
    ax.text(3.2, 3.4, "outer training part  (4/5)", ha="center", va="center",
            fontsize=10.5, color=SLATE)
    ax.add_patch(Rectangle((6.5, 3.0), 1.6, 0.8, facecolor=RUST, alpha=.28,
                           edgecolor=SLATE, linewidth=1.2))
    ax.text(7.3, 3.4, "outer test", ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=INK)
    ax.text(-0.2, 3.4, "outer", ha="right", va="center", fontsize=11,
            color=INK, fontweight="bold")

    # Inner loop: the outer training part, split again, several times.
    for row, fold in enumerate(range(3)):
        y = 1.75 - row * 0.62
        for block in range(4):
            is_test = block == fold
            ax.add_patch(Rectangle((block * 1.6, y), 1.5, 0.5,
                                   facecolor=GOLD if is_test else "#EDF2F8",
                                   edgecolor=SLATE, linewidth=1.0))
            ax.text(block * 1.6 + 0.75, y + 0.25,
                    "valid" if is_test else "train", ha="center", va="center",
                    fontsize=9, color=INK if is_test else SLATE)
    ax.text(-0.2, 1.35, "inner", ha="right", va="center", fontsize=11,
            color=INK, fontweight="bold")

    ax.annotate("", xy=(3.2, 2.55), xytext=(3.2, 2.95),
                arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.4))
    ax.text(3.45, 2.72, "the whole search runs in here", fontsize=9.5,
            color=SLATE, va="center")

    ax.text(0, 0.02, "The inner loop may overfit itself as much as it likes.",
            fontsize=10.5, color=INK)
    ax.text(0, -0.38,
            "The outer test block was never part of it: that is the number you report.",
            fontsize=10.5, color=INK, fontweight="bold")

    ax.set_xlim(-1.5, 8.4)
    ax.set_ylim(-0.7, 4.1)
    ax.axis("off")
    fig.tight_layout()
    save(fig, "nested_cv_diagram.png")


if __name__ == "__main__":
    print("figure concettuali della lezione 5:")
    kfold_diagram()
    nested_cv_diagram()
