"""Generate the conceptual figures for lesson 5.

    python Figures/make_figures.py

Data figures come from the notebooks. These three are diagrams of a *procedure*
rather than plots of numbers: the two recipes for choosing a hyperparameter, how
k-fold rotates the validation block, and how nested cross-validation puts a second
loop around the first. Both are far easier to
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


def two_recipes():
    """How a hyperparameter is chosen: one fit per candidate when it sits inside
    the cost (lambda), one fit in all when it acts on the output (the threshold)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6))
    recipes = [
        ("Fixed before the fit: the penalty (C), tree depth, neighbours",
         ["1. Set the test set aside",
          "2. For each candidate, C = 0.01, 1, 100:\n5-fold cross-validation on the rest\n= 5 fits each, 3 × 5 = 15 fits",
          "3. Keep the C with the best mean score",
          "4. Refit with that C on all non-test data\n(fit number 16)",
          "5. Score the test set, once"],
         BLUE),
        ("Chosen after the fit: the decision threshold",
         ["1. Set the test set aside,\nthen cut a validation set from the rest",
          "2. Fit the model on the training part:\none fit in all",
          "3. Predict probabilities on the validation set",
          "4. Try every threshold on those probabilities;\nkeep the cheapest",
          "5. Score the test set, once"],
         TEAL),
    ]
    for ax, (title, steps, colour) in zip(axes, recipes):
        heights = [1.25 if "\n" in s and s.count("\n") == 2 else
                   0.95 if "\n" in s else 0.7 for s in steps]
        y = 0.0
        for index, (step, h) in enumerate(zip(steps, heights)):
            last = index == len(steps) - 1
            ax.add_patch(Rectangle((0, y - h), 10, h,
                                   facecolor=RUST if last else "#DCE6F0",
                                   alpha=.25 if last else 1.0,
                                   edgecolor=colour, linewidth=1.4))
            ax.text(0.35, y - h / 2, step, ha="left", va="center", fontsize=10.5,
                    color=INK, fontweight="bold" if last else "normal")
            if not last:
                ax.annotate("", xy=(5, y - h - 0.32), xytext=(5, y - h - 0.02),
                            arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.3))
            y -= h + 0.34
        ax.set_title(title, fontsize=11.5, color=colour, fontweight="bold",
                     loc="left")
        ax.set_xlim(-0.1, 10.1)
        ax.set_ylim(y + 0.1, 0.15)
        ax.axis("off")
    fig.tight_layout(w_pad=2.5)
    save(fig, "two_recipes.png")


def kfold_diagram(k=5):
    """The validation fold moving along: one row per round, one column per fold,
    one score per round, and the estimate as their mean."""
    fig, ax = plt.subplots(figsize=(9.6, 4.2))

    for fold in range(k):
        ax.text(fold + 0.47, k + 0.05, f"fold {fold + 1}", ha="center",
                va="bottom", fontsize=10, color=SLATE)

    for round_ in range(k):
        y = k - round_ - 1
        for block in range(k):
            is_valid = block == round_
            ax.add_patch(Rectangle(
                (block, y), 0.94, 0.78,
                facecolor=GOLD if is_valid else "#DCE6F0",
                edgecolor=SLATE, linewidth=1.1))
            ax.text(block + 0.47, y + 0.39, "validation" if is_valid else "train",
                    ha="center", va="center", fontsize=9.5,
                    color=INK if is_valid else SLATE,
                    fontweight="bold" if is_valid else "normal")
        ax.text(-0.25, y + 0.39, f"round {round_ + 1}", ha="right", va="center",
                fontsize=10, color=SLATE)
        ax.annotate("", xy=(k + 0.35, y + 0.39), xytext=(k + 0.02, y + 0.39),
                    arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.1))
        ax.text(k + 0.42, y + 0.39, f"score {round_ + 1}", ha="left",
                va="center", fontsize=10, color=INK)

    ax.text(k / 2, k + 0.55, "the dataset, cut into 5 equal parts (folds)",
            ha="center", fontsize=11, color=INK)
    ax.text(k + 0.42, -0.3, "estimate =\n(score 1 + ... + score 5) ÷ 5",
            ha="left", va="top", fontsize=10, color=INK, fontweight="bold")

    ax.set_xlim(-1.7, k + 1.75)
    ax.set_ylim(-1.05, k + 0.85)
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
    print("conceptual figures for lesson 5:")
    two_recipes()
    kfold_diagram()
    nested_cv_diagram()
