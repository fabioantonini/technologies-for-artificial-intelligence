"""Generate the conceptual figures for lesson 5.

    python Figures/make_figures.py

Data figures come from the notebooks. These six are diagrams of a *procedure*
rather than plots of numbers: the two recipes for choosing a hyperparameter, how
k-fold rotates the validation block, how GroupKFold keeps a drive in one fold,
where the feature selector must sit, how GridSearchCV fills its grid, and how
nested cross-validation puts a second loop around the first. Each is far easier
to understand as a picture than as a paragraph, and each is the kind of thing
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
    """How a hyperparameter is chosen on a validation set: one fit per candidate
    when it sits inside the cost (lambda), one fit in all when it acts on the
    output (the threshold). Sizes are slide 54's split of the 800 drives."""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6))
    split = ("1. Set 200 test drives aside; cut 150 validation\n"
             "drives from the other 600: 450 left to train")
    recipes = [
        ("Fixed before the fit: the penalty (C), tree depth, neighbours",
         [split,
          "2. For each candidate, C = 0.01, 1, 100:\nfit on the 450, score on the 150\n= one fit each, 3 fits",
          "3. Keep the C with the best validation score:\nthe winner is a value of C, not a model",
          "4. Refit with that C on all 600 non-test drives\n(fit number 4)",
          "5. Score the 200 test drives, once"],
         BLUE),
        ("Chosen after the fit: the decision threshold",
         [split,
          "2. Fit the model on the 450:\none fit in all",
          "3. Predict probabilities on the 150 validation drives",
          "4. Try every threshold on those probabilities;\nkeep the cheapest",
          "5. Score the 200 test drives, once"],
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
    """Why there have to be two loops, and which one you may look at.

    Drawn to match notebook 3: 5 outer folds, and inside each outer training
    part a 5-fold search over the 25 candidates.
    """
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    width = 8.0

    # Outer loop: round 1 of 5, the last fifth held out.
    ax.text(width / 2, 4.75, "Outer round 1 of 5: one fifth of the 800 drives held out",
            ha="center", va="center", fontsize=11.5, color=INK, fontweight="bold")
    for block in range(5):
        held = block == 4
        ax.add_patch(Rectangle((block * 1.6, 3.85), 1.5, 0.6,
                               facecolor=RUST if held else "#DCE6F0",
                               alpha=.35 if held else 1.0,
                               edgecolor=SLATE, linewidth=1.2))
        ax.text(block * 1.6 + 0.75, 4.15,
                "outer test\n160 drives" if held else "train\n160 drives",
                ha="center", va="center", fontsize=10,
                fontweight="bold" if held else "normal",
                color=INK if held else SLATE)
    ax.text(-0.2, 4.15, "outer", ha="right", va="center", fontsize=11,
            color=INK, fontweight="bold")

    ax.annotate("", xy=(3.15, 3.35), xytext=(3.15, 3.8),
                arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.4))
    ax.text(3.4, 3.57, "these 640 drives only, re-split into 5 new inner folds of 128",
            fontsize=10, color=SLATE, va="center")

    # Inner loop: the outer training part, split into 5 folds.
    inner_width = 6.3 / 5
    for row in range(5):
        y = 2.85 - row * 0.5
        for block in range(5):
            valid = block == row
            ax.add_patch(Rectangle((block * inner_width, y), inner_width - 0.08, 0.4,
                                   facecolor=GOLD if valid else "#EDF2F8",
                                   edgecolor=SLATE, linewidth=1.0))
            ax.text(block * inner_width + (inner_width - 0.08) / 2, y + 0.2,
                    "valid" if valid else "train", ha="center", va="center",
                    fontsize=8.5, color=INK if valid else SLATE)
    ax.text(-0.2, 2.05, "inner", ha="right", va="center", fontsize=11,
            color=INK, fontweight="bold")
    ax.text(6.45, 2.05, "for each of the\n25 candidates:\ntrain on 512,\nscore on 128;\n"
            "best mean wins,\nrefitted on the 640",
            ha="left", va="center", fontsize=9.5, color=SLATE)

    ax.text(0, 0.05, "The winner is scored once on the 160 outer test drives: 0.698.",
            fontsize=10.5, color=INK)
    ax.text(0, -0.35, "5 outer rounds: (0.698 + 0.852 + 0.754 + 0.487 + 0.558) ÷ 5 = 0.670",
            fontsize=10.5, color=INK, fontweight="bold")
    ax.text(0, -0.75, "It measures what choosing costs; it does not choose.",
            fontsize=10.5, color=INK)

    ax.set_xlim(-1.5, 8.6)
    ax.set_ylim(-1.0, 5.0)
    ax.axis("off")
    fig.tight_layout()
    save(fig, "nested_cv_diagram.png")


def step_boxes(ax, steps, colour, y=0.0, result=None):
    """A column of numbered boxes joined by arrows, the last one optionally a
    result in bold; returns the y where the column ends."""
    for index, step in enumerate(steps):
        h = 0.62 + 0.3 * step.count("\n")
        ax.add_patch(Rectangle((0, y - h), 10, h, facecolor="#DCE6F0",
                               edgecolor=colour, linewidth=1.4))
        ax.text(0.3, y - h / 2, step, ha="left", va="center", fontsize=10.2, color=INK)
        ax.annotate("", xy=(5, y - h - 0.3), xytext=(5, y - h - 0.02),
                    arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.3))
        y -= h + 0.32
    if result:
        h = 0.62 + 0.3 * result.count("\n")
        ax.add_patch(Rectangle((0, y - h), 10, h, facecolor=RUST, alpha=.22,
                               edgecolor=colour, linewidth=1.4))
        ax.text(0.3, y - h / 2, result, ha="left", va="center", fontsize=10.5,
                color=INK, fontweight="bold")
        y -= h
    return y


def fold_strip(ax, y, selector_sees, colour):
    """Round 1 of 5: the 800 drives in five folds of 160, the first one
    validating, and a bracket over the rows the selector looks at."""
    for block in range(5):
        valid = block == 0
        ax.add_patch(Rectangle((block * 2, y), 1.9, 0.55,
                               facecolor=GOLD if valid else "#EDF2F8",
                               edgecolor=SLATE, linewidth=1.0))
        ax.text(block * 2 + 0.95, y + 0.27, "validation\n160" if valid else "train\n160",
                ha="center", va="center", fontsize=8.8,
                color=INK if valid else SLATE, fontweight="bold" if valid else "normal")
    start = 0 if selector_sees == "all" else 2
    ax.plot([start + 0.05, start + 0.05, 9.85, 9.85], [y + 0.75, y + 0.88, y + 0.88, y + 0.75],
            color=colour, lw=2)
    label = ("SelectKBest looks at all 800 drives and their labels" if selector_sees == "all"
             else "SelectKBest looks at the 640 training drives only")
    ax.text((start + 9.85) / 2, y + 0.98, label, ha="center", va="bottom", fontsize=10,
            color=colour, fontweight="bold")
    ax.text(0, y - 0.12, "round 1 of 5 shown; the validation fold moves each round",
            ha="left", va="top", fontsize=8.8, color=SLATE)


def groupkfold_diagram():
    """KFold deals readings at random, so a drive sits on both sides of the
    split; GroupKFold deals whole drives. Drawn small: 10 drives, 4 readings."""
    import numpy as np
    drives = np.repeat(np.arange(1, 11), 4)
    shuffled = np.random.default_rng(4).permutation(drives)
    panels = [("KFold: the readings are dealt at random", shuffled, RUST),
              ("GroupKFold(5), groups=drive_id: whole drives are dealt",
               drives, TEAL)]
    palette = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    for ax, (title, order, colour) in zip(axes, panels):
        for fold in range(5):
            valid = fold == 0
            ax.add_patch(Rectangle((fold * 2.1, 0), 1.95, 4.3,
                                   facecolor=GOLD if valid else "#EDF2F8",
                                   alpha=.55 if valid else 1.0,
                                   edgecolor=SLATE, linewidth=1.1))
            ax.text(fold * 2.1 + 0.975, 4.45, "validation" if valid else "train",
                    ha="center", va="bottom", fontsize=10,
                    fontweight="bold" if valid else "normal",
                    color=INK if valid else SLATE)
            for slot in range(8):
                drive = order[fold * 8 + slot]
                x = fold * 2.1 + 0.18 + (slot % 2) * 0.85
                y = 3.3 - (slot // 2) * 1.0
                seen_twice = (title.startswith("KFold") and drive == order[0])
                ax.add_patch(Rectangle((x, y), 0.75, 0.8, facecolor=palette(drive - 1),
                                       alpha=.75, edgecolor=RUST if seen_twice else "white",
                                       linewidth=3 if seen_twice else 1))
                ax.text(x + 0.375, y + 0.4, str(drive), ha="center", va="center",
                        fontsize=10, color="white", fontweight="bold")
        if title.startswith("KFold"):
            d = order[0]
            inside = int((order[:8] == d).sum())
            note = (f"Drive {d}: {inside} of its 4 readings validate, {4 - inside} trained the model.\n"
                    "The model has already seen this drive: near-copies, same label.")
        else:
            note = ("Drives 1 and 2: all their readings validate, none trained.\n"
                    "The model meets them for the first time, as it will in use.")
        ax.text(0, -0.25, note, ha="left", va="top", fontsize=10.2, color=INK)
        ax.set_title(title, fontsize=11.5, color=colour, fontweight="bold", loc="left")
        ax.set_xlim(-0.1, 10.4)
        ax.set_ylim(-0.95, 5.0)
        ax.axis("off")
    fig.text(0.01, 0.0, "Drawn small: 10 drives, 4 readings each, one square per reading, "
             "numbered by drive. On the lesson's data, 800 drives and 10 readings each:\n"
             "a logistic regression on degree-3 features, C = 10,000, scores 0.999 split at random and 0.724 split by drive.",
             ha="left", va="top", fontsize=10.2, color=SLATE)
    fig.tight_layout(w_pad=2.5)
    save(fig, "groupkfold_diagram.png")


def selection_inside_diagram():
    """Where the feature selector sits: before the cross-validation, so it
    sees the validation rows; or inside each round, so it does not."""
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))
    sides = [
        ("Outside the loop: the code almost everyone writes", "all", RUST,
         ["1. SelectKBest keeps the 10 noise columns that\nbest match the labels of all 800 drives",
          "2. Each round: fit on 640, score on 160;\nbut those 160 already helped pick the columns"],
         "0.931 on pure noise"),
        ("Inside the pipeline: the fix", "train", TEAL,
         ["1. Each round: SelectKBest picks its own 10 columns\nfrom the 640 training drives",
          "2. Fit on the same 640, score on the 160\nthe selector never saw"],
         "0.658 on average over 20 fold seeds;\nthe truth is 0.500"),
    ]
    ends = []
    for ax, (title, sees, colour, steps, result) in zip(axes, sides):
        fold_strip(ax, 0.6, sees, colour)
        ends.append(step_boxes(ax, steps, colour, y=0.1, result=result))
        ax.set_title(title, fontsize=11.5, color=colour, fontweight="bold", loc="left")
    for ax in axes:
        ax.set_xlim(-0.1, 10.1)
        ax.set_ylim(min(ends) - 0.1, 2.0)
        ax.axis("off")
    fig.tight_layout(w_pad=2.5)
    save(fig, "selection_inside_diagram.png")


def gridsearch_diagram():
    """GridSearchCV as a grid: 25 cells, each one a 5-fold cross-validation,
    the best mean kept, then one refit on all the data."""
    ks, cs = [2, 5, 10, 20, 40], ["0.01", "0.1", "1", "10", "100"]
    fig, (grid, steps) = plt.subplots(1, 2, figsize=(12.4, 5.4),
                                      gridspec_kw={"width_ratios": [1.05, 1]})
    for row, k in enumerate(ks):
        y = 4 - row
        grid.text(-0.15, y + 0.4, f"k = {k}", ha="right", va="center", fontsize=10.5,
                  color=INK)
        for col, c in enumerate(cs):
            winner = k == 20 and c == "10"
            grid.add_patch(Rectangle((col, y), 0.92, 0.82,
                                     facecolor="#FFF4D6" if winner else "#EDF2F8",
                                     edgecolor=RUST if winner else SLATE,
                                     linewidth=2.6 if winner else 1.0))
            for fold in range(5):
                grid.add_patch(Rectangle((col + 0.08 + fold * 0.155, y + 0.48), 0.13, 0.22,
                                         facecolor=GOLD if fold == 0 else "#B8C7D9",
                                         edgecolor="none"))
            grid.text(col + 0.46, y + 0.22, "best mean\n0.800" if winner else "mean AUC",
                      ha="center", va="center", fontsize=8.2 if not winner else 8.8,
                      color=INK if winner else SLATE,
                      fontweight="bold" if winner else "normal")
    for col, c in enumerate(cs):
        grid.text(col + 0.46, 5.0, f"C = {c}", ha="center", va="bottom", fontsize=10.5,
                  color=INK)
    grid.text(2.3, 5.45, "k = columns kept (rows)  ×  C = penalty (columns)",
              ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold")
    grid.text(0, -0.15, "Each cell: 5 small blocks = its own 5-fold cross-validation.",
              ha="left", va="top", fontsize=9.8, color=SLATE)
    grid.set_xlim(-1.1, 5.0)
    grid.set_ylim(-0.6, 6.0)
    grid.axis("off")

    end = step_boxes(steps, [
        "1. For each of the 25 cells: 5-fold cross-validation,\n"
        "5 fits each, 25 × 5 = 125 fits",
        "2. The cell with the highest mean AUC wins:\n"
        "best_params_ = k 20, C 10; best_score_ = 0.800",
        "3. Refit that cell's settings on all 800 drives (fit 126):\n"
        "the model you ship, not scored again"],
        BLUE, y=0.0,
        result="0.800 is the best of 25 means, measured on the\n"
               "folds that also chose it; the truth is 0.500")
    steps.set_xlim(-0.1, 10.1)
    steps.set_ylim(end - 0.1, 0.4)
    steps.axis("off")
    fig.tight_layout(w_pad=2.0)
    save(fig, "gridsearch_diagram.png")


if __name__ == "__main__":
    print("conceptual figures for lesson 5:")
    two_recipes()
    kfold_diagram()
    groupkfold_diagram()
    selection_inside_diagram()
    gridsearch_diagram()
    nested_cv_diagram()
