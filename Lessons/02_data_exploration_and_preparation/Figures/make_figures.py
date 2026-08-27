"""Generate the conceptual figures for lesson 2.

    python Figures/make_figures.py

Data figures (distributions, correlations, leakage scores) come from the
notebooks, where the code that produces them is part of what students read.
These are different: diagrams that illustrate an idea rather than a dataset.
They still live in code so they stay reproducible and restyleable, but they
would be noise inside a teaching notebook.

Run from the lesson folder, or from anywhere - paths are resolved relative to
this file.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats

OUT = Path(__file__).resolve().parent

# Course palette, matching Course/template.pptx and lesson 1
BLUE = "#1F4E79"
TEAL = "#2E7D8A"
RUST = "#9C4221"
GOLD = "#F0BE50"
INK = "#1A1A1A"
SLATE = "#4A5568"
PAPER = "#F7F7F5"

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": SLATE,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
})


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {name}")


def missingness_mechanisms():
    """MCAR, MAR and MNAR - what the probability of being missing depends on."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    titles = [
        "MCAR", "MAR", "MNAR",
    ]
    subtitles = [
        "depends on nothing",
        "depends on an\nOBSERVED column",
        "depends on the\nMISSING value itself",
    ]
    colours = [TEAL, BLUE, RUST]

    for ax, title, subtitle, colour in zip(axes, titles, subtitles, colours):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((1.0, 3.2), 8.0, 1.3,
                                    boxstyle="round,pad=0.03,rounding_size=0.1",
                                    facecolor=colour, edgecolor="none"))
        ax.text(5.0, 3.85, "P(missing)", ha="center", va="center",
                color="white", fontsize=13, weight="bold")
        ax.add_patch(FancyArrowPatch((5.0, 3.1), (5.0, 1.9), arrowstyle="-|>",
                                     mutation_scale=15, color=SLATE, lw=1.8))
        ax.text(5.0, 1.3, subtitle, ha="center", va="center", fontsize=10.5,
                color=INK, linespacing=1.4)
        ax.set_title(title, fontsize=13.5, weight="bold", color=colour, pad=10)

    fig.suptitle("Three reasons a value can be missing", fontsize=13, y=1.05)
    fig.tight_layout()
    save(fig, "missingness_mechanisms.png")


def outlier_fences():
    """z-score and IQR fences on a normal curve, and how they were calibrated to agree."""
    x = np.linspace(-4.5, 4.5, 600)
    y = stats.norm.pdf(x)

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.plot(x, y, lw=2.2, color=INK)
    ax.fill_between(x, y, color=SLATE, alpha=.08)

    z_fence = 3.0
    iqr_fence = 2.698  # derived in handout 4.1

    for fence, colour, label, y_text in [(z_fence, RUST, "z-score fence (k=3)", 0.36),
                                          (iqr_fence, TEAL, "IQR fence (1.5×IQR)", 0.30)]:
        ax.axvline(fence, color=colour, lw=2.0, ls="--")
        ax.axvline(-fence, color=colour, lw=2.0, ls="--")
        ax.text(fence + 0.15, y_text, label, fontsize=9.5, color=colour)

    ax.fill_between(x[x > z_fence], y[x > z_fence], color=RUST, alpha=.5)
    ax.fill_between(x[x < -z_fence], y[x < -z_fence], color=RUST, alpha=.5)

    ax.set_xlabel("standardised value")
    ax.set_yticks([])
    ax.set_title("Calibrated to agree — on data that is actually normal",
                 fontsize=12.5, weight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    save(fig, "outlier_fences.png")


def dummy_variable_trap():
    """The k one-hot columns sum to the intercept column - shown as arithmetic."""
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")

    categories = ["month-to-month", "one-year", "two-year"]
    rows = [
        (categories[0], [1, 0, 0]),
        (categories[1], [0, 1, 0]),
        (categories[2], [0, 0, 1]),
    ]

    col_x = [3.6, 4.9, 6.2]
    ax.text(1.4, 5.3, "contract_type", fontsize=10.5, weight="bold", color=SLATE)
    for cx, label in zip(col_x, ["m2m", "1y", "2y"]):
        ax.text(cx, 5.3, label, ha="center", fontsize=10.5, weight="bold", color=SLATE)
    ax.text(7.7, 5.3, "sum", ha="center", fontsize=10.5, weight="bold", color=RUST)
    ax.text(9.0, 5.3, "intercept", ha="center", fontsize=10.5, weight="bold", color=BLUE)

    for i, (label, dummies) in enumerate(rows):
        y = 4.2 - i * 1.1
        ax.text(1.4, y, label, fontsize=10, color=INK)
        for cx, d in zip(col_x, dummies):
            ax.text(cx, y, str(d), ha="center", fontsize=11,
                    color=TEAL if d else SLATE, weight="bold" if d else "normal")
        ax.text(7.7, y, "= 1", ha="center", fontsize=11, color=RUST, weight="bold")
        ax.text(9.0, y, "1", ha="center", fontsize=11, color=BLUE, weight="bold")

    ax.text(5.0, 1.0,
            "the k dummy columns always sum to the intercept column — rank deficient by exactly one",
            ha="center", fontsize=10.5, color=RUST, style="italic")
    ax.set_title("Why all k dummies plus an intercept cannot work",
                 fontsize=12.5, weight="bold")
    fig.tight_layout()
    save(fig, "dummy_variable_trap.png")


def pipeline_architecture():
    """ColumnTransformer branches into a single Pipeline."""
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, label, colour, text_colour="white", fontsize=10.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=colour, edgecolor="none"))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color=text_colour, weight="bold", linespacing=1.3)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, color=SLATE, lw=1.5))

    box(0.2, 4.4, 1.7, 1.0, "X_train\nnumeric", SLATE)
    box(0.2, 2.6, 1.7, 1.0, "X_train\ncategorical", SLATE)

    box(2.6, 4.4, 2.4, 1.0, "impute\n(median)", TEAL)
    box(2.6, 2.6, 2.4, 1.0, "impute\n(most frequent)", TEAL)
    box(5.6, 4.4, 2.2, 1.0, "scale", TEAL)
    box(5.6, 2.6, 2.2, 1.0, "one-hot\nencode", TEAL)

    arrow(1.9, 4.9, 2.55, 4.9)
    arrow(1.9, 3.1, 2.55, 3.1)
    arrow(5.0, 4.9, 5.55, 4.9)
    arrow(5.0, 3.1, 5.55, 3.1)

    box(8.4, 3.5, 1.7, 1.4, "concat", BLUE, fontsize=10.5)
    arrow(7.8, 4.9, 8.35, 3.9)
    arrow(7.8, 3.1, 8.35, 3.6)

    box(10.4, 3.5, 1.4, 1.4, "model", RUST)
    arrow(10.1, 4.2, 10.35, 4.2)

    ax.add_patch(FancyBboxPatch((2.4, 2.2), 5.6, 3.5, boxstyle="round,pad=0.05,rounding_size=0.1",
                                facecolor="none", edgecolor=SLATE, lw=1.4, linestyle="--"))
    ax.text(5.2, 5.9, "ColumnTransformer", fontsize=10.5, color=SLATE, style="italic")

    ax.add_patch(FancyBboxPatch((0.05, 0.15), 11.85, 5.95, boxstyle="round,pad=0.05,rounding_size=0.08",
                                facecolor="none", edgecolor=BLUE, lw=1.8))
    ax.text(11.9, 0.15, "one Pipeline.fit(X_train, y_train) call", fontsize=10, color=BLUE,
            ha="right", weight="bold", style="italic")

    ax.set_title("Every learned parameter comes from inside this box",
                 fontsize=12.5, weight="bold")
    fig.tight_layout()
    save(fig, "pipeline_architecture.png")


def invisible_leaks():
    """Contrast the visible Lesson 1 leak with the two invisible ones today."""
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    rows = [
        (3.9, "Lesson 1", "select features on the full dataset", "visible — an obvious extra step", RUST),
        (2.5, "Today, leak 1", "impute a missing value on the full dataset", "invisible — looks like ordinary cleaning", BLUE),
        (1.1, "Today, leak 2", "encode a category on the full dataset", "invisible — two unremarkable lines of code", TEAL),
    ]
    for y, tag, action, note, colour in rows:
        ax.add_patch(FancyBboxPatch((0.1, y), 1.7, 0.85, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=colour, edgecolor="none"))
        ax.text(0.95, y + 0.42, tag, ha="center", va="center", color="white",
                fontsize=9.5, weight="bold")
        ax.text(2.1, y + 0.62, action, fontsize=10.5, color=INK)
        ax.text(2.1, y + 0.18, note, fontsize=9.5, color=colour, style="italic")

    ax.text(5.0, 5.0, "Same rule, broken three ways — only the first one looks like a bug",
            ha="center", fontsize=12, weight="bold")
    fig.tight_layout()
    save(fig, "invisible_leaks.png")


def leak_shrinks_with_group_size():
    """Plot of the 1/n_c leak formula derived in handout 9.2."""
    n_c = np.arange(1, 51)
    leak = 1 / n_c

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(n_c, leak, lw=2.4, color=RUST, marker="o", markersize=3.5)
    ax.axvline(4.1, ls=":", lw=1.6, color=SLATE)
    ax.text(4.4, 0.75, "zip_code's average\ngroup size (4.1)", fontsize=9.5, color=SLATE)
    ax.scatter([1], [1.0], s=90, color=RUST, zorder=5)
    ax.annotate("n_c = 1 → the feature IS the label",
                xy=(1, 1.0), xytext=(12, 0.9),
                fontsize=10, color=RUST, weight="bold",
                arrowprops=dict(arrowstyle="-|>", color=RUST, lw=1.4))
    ax.set_xlabel("category size  $n_c$")
    ax.set_ylabel(r"leak size:  $(\bar{y}_c - \bar{y}_c^{(-i)})$ per unit  $(y_i - \bar{y}_c^{(-i)})$")
    ax.set_title(r"The leak shrinks as $1/n_c$", fontsize=12.5, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "leak_shrinks_with_group_size.png")

# ---------------------------------------------------------------------------
# Second batch: concepts the slides previously asserted in words alone.
# ---------------------------------------------------------------------------


def correlation_attenuation():
    """Mean imputation shrinks every correlation by sqrt(1 - p)."""
    p = np.linspace(0, 0.6, 300)
    factor = np.sqrt(1 - p)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(p * 100, factor, lw=2.8, color=RUST)
    ax.fill_between(p * 100, factor, 1, color=RUST, alpha=.09)
    ax.axhline(1.0, ls="--", lw=1.3, color=SLATE)
    ax.text(1, 1.012, "the true correlation", fontsize=9.5, color=SLATE)

    for missing, label, colour in [(0.08, "age\n8% missing", TEAL), (0.40, "40% missing", RUST)]:
        value = np.sqrt(1 - missing)
        ax.scatter([missing * 100], [value], s=70, color=colour, zorder=3)
        ax.annotate(f"{label}\n×{value:.2f}",
                    xy=(missing * 100, value),
                    xytext=(missing * 100 + 3.5, value - 0.11),
                    fontsize=10, color=colour, weight="bold", linespacing=1.4,
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.3))

    ax.set_xlabel("percentage of the column filled with the mean")
    ax.set_ylabel("what survives of the correlation")
    ax.set_title("Filling gaps with a constant erases relationships",
                 fontsize=12.5, weight="bold")
    ax.set_xlim(0, 60); ax.set_ylim(0.55, 1.06)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "correlation_attenuation.png")


def condition_number_geometry():
    """Why one learning rate cannot suit two very different variances."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))

    for ax, (sx, sy), title, note in zip(
        axes,
        [(1.0, 0.28), (1.0, 1.0)],
        ["Unscaled: a narrow valley", "Scaled: a round bowl"],
        ["one step size, bound by the steep axis:\n"
         "it bounces across and crawls along",
         "every step points at the minimum"],
    ):
        grid = np.linspace(-2.4, 2.4, 400)
        X, Y = np.meshgrid(grid, grid)
        Z = (X / sx) ** 2 + (Y / sy) ** 2
        ax.contour(X, Y, Z, levels=[0.5, 2, 4.5, 8, 12.5], colors=SLATE,
                   linewidths=1.0, alpha=.55)

        # The safe step for a quadratic of curvature c is alpha < 2/c, and the
        # binding constraint is the largest curvature. Sitting just under it is
        # what produces the zig-zag.
        # Just under the stability limit when the axes disagree, which is what
        # produces the zig-zag; comfortably inside it when they do not, which
        # is the point - scaled features let one step size suit every direction.
        curvature = 2 / min(sx, sy) ** 2
        alpha = (0.94 if sx != sy else 0.45) * (2 / curvature)

        point = np.array([2.1, 0.72 * sy])
        path = [point.copy()]
        for _ in range(26):
            grad = np.array([2 * point[0] / sx**2, 2 * point[1] / sy**2])
            point = point - alpha * grad
            path.append(point.copy())
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "o-", color=RUST, ms=3.4, lw=1.5, zorder=3)
        ax.scatter([0], [0], s=90, marker="*", color=BLUE, zorder=4)

        ax.set_title(title, fontsize=12, weight="bold")
        ax.text(0, -2.15, note, ha="center", fontsize=9.5, color=SLATE,
                style="italic", linespacing=1.4)
        ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.4, 2.4)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
    fig.suptitle("One stride length, two very different directions", fontsize=12.5,
                 weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "condition_number_geometry.png")


def missing_data_decision():
    """Four options, and the question that chooses between them."""
    fig, ax = plt.subplots(figsize=(10.6, 4.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.4); ax.axis("off")

    ax.add_patch(FancyBboxPatch((4.1, 4.2), 3.8, 0.85,
                                boxstyle="round,pad=0.03,rounding_size=0.08",
                                facecolor=INK, edgecolor="none"))
    ax.text(6.0, 4.62, "a column has gaps", ha="center", va="center",
            color="white", fontsize=11.5, weight="bold")

    options = [
        (0.3, "Drop rows", "few gaps,\nand MCAR", TEAL),
        (3.2, "Drop column", "too empty\nto be useful", SLATE),
        (6.1, "Impute", "mean, median,\nkNN, regression", BLUE),
        (9.0, "Add indicator", "keeps the MAR\nsignal itself", RUST),
    ]
    for x, label, when, colour in options:
        ax.add_patch(FancyBboxPatch((x, 2.2), 2.6, 0.85,
                                    boxstyle="round,pad=0.03,rounding_size=0.08",
                                    facecolor=colour, edgecolor="none"))
        ax.text(x + 1.3, 2.62, label, ha="center", va="center",
                color="white", fontsize=11, weight="bold")
        ax.text(x + 1.3, 1.75, when, ha="center", va="top", fontsize=9.5,
                color=SLATE, linespacing=1.4)
        ax.add_patch(FancyArrowPatch((6.0, 4.15), (x + 1.3, 3.12),
                                     arrowstyle="-|>", mutation_scale=12,
                                     color=SLATE, lw=1.3))

    ax.text(6.0, 0.55,
            "the mechanism decides — and every option except dropping the column\n"
            "learns a statistic, so it belongs inside the training fold",
            ha="center", fontsize=10.5, color=INK, linespacing=1.6)
    ax.set_title("Which one is a question about why the data is missing",
                 fontsize=12.5, weight="bold")
    fig.tight_layout()
    save(fig, "missing_data_decision.png")


def encoding_comparison():
    """The same column, three encodings, three different claims."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    categories = ["basic", "standard", "premium"]

    # Ordinal
    ax = axes[0]
    ax.barh(range(3), [1, 2, 3], color=SLATE, height=.55)
    ax.set_yticks(range(3)); ax.set_yticklabels(categories, fontsize=10)
    ax.set_xticks([1, 2, 3])
    ax.set_title("Ordinal\none column", fontsize=11.5, weight="bold")
    ax.text(1.6, -0.95, "claims premium − standard\n= standard − basic",
            fontsize=9, color=RUST, style="italic", linespacing=1.4)
    ax.set_xlim(0, 3.6)

    # One-hot
    ax = axes[1]
    matrix = np.eye(3)
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1.6)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                    fontsize=12, color="white" if matrix[i, j] else INK)
    ax.set_yticks(range(3)); ax.set_yticklabels(categories, fontsize=10)
    ax.set_xticks(range(3)); ax.set_xticklabels(categories, fontsize=8.5, rotation=30)
    ax.set_title("One-hot\nk columns", fontsize=11.5, weight="bold")
    ax.text(1.0, 3.15, "claims nothing — but k columns,\nand k = 493 for zip code",
            ha="center", fontsize=9, color=SLATE, style="italic", linespacing=1.4)

    # Target
    ax = axes[2]
    rates = [0.31, 0.22, 0.11]
    bars = ax.barh(range(3), rates, color=RUST, height=.55)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9.5)
    ax.set_yticks(range(3)); ax.set_yticklabels(categories, fontsize=10)
    ax.set_xticks([])
    ax.set_title("Target\none column", fontsize=11.5, weight="bold")
    ax.text(0.02, -0.95, "computed from the labels —\nwhich is where leakage enters",
            fontsize=9, color=RUST, style="italic", linespacing=1.4)
    ax.set_xlim(0, 0.42)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "encoding_comparison.png")


def smoking_gun():
    """Most imputed training rows borrowed from a test row."""
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    total, contaminated = 128, 98

    ax.barh([0], [total], color=SLATE, height=.52, label="rows with age imputed")
    ax.barh([0], [contaminated], color=RUST, height=.52,
            label="borrowed from at least one test row")
    ax.text(contaminated / 2, 0, f"{contaminated}", ha="center", va="center",
            color="white", fontsize=15, weight="bold")
    ax.text((contaminated + total) / 2, 0, f"{total - contaminated}",
            ha="center", va="center", color="white", fontsize=13)

    ax.set_yticks([])
    ax.set_xlim(0, total * 1.02)
    ax.set_xlabel("training rows")
    ax.legend(frameon=False, fontsize=9.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.28), ncol=2)
    ax.set_title(f"{contaminated} of {total} imputed values touched the test set",
                 fontsize=12.5, weight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    save(fig, "smoking_gun.png")


if __name__ == "__main__":
    print("generating lesson 2 conceptual figures:")
    missingness_mechanisms()
    outlier_fences()
    dummy_variable_trap()
    pipeline_architecture()
    invisible_leaks()
    leak_shrinks_with_group_size()
    correlation_attenuation()
    condition_number_geometry()
    missing_data_decision()
    encoding_comparison()
    smoking_gun()
