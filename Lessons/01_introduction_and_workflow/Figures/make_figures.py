"""Generate the conceptual figures for lesson 1.

    python Figures/make_figures.py

Data figures come from the notebooks, where the code that produces them is part
of what students read. These are different: diagrams that illustrate an idea
rather than a dataset. They still live in code so they stay reproducible and
restyleable, but they would be noise inside a teaching notebook.

Run from the lesson folder, or from anywhere - paths are resolved relative to
this file.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent

# Course palette, matching Course/template.pptx
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


def rules_versus_learning():
    """The inversion: rules + data -> answers, versus data + answers -> rules."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

    def box(ax, x, y, w, h, label, colour, text_colour="white"):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=colour, edgecolor="none"))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=11, color=text_colour, weight="bold")

    def arrow(ax, x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=16, color=SLATE, lw=1.6))

    for ax, title in zip(axes, ["Traditional programming", "Machine learning"]):
        ax.set_xlim(0, 10); ax.set_ylim(0, 4)
        ax.axis("off")
        ax.set_title(title, fontsize=12.5, weight="bold", pad=12)

    # Traditional: rules + data -> answers
    box(axes[0], 0.3, 2.3, 2.6, 1.0, "Rules", BLUE)
    box(axes[0], 0.3, 0.7, 2.6, 1.0, "Data", TEAL)
    box(axes[0], 6.9, 1.5, 2.6, 1.0, "Answers", SLATE)
    arrow(axes[0], 3.1, 2.8, 6.7, 2.2)
    arrow(axes[0], 3.1, 1.2, 6.7, 1.8)
    axes[0].text(4.9, 0.25, "you write the rules", ha="center",
                 fontsize=10, style="italic", color=SLATE)

    # ML: data + answers -> rules
    box(axes[1], 0.3, 2.3, 2.6, 1.0, "Data", TEAL)
    box(axes[1], 0.3, 0.7, 2.6, 1.0, "Answers", SLATE)
    box(axes[1], 6.9, 1.5, 2.6, 1.0, "Rules", RUST)
    arrow(axes[1], 3.1, 2.8, 6.7, 2.2)
    arrow(axes[1], 3.1, 1.2, 6.7, 1.8)
    axes[1].text(4.9, 0.25, "the rules are learned", ha="center",
                 fontsize=10, style="italic", color=RUST)

    save(fig, "rules_versus_learning.png")


def overfitting():
    """Underfitting, a good fit, and memorisation - on identical data."""
    rng = np.random.default_rng(3)
    n = 22
    x = np.sort(rng.uniform(0, 1, n))
    true = lambda t: np.sin(2.2 * np.pi * t) * 0.8
    y = true(x) + rng.normal(0, 0.22, n)
    grid = np.linspace(0, 1, 400)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
    for ax, degree, title, colour in zip(
        axes, [1, 4, 18],
        ["Underfitting\n(too rigid)", "A good fit", "Overfitting\n(memorising the sample)"],
        [SLATE, TEAL, RUST],
    ):
        coeffs = np.polyfit(x, y, degree)
        ax.plot(grid, true(grid), lw=1.4, ls="--", color=GOLD, label="true pattern")
        ax.scatter(x, y, s=30, color=INK, zorder=3, label="sample")
        ax.plot(grid, np.polyval(coeffs, grid), lw=2.4, color=colour, label=f"degree {degree}")
        ax.set_title(title, fontsize=11.5, weight="bold")
        ax.set_ylim(-1.7, 1.7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8.5, frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "overfitting.png")


def risk_gap():
    """Empirical risk falls forever; expected risk turns back up."""
    complexity = np.linspace(1, 10, 200)
    empirical = 0.95 * np.exp(-0.42 * complexity) + 0.03
    expected = empirical + 0.011 * (complexity - 1) ** 2.05
    best = complexity[np.argmin(expected)]

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(complexity, empirical, lw=2.6, color=TEAL, label="Empirical risk (training data)")
    ax.plot(complexity, expected, lw=2.6, color=RUST, label="Expected risk (unseen data)")
    ax.axvline(best, ls=":", lw=1.6, color=SLATE)
    ax.annotate("the gap is overfitting",
                xy=(8.6, (empirical[-1] + expected[-1]) / 2), xytext=(6.0, 0.78),
                fontsize=10.5, color=INK,
                arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.4))
    ax.fill_between(complexity, empirical, expected, color=RUST, alpha=.10)
    ax.text(best + 0.12, 0.02, "best trade-off", fontsize=9.5, color=SLATE)

    ax.set_xlabel("model flexibility")
    ax.set_ylabel("risk (error)")
    ax.set_title("What we minimise is not what we care about", fontsize=12.5, weight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=10, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "risk_gap.png")


def workflow():
    """The eight steps, with the one that cannot move highlighted."""
    steps = [
        "1. Frame\nthe problem", "2. Inspect\nthe data", "3. Split", "4. Baseline",
        "5. Pipeline\n+ model", "6. Evaluate", "7. Diagnose", "8. Iterate",
    ]
    fig, ax = plt.subplots(figsize=(13, 2.5))
    ax.set_xlim(0, len(steps) * 1.62); ax.set_ylim(0, 2)
    ax.axis("off")

    for i, label in enumerate(steps):
        x = i * 1.62 + 0.06
        highlight = i == 2  # the split
        ax.add_patch(FancyBboxPatch(
            (x, 0.55), 1.36, 0.92,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor=RUST if highlight else BLUE, edgecolor="none"))
        ax.text(x + 0.68, 1.01, label, ha="center", va="center",
                fontsize=9.6, color="white", weight="bold" if highlight else "normal")
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch(
                (x + 1.44, 1.01), (x + 1.60, 1.01),
                arrowstyle="-|>", mutation_scale=13, color=SLATE, lw=1.5))

    ax.annotate("before anything is learned\nfrom the data",
                xy=(2 * 1.62 + 0.74, 0.52), xytext=(2 * 1.62 + 0.74, 0.02),
                ha="center", fontsize=9.5, color=RUST, weight="bold",
                arrowprops=dict(arrowstyle="-|>", color=RUST, lw=1.4))
    fig.tight_layout()
    save(fig, "ml_workflow.png")


def timeline():
    """Advance, overclaim, correction - three times over."""
    events = [
        (1943, "McCulloch\n& Pitts", "up"), (1950, "Turing\ntest", "down"),
        (1956, "Dartmouth", "up"), (1957, "Perceptron", "down"),
        (1969, "Perceptrons\n(XOR)", "up"), (1986, "Back-\npropagation", "down"),
        (1995, "SVMs,\nensembles", "up"), (2012, "AlexNet", "down"),
        (2017, "Transformer", "up"),
    ]
    winters = [(1974, 1980, "first winter"), (1987, 1993, "second winter")]

    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.set_xlim(1938, 2026); ax.set_ylim(-1.5, 1.6)
    ax.axhline(0, color=SLATE, lw=1.8, zorder=1)

    for start, end, label in winters:
        ax.axvspan(start, end, color=SLATE, alpha=.16, zorder=0)
        ax.text((start + end) / 2, -1.32, label, ha="center", fontsize=9.5,
                color=SLATE, style="italic")

    for year, label, side in events:
        y = 0.42 if side == "up" else -0.42
        va = "bottom" if side == "up" else "top"
        ax.plot([year, year], [0, y * 0.75], lw=1.3, color=SLATE, zorder=2)
        ax.scatter([year], [0], s=52, color=BLUE, zorder=3)
        ax.text(year, y, f"{year}\n{label}", ha="center", va=va,
                fontsize=9, color=INK, linespacing=1.3)

    ax.set_yticks([])
    ax.set_xticks([1940, 1960, 1980, 2000, 2020])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title("Advance, overclaim, correction - three times over",
                 fontsize=12.5, weight="bold", pad=14)
    fig.tight_layout()
    save(fig, "ai_timeline.png")


def precision_recall():
    """Moving the threshold trades one error against the other."""
    rng = np.random.default_rng(7)
    n = 900
    negative = rng.normal(-1.0, 1.0, int(n * 0.6))
    positive = rng.normal(1.2, 1.0, int(n * 0.4))
    grid = np.linspace(-4.5, 4.5, 300)

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.hist(negative, bins=45, alpha=.7, color=TEAL, label="benign", density=True)
    ax.hist(positive, bins=45, alpha=.7, color=RUST, label="malignant", density=True)

    threshold = 0.15
    ax.axvline(threshold, lw=2.2, color=INK)
    ax.text(threshold + 0.1, 0.40, "decision\nthreshold", fontsize=9.6, color=INK)

    ax.annotate("", xy=(threshold - 1.5, 0.44), xytext=(threshold - 0.15, 0.44),
                arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.5))
    ax.text(threshold - 1.6, 0.445, "fewer false alarms,\nmore missed cases",
            ha="right", fontsize=9.2, color=SLATE)
    ax.annotate("", xy=(threshold + 1.5, 0.30), xytext=(threshold + 0.15, 0.30),
                arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.5))
    ax.text(threshold + 1.6, 0.305, "catch more cases,\nmore false alarms",
            fontsize=9.2, color=SLATE)

    ax.set_xlabel("model's score for 'malignant'")
    ax.set_yticks([])
    ax.set_title("No threshold is correct on its own", fontsize=12.5, weight="bold")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    save(fig, "precision_recall_tradeoff.png")


if __name__ == "__main__":
    print("generating lesson 1 conceptual figures:")
    rules_versus_learning()
    overfitting()
    risk_gap()
    workflow()
    timeline()
    precision_recall()
