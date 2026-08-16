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

# ---------------------------------------------------------------------------
# History section. Each figure carries the technical content of its moment,
# not a portrait - the point is that these are the same objects we use later.
# ---------------------------------------------------------------------------


def mcculloch_pitts():
    """The 1943 threshold unit - the neuron of lesson 9, with a hard step."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.5),
                                  gridspec_kw={"width_ratios": [1.35, 1]})
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    inputs = [("x₁", 4.6), ("x₂", 3.0), ("x₃", 1.4)]
    weights = ["w₁", "w₂", "w₃"]
    for (label, y), w in zip(inputs, weights):
        ax.add_patch(plt.Circle((1.2, y), 0.42, facecolor=TEAL, edgecolor="none"))
        ax.text(1.2, y, label, ha="center", va="center", color="white",
                fontsize=11, weight="bold")
        ax.add_patch(FancyArrowPatch((1.7, y), (4.5, 3.0), arrowstyle="-|>",
                                     mutation_scale=13, color=SLATE, lw=1.4))
        ax.text(2.9, (y + 3.0) / 2 + 0.18, w, fontsize=10, color=SLATE,
                ha="center", style="italic")

    ax.add_patch(plt.Circle((5.3, 3.0), 0.85, facecolor=BLUE, edgecolor="none"))
    ax.text(5.3, 3.0, "Σ", ha="center", va="center", color="white", fontsize=20)
    ax.add_patch(FancyArrowPatch((6.2, 3.0), (7.6, 3.0), arrowstyle="-|>",
                                 mutation_scale=13, color=SLATE, lw=1.4))
    ax.add_patch(plt.Circle((8.3, 3.0), 0.62, facecolor=RUST, edgecolor="none"))
    ax.text(8.3, 3.0, "y", ha="center", va="center", color="white",
            fontsize=13, weight="bold")
    ax.text(5.3, 1.55, "fires if  Σ wᵢxᵢ ≥ θ", ha="center", fontsize=11.5, color=INK)
    ax.set_title("A neuron as a threshold unit", fontsize=12.5, weight="bold")

    # The step function itself
    z = np.linspace(-3, 3, 400)
    ax2.plot(z, (z >= 0).astype(float), lw=2.6, color=RUST)
    ax2.axvline(0, ls=":", lw=1.3, color=SLATE)
    ax2.text(0.15, 0.42, "θ", fontsize=12, color=SLATE)
    ax2.set_ylim(-0.15, 1.25); ax2.set_yticks([0, 1])
    ax2.set_xticks([])
    ax2.set_title("All or nothing", fontsize=12.5, weight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "hist_1943_neuron.png")


def imitation_game():
    """1950: replace an unanswerable question with an observable one."""
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")

    def actor(x, y, label, colour):
        ax.add_patch(FancyBboxPatch((x, y), 2.5, 1.15,
                                    boxstyle="round,pad=0.03,rounding_size=0.1",
                                    facecolor=colour, edgecolor="none"))
        ax.text(x + 1.25, y + 0.58, label, ha="center", va="center",
                color="white", fontsize=11, weight="bold")

    actor(0.4, 2.4, "Interrogator", SLATE)
    actor(8.6, 4.0, "A human", TEAL)
    actor(8.6, 0.8, "A machine", RUST)

    ax.plot([6.2, 6.2], [0.3, 5.6], lw=2.2, color=INK, ls="--")
    ax.text(6.2, 5.8, "screen", ha="center", fontsize=10, color=INK, style="italic")

    for y_end in (4.6, 1.4):
        ax.add_patch(FancyArrowPatch((3.0, 3.0), (8.5, y_end), arrowstyle="<|-|>",
                                     mutation_scale=12, color=SLATE, lw=1.4))
    ax.text(6.2, 3.05, "text only", ha="center", fontsize=9.5, color=SLATE,
            bbox=dict(facecolor="white", edgecolor="none", pad=2))
    ax.text(6.0, -0.15, "Can the interrogator tell which is which?",
            ha="center", fontsize=11.5, color=INK, weight="bold")
    ax.set_title("An unanswerable question, replaced by an observable one",
                 fontsize=12.5, weight="bold")
    fig.tight_layout()
    save(fig, "hist_1950_imitation.png")


def dartmouth():
    """1956: the proposal's agenda, most of it still open."""
    topics = [
        "Automatic computers", "Language use", "Neuron nets",
        "Theory of computation", "Self-improvement",
        "Abstractions", "Randomness and creativity",
    ]
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, len(topics) + 1.6); ax.axis("off")

    for i, topic in enumerate(topics):
        y = len(topics) - i - 0.4
        ax.add_patch(FancyBboxPatch((0.5, y), 5.6, 0.62,
                                    boxstyle="round,pad=0.02,rounding_size=0.05",
                                    facecolor=BLUE if i % 2 == 0 else TEAL,
                                    edgecolor="none"))
        ax.text(0.8, y + 0.31, topic, va="center", fontsize=10.5, color="white")

    ax.text(6.6, len(topics) / 2 + 0.2,
            "“a 2 month, 10 man study”\n\n"
            "Seventy years later,\nmost of this list\nis still open.",
            fontsize=11, color=INK, va="center", linespacing=1.5)
    ax.set_title("The seven topics of the Dartmouth proposal",
                 fontsize=12.5, weight="bold")
    fig.tight_layout()
    save(fig, "hist_1956_dartmouth.png")


def perceptron_learning():
    """1957: the boundary moves as examples arrive."""
    rng = np.random.default_rng(11)
    a = rng.normal([-1.1, -0.7], 0.55, (26, 2))
    b = rng.normal([1.2, 1.0], 0.55, (26, 2))

    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    ax.scatter(a[:, 0], a[:, 1], s=34, color=TEAL, label="class A")
    ax.scatter(b[:, 0], b[:, 1], s=34, color=RUST, label="class B")

    grid = np.linspace(-2.8, 2.8, 10)
    for i, (slope, intercept, alpha) in enumerate(
        [(-2.6, -1.5, .22), (-1.5, -0.7, .38), (-0.95, -0.1, .55), (-0.75, 0.15, 1.0)]
    ):
        ax.plot(grid, slope * grid + intercept, lw=2.4 if alpha == 1 else 1.6,
                color=BLUE, alpha=alpha,
                label="learned boundary" if alpha == 1 else None)

    ax.annotate("updates as it sees mistakes", xy=(-1.9, 1.9), xytext=(-2.7, 2.5),
                fontsize=9.5, color=SLATE)
    ax.set_xlim(-2.8, 2.8); ax.set_ylim(-2.6, 2.9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.set_title("Weights adjusted from examples, not set by hand",
                 fontsize=12.5, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "hist_1957_perceptron.png")


def xor_problem():
    """1969: AND yields to a line, XOR does not."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    for ax, labels, title, separable in zip(
        axes,
        [np.array([0, 0, 0, 1]), np.array([0, 1, 1, 0])],
        ["AND — one line is enough", "XOR — no line works"],
        [True, False],
    ):
        for (x, y), label in zip(points, labels):
            ax.scatter(x, y, s=340, marker="o" if label else "s",
                       color=RUST if label else TEAL, zorder=3, edgecolors="white",
                       linewidths=1.8)
            ax.text(x, y - 0.19, str(label), ha="center", fontsize=11, color=SLATE)
        if separable:
            grid = np.linspace(-0.35, 1.35, 10)
            ax.plot(grid, 1.55 - grid, lw=2.6, color=BLUE)
        else:
            for slope, intercept in [(-1, 0.5), (-1, 1.5), (1, 0.15)]:
                grid = np.linspace(-0.35, 1.35, 10)
                ax.plot(grid, slope * grid + intercept, lw=1.7, color=SLATE,
                        alpha=.45, ls="--")
            ax.text(0.5, -0.42, "every line leaves one point wrong",
                    ha="center", fontsize=10, color=RUST, style="italic")
        ax.set_xlim(-0.4, 1.4); ax.set_ylim(-0.55, 1.4)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_title(title, fontsize=12, weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "hist_1969_xor.png")


def backpropagation():
    """1986: the error travels backwards, so hidden layers become trainable."""
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    ax.set_xlim(0, 11); ax.set_ylim(0, 6); ax.axis("off")

    layers = [(1.5, 3, "input"), (4.4, 4, "hidden"), (7.3, 3, "hidden"), (9.8, 1, "output")]
    positions = []
    for x, n, _ in layers:
        ys = np.linspace(1.2, 4.8, n) if n > 1 else [3.0]
        positions.append([(x, y) for y in ys])

    for left, right in zip(positions, positions[1:]):
        for x1, y1 in left:
            for x2, y2 in right:
                ax.plot([x1, x2], [y1, y2], lw=0.7, color=SLATE, alpha=.35, zorder=1)

    for (x, _, name), pts in zip(layers, positions):
        for px, py in pts:
            ax.add_patch(plt.Circle((px, py), 0.30, facecolor=BLUE,
                                    edgecolor="white", lw=1.4, zorder=2))
        ax.text(x, 0.45, name, ha="center", fontsize=10, color=SLATE)

    ax.add_patch(FancyArrowPatch((1.2, 5.4), (10.0, 5.4), arrowstyle="-|>",
                                 mutation_scale=15, color=TEAL, lw=2.0))
    ax.text(5.6, 5.62, "forward: compute the prediction", ha="center",
            fontsize=10.5, color=TEAL)
    ax.add_patch(FancyArrowPatch((10.0, 0.05), (1.2, 0.05), arrowstyle="-|>",
                                 mutation_scale=15, color=RUST, lw=2.0))
    ax.text(5.6, -0.35, "backward: send the error to every weight", ha="center",
            fontsize=10.5, color=RUST)
    ax.set_title("What made depth trainable", fontsize=12.5, weight="bold")
    fig.tight_layout()
    save(fig, "hist_1986_backprop.png")


def svm_margin():
    """1995: not just a separator - the one furthest from both classes."""
    rng = np.random.default_rng(5)
    a = rng.normal([-1.25, -0.85], 0.42, (18, 2))
    b = rng.normal([1.25, 0.95], 0.42, (18, 2))

    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    ax.scatter(a[:, 0], a[:, 1], s=34, color=TEAL)
    ax.scatter(b[:, 0], b[:, 1], s=34, color=RUST)

    grid = np.linspace(-2.8, 2.8, 10)
    ax.plot(grid, -0.95 * grid, lw=2.6, color=BLUE, label="maximum-margin boundary")
    for offset in (0.95, -0.95):
        ax.plot(grid, -0.95 * grid + offset, lw=1.5, ls="--", color=SLATE)
    ax.fill_between(grid, -0.95 * grid - 0.95, -0.95 * grid + 0.95,
                    color=BLUE, alpha=.09)
    ax.plot(grid, -1.9 * grid - 0.35, lw=1.4, ls=":", color=SLATE, alpha=.75,
            label="also separates, but barely")

    ax.text(1.5, 1.9, "margin", fontsize=10.5, color=BLUE, weight="bold")
    ax.set_xlim(-2.8, 2.8); ax.set_ylim(-2.6, 2.9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("Theory that explains why it generalises",
                 fontsize=12.5, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "hist_1995_svm.png")


def imagenet():
    """2012: the drop, and how ordinary the years around it look."""
    years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
    errors = [28.2, 25.8, 16.4, 11.7, 6.7, 3.6, 3.0, 2.3]

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    colours = [SLATE, SLATE, RUST] + [BLUE] * 5
    bars = ax.bar([str(y) for y in years], errors, color=colours, width=.66)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9.5)

    ax.axhline(5.1, ls="--", lw=1.6, color=GOLD)
    ax.text(7.4, 5.7, "human ≈ 5.1", fontsize=9.5, color="#8A6A18", ha="right")

    ax.annotate("AlexNet", xy=(2, 16.4), xytext=(2.5, 22.5),
                fontsize=11, color=RUST, weight="bold",
                arrowprops=dict(arrowstyle="-|>", color=RUST, lw=1.5))
    ax.set_ylabel("top-5 error (%)")
    ax.set_title("ImageNet: a scaling result, not a new principle",
                 fontsize=12.5, weight="bold")
    ax.set_ylim(0, 31)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "hist_2012_imagenet.png")


if __name__ == "__main__":
    print("generating lesson 1 conceptual figures:")
    rules_versus_learning()
    overfitting()
    risk_gap()
    workflow()
    timeline()
    precision_recall()
    mcculloch_pitts()
    imitation_game()
    dartmouth()
    perceptron_learning()
    xor_problem()
    backpropagation()
    svm_margin()
    imagenet()
