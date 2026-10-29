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
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

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
            label="the same reading - one counter's value - on every drive: odds × 6")
    ax.plot([], [], color=RUST, lw=2.4, label="what it does to the risk")
    ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    ax.set_xlabel("log-odds  log(p ÷ (1 − p))")
    ax.set_ylabel("probability of failing  p")
    ax.set_ylim(-0.04, 1.06)
    fig.tight_layout()
    save(fig, "evidence_on_the_sigmoid.png")


def surprise_curve():
    """-log p, with the six factors of section 3.1's two settings marked."""
    p = np.linspace(0.02, 1, 400)
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.plot(p, -np.log(p), lw=2.8, color=BLUE, zorder=2)
    for factors, colour, name in (((0.8, 0.9, 0.8), TEAL, "setting 1"),
                                  ((0.3, 0.4, 0.5), RUST, "setting 2")):
        values = np.array(factors)
        ax.scatter(values, -np.log(values), s=70, color=colour, edgecolor=INK,
                   zorder=4, label=f"{name}: factors {', '.join(map(str, factors))}")
    ax.annotate("drive A under setting 2:\n−log 0.3 = 1.20", xy=(0.3, -np.log(0.3)),
                xytext=(0.42, 2.1), fontsize=10,
                arrowprops=dict(arrowstyle="->", color=INK))
    ax.set_xlabel("probability the model gave to what happened, p")
    ax.set_ylabel("surprise, −log p")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 4)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.set_title("Sure and right costs nothing; unlikely and true costs a lot",
                 fontsize=10.5)
    fig.tight_layout()
    save(fig, "surprise_curve.png")


def harmonic_vs_arithmetic():
    """Both means for recall = 1, as precision varies; the flag-all model marked."""
    precision = np.linspace(0.001, 1, 400)
    arithmetic = (precision + 1) / 2
    harmonic = 2 * precision / (precision + 1)
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.plot(precision, arithmetic, lw=2.8, color=RUST, ls="--",
            label="arithmetic mean, (P + 1) / 2")
    ax.plot(precision, harmonic, lw=2.8, color=BLUE,
            label="harmonic mean, F1 = 2PR / (P + R), here 2P / (P + 1)")
    flag_all = 76 / 2000
    for value in ((flag_all + 1) / 2, 2 * flag_all / (flag_all + 1)):
        ax.scatter([flag_all], [value], s=70, color=GOLD, edgecolor=INK, zorder=4)
    ax.annotate("flag every drive, P = 76 / 2,000 = 0.038:\narithmetic (0.038 + 1) / 2 = 0.519",
                xy=(flag_all, (flag_all + 1) / 2), xytext=(0.42, 0.30), fontsize=10,
                arrowprops=dict(arrowstyle="->", color=INK))
    ax.annotate("harmonic 2 × 0.038 / 1.038 = 0.073",
                xy=(flag_all, 2 * flag_all / (flag_all + 1)), xytext=(0.14, 0.12),
                fontsize=10, arrowprops=dict(arrowstyle="->", color=INK))
    ax.set_xlabel("precision P, with recall fixed at 1")
    ax.set_ylabel("mean of precision and recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.set_title("The arithmetic mean never drops below one half", fontsize=10.5)
    fig.tight_layout()
    save(fig, "harmonic_vs_arithmetic.png")


def expected_cost_lines():
    """Expected cost of flagging and of leaving a drive, against its p."""
    cost_fp, cost_fn = 140, 2600
    p = np.linspace(0, 0.15, 300)
    star = cost_fp / (cost_fp + cost_fn)
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.plot(p, (1 - p) * cost_fp, lw=2.8, color=TEAL,
            label="flag it: (1 − p) × €140")
    ax.plot(p, p * cost_fn, lw=2.8, color=RUST,
            label="leave it: p × €2,600")
    ax.axvline(star, color=GOLD, ls="--", lw=2)
    ax.text(star + 0.003, 330, "t* = 140 / (140 + 2,600) = 0.051", fontsize=10)
    ax.set_xlabel("the drive's probability of failing, p")
    ax.set_ylabel("expected cost (EUR)")
    ax.set_xlim(0, 0.15)
    ax.set_ylim(0, 400)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.set_title("Flag the drive where flagging is the cheaper line", fontsize=10.5)
    fig.tight_layout()
    save(fig, "expected_cost_lines.png")


def logistic_neuron():
    """Logistic regression drawn as one unit: weigh the six counters, add b,
    squash with the sigmoid. Beside it, the sigmoid against the hard step of
    Rosenblatt's perceptron, the same diagram with a different squash."""
    fig, (ax, curve) = plt.subplots(1, 2, figsize=(12.4, 5.0),
                                    gridspec_kw={"width_ratios": [1.55, 1]})

    def node(xy, label, colour, r=0.36, size=15):
        ax.add_patch(Circle(xy, r, facecolor="white", edgecolor=colour, lw=2.2,
                            zorder=3))
        ax.text(*xy, label, ha="center", va="center", fontsize=size, zorder=4)

    inputs = [("x₁", "reallocated sectors"), ("x₂", "spin retries"),
              ("x₃", "read errors"), ("x₄", "seek errors"),
              ("x₅", "temperature"), ("x₆", "power-on hours")]
    s, out = (5.0, 2.6), (7.4, 2.6)
    for k, (symbol, name) in enumerate(inputs):
        xy = (2.2, 5.2 - 0.86 * k)
        ax.add_patch(FancyArrowPatch(xy, s, arrowstyle="-|>", mutation_scale=12,
                                     lw=1.4, color=BLUE, shrinkA=20, shrinkB=26,
                                     zorder=2))
        node(xy, symbol, SLATE)
        ax.text(xy[0] - 0.5, xy[1], name, ha="right", va="center", fontsize=11,
                color=SLATE)
        at = 0.36
        ax.text(xy[0] + at * (s[0] - xy[0]), xy[1] + at * (s[1] - xy[1]),
                f"w{symbol[1]}", ha="center", va="center", fontsize=11, color=BLUE,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"), zorder=4)
    one = (2.2, 5.2 - 0.86 * 6)
    ax.add_patch(FancyArrowPatch(one, s, arrowstyle="-|>", mutation_scale=12,
                                 lw=1.4, color=RUST, shrinkA=20, shrinkB=26, zorder=2))
    node(one, "1", SLATE)
    ax.text(one[0] - 0.5, one[1], "a constant 1", ha="right", va="center",
            fontsize=11, color=SLATE)
    ax.text(one[0] + 0.36 * (s[0] - one[0]), one[1] + 0.36 * (s[1] - one[1]), "b",
            ha="center", va="center", fontsize=12, color=RUST,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"), zorder=4)
    ax.add_patch(FancyArrowPatch(s, out, arrowstyle="-|>", mutation_scale=14, lw=1.6,
                                 color=SLATE, shrinkA=26, shrinkB=26, zorder=2))
    node(s, "Σ", BLUE, r=0.48, size=18)
    node(out, "σ", GOLD, r=0.48, size=18)
    ax.text(s[0], s[1] + 0.85, "z = wᵀx + b", ha="center", fontsize=13)
    ax.text(out[0] + 0.65, out[1], "p = P(fail | x)", ha="left", va="center",
            fontsize=13)
    ax.text(5.2, 5.95, "logistic regression is one neuron: weigh, add, squash", ha="center",
            fontsize=13, weight="bold", color=SLATE)
    ax.set_xlim(-0.6, 10.2)
    ax.set_ylim(-0.4, 6.3)
    ax.set_aspect("equal")
    ax.axis("off")

    zz = np.linspace(-6, 6, 400)
    curve.plot(zz, 1 / (1 + np.exp(-zz)), color=GOLD, lw=2.8,
               label="sigmoid: a probability")
    curve.plot(zz, (zz > 0).astype(float), color=SLATE, lw=2, ls="--",
               label="step: 0 or 1")
    curve.axhline(0.5, color=SLATE, lw=.6, ls=":")
    curve.axvline(0, color=SLATE, lw=.6, ls=":")
    curve.set_xlabel("z = wᵀx + b")
    curve.set_ylabel("output")
    curve.set_yticks([0, 0.5, 1])
    curve.text(1.3, 0.42, "sigmoid:\na probability", fontsize=11, color=INK)
    curve.text(-5.8, 0.07, "step: 0 or 1", fontsize=11, color=SLATE)
    curve.set_title("The squash: the sigmoid here, a step in Rosenblatt's\n"
                    "perceptron (1958); lesson 9 builds networks of such units",
                    fontsize=11)
    curve.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "logistic_neuron.png")


if __name__ == "__main__":
    print("figure concettuali della lezione 4:")
    sigmoid_and_odds()
    metric_denominators()
    evidence_on_the_sigmoid()
    surprise_curve()
    harmonic_vs_arithmetic()
    expected_cost_lines()
    logistic_neuron()
