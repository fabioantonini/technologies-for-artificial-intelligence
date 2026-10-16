"""Generate the conceptual figures for lesson 9.

    python Figures/make_figures.py

Data figures come from the notebooks, where the code that produces them is part
of what students read. The two here introduce the drift problem before the
hand-built network is described, and draw that network, and one_neuron draws a
single unit beside the line it stands for: they illustrate an idea
rather than a result, and adding them to notebook 01 would mean re-running it,
which takes hours. Run them in the course container, like every figure.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "Notebooks"))
from instrument_data import make_channel_drift  # noqa: E402

# The palette notebook 01 uses, so these sit beside xor_hidden_space.png unchanged.
BLUE, TEAL, RUST, GOLD, SLATE = "#1F4E79", "#2E8B8B", "#B4483E", "#C9A227", "#4A5568"


def drift_clouds():
    """The four clouds of the drift problem, before any model."""
    drift = make_channel_drift()
    a, b, ok = drift["drift_a_mv"], drift["drift_b_mv"], drift["correctable"]
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    for label, colour, name in (
            (1, TEAL, "same sign (common mode): correctable"),
            (0, RUST, "opposite signs (differential): not correctable")):
        m = ok == label
        ax.scatter(a[m], b[m], s=14, c=colour, alpha=.6, edgecolor="none", label=name)
    ax.axhline(0, color=SLATE, lw=.8)
    ax.axvline(0, color=SLATE, lw=.8)
    ax.set_xlabel("channel A drift (mV)", fontsize=12)
    ax.set_ylabel("channel B drift (mV)", fontsize=12)
    ax.set_title("Two channels drift: 800 units, four clouds", fontsize=14)
    ax.legend(loc="center left", bbox_to_anchor=(1.03, 0.5), fontsize=12, frameon=False,
              markerscale=2.2)
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-2.1, 2.1)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT / "drift_clouds.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def node(ax, xy, label, colour, r=0.42):
    ax.add_patch(Circle(xy, r, facecolor="white", edgecolor=colour, lw=2.4, zorder=3))
    ax.text(*xy, label, ha="center", va="center", fontsize=18, zorder=4)


def relu_icon(ax, centre, colour, w=0.5, h=0.32):
    """The ReLU's shape, drawn inside a hidden node: flat, then rising."""
    x0, y0 = centre[0] - w / 2, centre[1] - h / 2
    ax.plot([x0, x0 + w / 2, x0 + w], [y0, y0, y0 + h], color=colour, lw=2.2,
            solid_capstyle="round", zorder=5)


def edge(ax, start, end, weight, colour, offset=(0, 0.17), at=0.5):
    """An arrow with its weight written at fraction `at` of the way along it."""
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                 lw=1.6, color=colour, shrinkA=24, shrinkB=24, zorder=2))
    mx = start[0] + at * (end[0] - start[0]) + offset[0]
    my = start[1] + at * (end[1] - start[1]) + offset[1]
    ax.text(mx, my, weight, ha="center", va="center", fontsize=14, color=colour,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"), zorder=4)


def two_unit_network():
    """The hand-built network of the drift problem: inputs, two ReLUs, one output."""
    fig, ax = plt.subplots(figsize=(10.4, 4.9))
    a, b = (0.6, 2.9), (0.6, 0.9)
    h1, h2 = (4.0, 2.9), (4.0, 0.9)
    y = (7.4, 1.9)

    edge(ax, a, h1, "+1", BLUE, (0, 0.2))
    edge(ax, b, h1, "+1", BLUE, (0, 0), at=0.24)
    edge(ax, a, h2, "−1", RUST, (0, 0), at=0.24)
    edge(ax, b, h2, "−1", RUST, (0, -0.2))
    edge(ax, h1, y, "10", BLUE, (0, 0.22))
    edge(ax, h2, y, "10", BLUE, (0, -0.22))

    node(ax, a, "a", SLATE)
    node(ax, b, "b", SLATE)
    for h in (h1, h2):
        ax.add_patch(Circle(h, 0.42, facecolor="white", edgecolor=TEAL, lw=2.4, zorder=3))
        relu_icon(ax, h, TEAL)
    node(ax, y, "", GOLD)
    ax.text(*y, "σ", ha="center", va="center", fontsize=21, zorder=4)

    ax.text(a[0], 3.75, "channel A drift", ha="center", fontsize=13, color=SLATE)
    ax.text(b[0], 0.05, "channel B drift", ha="center", fontsize=13, color=SLATE)
    ax.text(h1[0], 3.75, "h₁ = max(0, a + b − 1)", ha="center", fontsize=14)
    ax.text(h2[0], 0.05, "h₂ = max(0, −a − b − 1)", ha="center", fontsize=14)
    ax.text(y[0], 3.45, "ŷ = σ(10h₁ + 10h₂ − 1)", ha="center", fontsize=14)
    ax.text(y[0], 1.0, "correctable\nwhen ŷ > 0.5", ha="center", fontsize=13, color=SLATE)
    ax.text(h1[0] + 0.62, h1[1] - 0.05, "bias −1", fontsize=12, color=SLATE, va="center")
    ax.text(h2[0] + 0.62, h2[1] + 0.05, "bias −1", fontsize=12, color=SLATE, va="center")
    ax.text(y[0] + 0.62, y[1] + 0.05, "bias −1", fontsize=12, color=SLATE, va="center")

    # A key to the shape drawn inside the hidden nodes.
    kx, ky = 9.5, 0.75
    ax.plot([kx - 0.5, kx, kx + 0.5], [ky - 0.25, ky - 0.25, ky + 0.25], color=TEAL, lw=2)
    ax.text(kx, ky - 0.62, "ReLU(z) = max(0, z):\n0 for negative z,\nz itself otherwise",
            ha="center", va="top", fontsize=12, color=SLATE)

    ax.text(0.6, 4.35, "inputs", ha="center", fontsize=14, weight="bold", color=SLATE)
    ax.text(4.0, 4.35, "hidden layer: two ReLU units", ha="center", fontsize=14,
            weight="bold", color=SLATE)
    ax.text(7.4, 4.35, "output unit", ha="center", fontsize=14, weight="bold", color=SLATE)
    ax.set_xlim(-0.3, 10.2)
    ax.set_ylim(-0.3, 4.6)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(OUT / "two_unit_network.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def four_unit_network():
    """The trained network of the acceptance problem: 2 inputs, 4 ReLUs, 1 output."""
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    inputs = [(0.8, 3.3), (0.8, 1.3)]
    hidden = [(4.2, 4.2), (4.2, 3.0), (4.2, 1.8), (4.2, 0.6)]
    out = (7.6, 2.4)
    for s in inputs:
        for h in hidden:
            ax.plot([s[0], h[0]], [s[1], h[1]], color="#B8BEC6", lw=1.6, zorder=1)
    for h in hidden:
        ax.plot([h[0], out[0]], [h[1], out[1]], color="#B8BEC6", lw=1.6, zorder=1)
    for xy, name in zip(inputs, ("z₁", "z₂")):
        node(ax, xy, name, SLATE)
    for i, h in enumerate(hidden, 1):
        ax.add_patch(Circle(h, 0.42, facecolor="white", edgecolor=TEAL, lw=2.4, zorder=3))
        relu_icon(ax, h, TEAL)
        ax.text(h[0] + 0.6, h[1], f"h{i}", fontsize=14, va="center", color=SLATE)
    node(ax, out, "ŷ", GOLD)

    ax.text(0.8, 4.5, "2 inputs", ha="center", fontsize=15, weight="bold", color=SLATE)
    ax.text(0.8, 0.45, "gain and phase\noffsets, standardised", ha="center", va="top",
            fontsize=13, color=SLATE)
    ax.text(4.2, 5.0, "4 hidden units (ReLU)", ha="center", fontsize=15, weight="bold",
            color=SLATE)
    ax.text(4.2, -0.2, "each one draws one line", ha="center", va="top", fontsize=13,
            color=SLATE)
    ax.text(7.6, 3.2, "1 output", ha="center", fontsize=15, weight="bold", color=SLATE)
    ax.text(7.6, 1.6, "probability\nthe unit passes", ha="center", va="top", fontsize=13,
            color=SLATE)
    ax.set_xlim(-0.3, 9.0)
    ax.set_ylim(-0.9, 5.4)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(OUT / "four_unit_network.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def one_neuron():
    """One unit, two ways: the diagram, and the line it draws in the input plane."""
    w1, w2, b = 1.0, 2.0, -1.0
    fig, (ax, plane) = plt.subplots(1, 2, figsize=(12.4, 5.2),
                                    gridspec_kw={"width_ratios": [1.3, 1]})

    # Left: inputs, weights, the weighted sum, the squash, the output.
    x1, x2, one = (0.5, 3.4), (0.5, 1.8), (0.5, 0.2)
    s, out = (4.0, 1.8), (6.9, 1.8)
    edge(ax, x1, s, "w₁ = 1", BLUE, (-0.35, 0.3), at=0.35)
    edge(ax, x2, s, "w₂ = 2", BLUE, (0, 0.28), at=0.45)
    edge(ax, one, s, "b = −1", RUST, (-0.3, -0.35), at=0.35)
    ax.add_patch(FancyArrowPatch(s, out, arrowstyle="-|>", mutation_scale=14, lw=1.6,
                                 color=SLATE, shrinkA=26, shrinkB=26, zorder=2))
    node(ax, x1, "x₁", SLATE)
    node(ax, x2, "x₂", SLATE)
    node(ax, one, "1", SLATE)
    node(ax, s, "Σ", BLUE, r=0.5)
    node(ax, out, "σ", GOLD, r=0.5)
    ax.text(out[0] + 0.7, out[1], "ŷ", ha="left", va="center", fontsize=22)
    ax.text(s[0], 2.75, "z = w₁x₁ + w₂x₂ + b", ha="center", fontsize=16)
    ax.text(one[0], -0.6, "a constant 1\ncarries the bias", ha="center", va="top",
            fontsize=14, color=SLATE)
    ax.text(3.8, 4.35, "one unit: weigh, add, squash", ha="center", fontsize=17,
            weight="bold", color=SLATE)

    # The squash itself, under the σ node: the sigmoid, and the step it replaced.
    curve = ax.inset_axes([0.47, 0.0, 0.5, 0.33])
    zz = np.linspace(-6, 6, 400)
    curve.plot(zz, 1 / (1 + np.exp(-zz)), color=GOLD, lw=2.6,
               label="sigmoid σ(z)")
    curve.plot(zz, (zz > 0).astype(float), color=SLATE, lw=1.6, ls="--",
               label="step (1958)")
    curve.axhline(0.5, color=SLATE, lw=.6, ls=":")
    curve.axvline(0, color=SLATE, lw=.6, ls=":")
    curve.set_yticks([0, 0.5, 1])
    curve.set_xticks([-6, 0, 6])
    curve.tick_params(labelsize=11)
    curve.set_xlabel("z", fontsize=13, labelpad=1)
    curve.set_ylabel("ŷ", fontsize=13, labelpad=6, rotation=0, va="center")
    curve.legend(fontsize=11, loc="lower right", bbox_to_anchor=(1.1, 0.0), frameon=False,
                 handlelength=1.4)
    curve.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.3, 8.0)
    ax.set_ylim(-2.1, 4.7)
    ax.set_aspect("equal")
    ax.axis("off")

    # Right: the same unit as a picture of the input plane.
    g = np.linspace(-2, 2, 300)
    X1, X2 = np.meshgrid(g, g)
    Y = 1 / (1 + np.exp(-(w1 * X1 + w2 * X2 + b)))
    shade = plane.contourf(X1, X2, Y, levels=np.linspace(0, 1, 11), cmap="RdBu_r",
                           alpha=.55)
    plane.plot(g, (-b - w1 * g) / w2, color="black", lw=2.6)
    plane.text(-0.9, 1.6, "ŷ > 0.5: output 1", fontsize=15, weight="bold")
    plane.text(-1.9, -1.75, "ŷ < 0.5: output 0", fontsize=15, weight="bold")
    plane.annotate("on the line z = 0, ŷ = 0.5:\nx₁ + 2x₂ − 1 = 0", xy=(1.2, -0.1),
                   xytext=(-1.1, -1.05), fontsize=14,
                   bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=.9),
                   arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2))
    plane.set_xlim(-2, 2)
    plane.set_ylim(-2, 2)
    plane.set_aspect("equal")
    plane.tick_params(labelsize=12)
    plane.set_xlabel("x₁", fontsize=16)
    plane.set_ylabel("x₂", fontsize=16)
    plane.set_title("the same unit: one straight line", fontsize=17, weight="bold",
                    color=SLATE)
    bar = fig.colorbar(shade, ax=plane, fraction=0.046, pad=0.04)
    bar.set_label("ŷ", fontsize=16)
    bar.ax.tick_params(labelsize=12)

    fig.tight_layout()
    fig.savefig(OUT / "one_neuron.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def forward_pass_map():
    """Where each symbol of forward propagation lives: the 2-4-1 acceptance network."""
    fig, ax = plt.subplots(figsize=(14.0, 6.2))
    X0, H0, O0 = 1.0, 6.4, 11.8
    xs = [(X0, 3.9), (X0, 2.3)]
    hs = [(H0, 5.2), (H0, 3.85), (H0, 2.5), (H0, 1.15)]
    out = (O0, 3.1)
    r_h, r_o = 0.58, 0.62

    for x in xs:
        for h in hs:
            ax.add_patch(FancyArrowPatch(x, h, arrowstyle="-|>", mutation_scale=11,
                                         lw=1.2, color=BLUE, alpha=.5,
                                         shrinkA=26, shrinkB=34, zorder=1))
    for h in hs:
        ax.add_patch(FancyArrowPatch(h, out, arrowstyle="-|>", mutation_scale=11,
                                     lw=1.2, color=RUST, alpha=.55,
                                     shrinkA=34, shrinkB=36, zorder=1))
    node(ax, xs[0], "x₁", SLATE)
    node(ax, xs[1], "x₂", SLATE)
    for j, h in enumerate(hs, start=1):
        ax.add_patch(Circle(h, r_h, facecolor="white", edgecolor=TEAL, lw=2.4, zorder=3))
        ax.plot([h[0], h[0]], [h[1] - r_h, h[1] + r_h], color=TEAL, lw=1, zorder=4)
        ax.text(h[0] - 0.3, h[1], f"$z_{j}$", ha="center", va="center", fontsize=17, zorder=5)
        ax.text(h[0] + 0.3, h[1], f"$a_{j}$", ha="center", va="center", fontsize=17, zorder=5)
    ax.add_patch(Circle(out, r_o, facecolor="white", edgecolor=GOLD, lw=2.4, zorder=3))
    ax.plot([out[0], out[0]], [out[1] - r_o, out[1] + r_o], color=GOLD, lw=1, zorder=4)
    ax.text(out[0] - 0.32, out[1], "$z$", ha="center", va="center", fontsize=18, zorder=5)
    ax.text(out[0] + 0.32, out[1], r"$\hat{y}$", ha="center", va="center", fontsize=18, zorder=5)

    # Column headings.
    for x, label in ((X0, "inputs"), (H0, "layer 1: hidden"),
                     (O0, "layer 2: output")):
        ax.text(x, 6.95, label, ha="center", fontsize=19, weight="bold", color=SLATE)

    # What lives on the edges.
    ax.text(2.6, 6.3, r"$W^{[1]}$: 2 × 4, one column per hidden unit",
            ha="center", fontsize=17, color=BLUE)
    ax.text(2.6, 5.85, r"$b^{[1]}$: 4 biases, one per hidden unit",
            ha="center", fontsize=16, color=BLUE)
    ax.text((H0 + O0) / 2 + 0.3, 5.1, r"$W^{[2]}$: 4 × 1", ha="center", fontsize=17,
            color=RUST)
    ax.text((H0 + O0) / 2 + 0.3, 4.65, r"$b^{[2]}$: 1 bias", ha="center", fontsize=16,
            color=RUST)

    # What each layer computes.
    ax.text(H0, 0.05, r"$Z^{[1]} = XW^{[1]} + b^{[1]}$,  $A^{[1]} = g(Z^{[1]})$",
            ha="center", fontsize=18)
    ax.text(H0, -0.42, "z: weighted sum (pre-activation);   a = g(z): activation, here ReLU",
            ha="center", fontsize=15.5, color=SLATE)
    ax.text(O0, 2.05, r"$Z^{[2]} = A^{[1]}W^{[2]} + b^{[2]}$", ha="center", fontsize=18)
    ax.text(O0, 1.5, r"$\hat{y} = \sigma(Z^{[2]})$", ha="center", fontsize=18)

    # The shapes, one row per example.
    for x, label in ((X0, r"$X$: $m \times 2$"),
                     (H0, r"$Z^{[1]}$ and $A^{[1]}$: $m \times 4$"),
                     (O0, r"$Z^{[2]}$ and $\hat{y}$: $m \times 1$")):
        ax.text(x, -1.1, label, ha="center", fontsize=18, color=SLATE)
    ax.text(H0, -1.6, "shapes: one row per example, m = 2,250 training units",
            ha="center", fontsize=15.5, color=SLATE, style="italic")

    ax.set_xlim(-0.6, 13.6)
    ax.set_ylim(-1.9, 7.3)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(OUT / "forward_pass_map.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    drift_clouds()
    two_unit_network()
    four_unit_network()
    one_neuron()
    forward_pass_map()
    print("wrote drift_clouds.png, two_unit_network.png, four_unit_network.png, "
          "one_neuron.png, forward_pass_map.png")
