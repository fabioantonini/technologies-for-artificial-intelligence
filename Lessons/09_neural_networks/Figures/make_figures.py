"""Generate the conceptual figures for lesson 9.

    python Figures/make_figures.py

Data figures come from the notebooks, where the code that produces them is part
of what students read. These illustrate an idea rather than a result: the drift
problem before any model, the two-unit network written down by hand, one unit
beside the line it stands for, where each symbol of the forward pass lives, and
how a network learns - the training loop, one gradient step on one unit with its
numbers, and the error travelling backwards. Run them in the course container,
like every figure.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

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


def learning_loop():
    """Training as a loop: predict, measure, send the blame back, adjust, repeat."""
    fig, ax = plt.subplots(figsize=(12.0, 5.6))
    steps = [
        ((2.0, 3.6), "1. predict", "a mini-batch goes\nforward through\nthe network", BLUE),
        ((6.0, 3.6), "2. measure", "the loss: how wrong\nthe predictions are", RUST),
        ((10.0, 3.6), "3. blame", "backpropagation sends\nthe error back: how much\neach weight contributed", GOLD),
        ((6.0, 0.6), "4. adjust", "every weight moves a little\nagainst its blame:\nw ← w − α × blame", TEAL),
    ]
    for (x, y), title, text, colour in steps:
        ax.add_patch(FancyBboxPatch((x - 1.75, y - 1.0), 3.5, 2.0,
                                    boxstyle="round,pad=0.08,rounding_size=0.25",
                                    facecolor="white", edgecolor=colour, lw=2.6, zorder=2))
        ax.text(x, y + 0.55, title, ha="center", va="center", fontsize=17,
                weight="bold", color=colour, zorder=3)
        ax.text(x, y - 0.25, text, ha="center", va="center", fontsize=13, zorder=3,
                linespacing=1.3)

    def arrow(start, end, rad=0.0):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=22,
                                     lw=2.2, color=SLATE, connectionstyle=f"arc3,rad={rad}",
                                     zorder=1))
    arrow((3.8, 3.6), (4.2, 3.6))
    arrow((7.8, 3.6), (8.2, 3.6))
    arrow((10.0, 2.55), (7.8, 0.9), rad=-0.25)
    arrow((4.2, 0.9), (2.0, 2.55), rad=-0.25)
    ax.text(1.4, 1.05, "next mini-batch:\nnew weights, so\nnew predictions", ha="center",
            fontsize=12.5, color=SLATE, style="italic")
    ax.text(10.7, 1.05, "one pass over all\nmini-batches = one epoch", ha="center",
            fontsize=12.5, color=SLATE, style="italic")
    ax.set_xlim(-0.2, 12.2)
    ax.set_ylim(-0.6, 4.9)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(OUT / "learning_loop.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def one_unit_step():
    """One gradient step on one unit, with every number on the picture."""
    w1, w2, b, x1, x2, y, alpha = 1.0, 2.0, -1.0, 1.0, 1.0, 0, 0.5
    z = w1 * x1 + w2 * x2 + b
    y_hat = 1 / (1 + np.exp(-z))
    blame = y_hat - y
    nw1, nw2, nb = w1 - alpha * blame * x1, w2 - alpha * blame * x2, b - alpha * blame
    z_new = nw1 * x1 + nw2 * x2 + nb
    y_new = 1 / (1 + np.exp(-z_new))

    fig, ax = plt.subplots(figsize=(13.0, 6.0))
    xs = [(0.8, 4.2), (0.8, 2.4), (0.8, 0.6)]
    s, out = (5.2, 2.4), (8.6, 2.4)
    for start, label in zip(xs, (f"w₁ = {w1:g}", f"w₂ = {w2:g}", "b = −1")):
        ax.add_patch(FancyArrowPatch(start, s, arrowstyle="-|>", mutation_scale=14,
                                     lw=1.8, color=BLUE, shrinkA=26, shrinkB=32, zorder=1))
        mx, my = (start[0] + s[0]) / 2, (start[1] + s[1]) / 2
        ax.text(mx - 0.2, my + 0.32, label, ha="center", fontsize=14, color=BLUE,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    ax.add_patch(FancyArrowPatch(s, out, arrowstyle="-|>", mutation_scale=14, lw=1.8,
                                 color=BLUE, shrinkA=32, shrinkB=32, zorder=1))
    node(ax, xs[0], "x₁", SLATE)
    node(ax, xs[1], "x₂", SLATE)
    node(ax, xs[2], "1", SLATE)
    node(ax, s, "Σ", BLUE, r=0.5)
    node(ax, out, "σ", GOLD, r=0.5)
    ax.text(xs[0][0] - 0.65, xs[0][1], f"{x1:g}", ha="right", va="center", fontsize=16)
    ax.text(xs[1][0] - 0.65, xs[1][1], f"{x2:g}", ha="right", va="center", fontsize=16)

    # Forward, in blue: the numbers that flow to the right.
    ax.text(s[0] + 0.3, s[1] + 1.25, f"z = 1·1 + 2·1 − 1 = {z:g}", ha="center", fontsize=15,
            color=BLUE)
    ax.text(out[0] + 0.75, out[1] + 0.35, f"ŷ = σ({z:g}) = {y_hat:.3f}", ha="left",
            fontsize=15, color=BLUE)
    ax.text(out[0] + 0.75, out[1] - 0.25, f"true y = {y}", ha="left", fontsize=15)

    # Backward, in rust: the blame and the step.
    ax.text(out[0] + 0.75, out[1] - 0.95, f"blame = ŷ − y = {blame:.3f}", ha="left",
            fontsize=15, color=RUST, weight="bold")
    ax.add_patch(FancyArrowPatch((out[0] - 0.3, 1.1), (s[0] + 0.3, 1.1), arrowstyle="-|>",
                                 mutation_scale=18, lw=2.2, color=RUST, zorder=1))
    ax.text((s[0] + out[0]) / 2, 0.65, "sent back", ha="center", fontsize=13, color=RUST,
            style="italic")
    ax.text(5.2, -0.55,
            f"each weight's blame = {blame:.3f} × its input;  step with α = {alpha:g}:",
            ha="center", fontsize=14, color=RUST)
    ax.text(5.2, -1.15,
            f"w₁ → {nw1:.3f},   w₂ → {nw2:.3f},   b → {nb:.3f}".replace("-", "−"),
            ha="center", fontsize=15,
            color=RUST, weight="bold")
    ax.text(5.2, -1.8,
            f"new prediction: ŷ = σ({z_new:.3f}) = {y_new:.3f}, closer to the true 0",
            ha="center", fontsize=14, color=TEAL, weight="bold")
    ax.set_xlim(-0.6, 12.6)
    ax.set_ylim(-2.2, 5.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(OUT / "one_unit_step.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def blame_backwards():
    """Backpropagation in a picture: the output's error travels back along the weights."""
    fig, ax = plt.subplots(figsize=(13.0, 5.6))
    xs = [(1.0, 3.4), (1.0, 1.6)]
    hs = [(5.6, 4.2), (5.6, 2.5), (5.6, 0.8)]
    out = (10.2, 2.5)
    for x in xs:
        for h in hs:
            ax.add_patch(FancyArrowPatch(x, h, arrowstyle="-", lw=1.0, color=BLUE,
                                         alpha=.35, shrinkA=24, shrinkB=24, zorder=1))
    for h in hs:
        ax.add_patch(FancyArrowPatch(out, h, arrowstyle="-|>", mutation_scale=16, lw=2.4,
                                     color=RUST, shrinkA=30, shrinkB=28, zorder=2))
    for h in hs:
        for x in xs:
            ax.add_patch(FancyArrowPatch(h, x, arrowstyle="-|>", mutation_scale=12, lw=1.5,
                                         color=RUST, alpha=.6, shrinkA=26, shrinkB=24,
                                         zorder=2))
    node(ax, xs[0], "x₁", SLATE)
    node(ax, xs[1], "x₂", SLATE)
    for h in hs:
        node(ax, h, "", TEAL)
    node(ax, out, "ŷ", GOLD, r=0.5)
    ax.text(out[0], out[1] - 1.05, "error\nŷ − y", ha="center", fontsize=15, color=RUST,
            weight="bold")
    ax.text(8.0, 4.55, "each hidden unit receives\nits share of the error, in\nproportion to its weight",
            ha="center", fontsize=13, color=RUST)
    ax.text(3.3, 4.75, "and passes its share\nfurther back, the same way",
            ha="center", fontsize=13, color=RUST)
    ax.text(5.6, -0.6, "one backward sweep gives every weight its blame, layer by layer, "
            "right to left", ha="center", fontsize=14, color=SLATE, style="italic")
    ax.set_xlim(-0.4, 12.2)
    ax.set_ylim(-1.0, 5.6)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(OUT / "blame_backwards.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    drift_clouds()
    two_unit_network()
    one_neuron()
    forward_pass_map()
    learning_loop()
    one_unit_step()
    blame_backwards()
    print("wrote drift_clouds, two_unit_network, one_neuron, forward_pass_map, "
          "learning_loop, one_unit_step, blame_backwards")
