"""Generate the conceptual figures for lesson 3.

    python Figures/make_figures.py

Data figures come from the notebooks. These are the diagrams that carry an idea:
the projection picture behind least squares, the two constraint regions behind
Ridge and Lasso, the shape of a learning rate that is too large. They live in
code so they stay reproducible and restyleable.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon

OUT = Path(__file__).resolve().parent

# Course palette, matching Course/template.pptx and lessons 1-2
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


def projection_picture():
    """Least squares as a shadow: the residual is perpendicular."""
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    # the plane the features can reach
    plane = Polygon([(0.6, 1.2), (8.6, 0.6), (9.4, 2.6), (1.4, 3.2)],
                    closed=True, facecolor=TEAL, alpha=.16, edgecolor=TEAL, lw=1.6)
    ax.add_patch(plane)
    ax.text(8.0, 1.15, "everything the model\ncan predict", fontsize=9.5,
            color=TEAL, ha="right", linespacing=1.4)

    target = (4.6, 5.1)
    shadow = (4.6, 2.05)

    ax.add_patch(FancyArrowPatch((1.6, 1.9), target, arrowstyle="-|>",
                                 mutation_scale=15, color=INK, lw=2.0))
    ax.text(target[0] + 0.15, target[1] + 0.15, "y  (the true prices)",
            fontsize=11, color=INK, weight="bold")

    ax.add_patch(FancyArrowPatch((1.6, 1.9), shadow, arrowstyle="-|>",
                                 mutation_scale=15, color=BLUE, lw=2.2))
    ax.text(shadow[0] + 0.2, shadow[1] - 0.42, "ŷ  (the fit)",
            fontsize=11, color=BLUE, weight="bold")

    ax.plot([target[0], shadow[0]], [target[1], shadow[1]], lw=2.2,
            color=RUST, ls="--")
    ax.text(target[0] + 0.2, (target[1] + shadow[1]) / 2,
            "residual", fontsize=10.5, color=RUST, weight="bold")

    # right-angle mark
    ax.plot([4.32, 4.32, 4.6], [2.05, 2.33, 2.33], lw=1.4, color=RUST)

    ax.text(5.0, 0.15,
            "the closest point on the surface is the shadow, and the leftover\n"
            "error is perpendicular, or you could have done better",
            ha="center", fontsize=10, color=INK, linespacing=1.5)
    ax.set_title("Least squares is a projection", fontsize=12.5, weight="bold")
    fig.tight_layout()
    save(fig, "projection_picture.png")


def ridge_lasso_geometry():
    """Disc versus diamond: why only one of them produces zeros."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    centre = (2.05, 1.5)

    for ax, kind in zip(axes, ("ridge", "lasso")):
        # error contours around the unconstrained solution
        for radius in (0.55, 0.95, 1.35, 1.75, 2.15):
            ax.add_patch(Circle(centre, radius, fill=False, edgecolor=SLATE,
                                lw=1.0, alpha=.55))
        ax.scatter(*centre, s=55, color=SLATE, zorder=4)
        ax.text(centre[0] + 0.12, centre[1] + 0.14, "least squares",
                fontsize=9.5, color=SLATE)

        if kind == "ridge":
            ax.add_patch(Circle((0, 0), 1.0, facecolor=BLUE, alpha=.18,
                                edgecolor=BLUE, lw=2.0))
            # touch point: along the line from origin to centre
            direction = np.array(centre) / np.linalg.norm(centre)
            touch = direction * 1.0
            title = "Ridge: a disc"
            note = "smooth everywhere, so the touch point\nhas both coefficients non-zero"
        else:
            ax.add_patch(Polygon([(1, 0), (0, 1), (-1, 0), (0, -1)], closed=True,
                                 facecolor=RUST, alpha=.18, edgecolor=RUST, lw=2.0))
            touch = np.array([1.0, 0.0])
            title = "Lasso: a diamond"
            note = "corners sit on the axes, and a corner\nis a coefficient of exactly zero"

        ax.scatter(*touch, s=95, color=RUST if kind == "lasso" else BLUE,
                   zorder=5, marker="o", edgecolors="white", linewidths=1.5)
        ax.annotate("best affordable fit", xy=touch,
                    xytext=(touch[0] - 1.35, touch[1] + 1.5),
                    fontsize=9.5, color=INK,
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))

        ax.axhline(0, color=INK, lw=1.1)
        ax.axvline(0, color=INK, lw=1.1)
        ax.set_xlim(-1.9, 4.0); ax.set_ylim(-1.9, 3.9)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("w₁"); ax.set_ylabel("w₂")
        ax.set_title(title, fontsize=12, weight="bold")
        ax.text(1.05, -1.72, note, fontsize=9.3, color=SLATE, ha="center",
                linespacing=1.4)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("The budget you can afford has a shape",
                 fontsize=12.5, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "ridge_lasso_geometry.png")


def learning_rate_regimes():
    """Three step sizes on the same bowl."""
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.6))
    cost = lambda w: 0.5 * w ** 2     # curvature 1, so 4.3's threshold is alpha < 2

    for ax, (rate, steps, span, title, colour) in zip(axes, [
        (0.12, 9, 3.2, "Too small: correct, slow", SLATE),
        (0.85, 9, 3.2, "About right", TEAL),
        (2.15, 5, 6.4, "Too large: it climbs out", RUST),
    ]):
        grid = np.linspace(-span, span, 300)
        ax.plot(grid, cost(grid), lw=2, color=SLATE, alpha=.55)
        w = 2.8
        points = [w]
        for _ in range(steps):
            w = w - rate * w          # gradient of 0.5 w^2 is w
            points.append(w)
        points = np.array(points)
        ax.plot(points, cost(points), "o-", ms=5, lw=1.5, color=colour, zorder=3)
        ax.scatter([0], [0], marker="*", s=130, color=BLUE, zorder=4)
        ax.set_title(f"{title}\nα = {rate}", fontsize=11)
        ax.set_xlabel("w")
        ax.set_ylim(-0.4, 0.5 * span ** 2 * 1.05)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("cost")
    fig.tight_layout()
    save(fig, "learning_rate_regimes.png")


def coefficient_trust():
    """Estimation error against correlation with another feature."""
    features = ["distance_km", "age_years", "garage", "bathrooms", "bedrooms"]
    correlation = [0.00, 0.02, 0.04, 0.49, 0.77]
    error = [0.7, 0.5, 9.6, 8.8, 16.6]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.scatter(correlation, error, s=110, color=RUST, zorder=3)
    offsets = {"distance_km": (6, 10), "age_years": (8, -14),
               "garage": (8, 4), "bathrooms": (8, 4), "bedrooms": (-4, 12)}
    for x, y, name in zip(correlation, error, features):
        ax.annotate(name, (x, y), xytext=offsets[name], textcoords="offset points",
                    fontsize=9.5, color=INK)

    fit = np.polyfit(correlation, error, 1)
    line = np.linspace(-0.03, 0.85, 50)
    ax.plot(line, np.polyval(fit, line), lw=1.6, ls="--", color=SLATE, alpha=.8)

    ax.set_xlabel("|correlation| with area_sqm")
    ax.set_ylabel("error in the estimated coefficient (%)")
    ax.set_title("A coefficient is only as trustworthy as its feature is independent",
                 fontsize=12, weight="bold")
    ax.set_xlim(-0.05, 0.9); ax.set_ylim(-1, 20)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "coefficient_trust.png")


def bias_variance_trade():
    """Training and test error as the penalty grows.

    Computed here rather than transcribed. The previous version hard-coded its
    six points, and one of them - the leftmost - still held a pre-restack value
    (test 182.0 where the truth is 17.3). That single number manufactured the
    whole left arm of the curve and put the minimum in the wrong place.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    rng = np.random.default_rng(42)
    temp = rng.uniform(-5, 35, 30)
    use = 240 + 1.15 * (temp - 18) ** 2 + rng.normal(0, 22, 30)
    t_fit, t_held, u_fit, u_held = train_test_split(temp, use, test_size=0.3,
                                                    random_state=42)

    penalties = np.logspace(-8, 2, 11)
    train, test = [], []
    for penalty in penalties:
        fitted = make_pipeline(PolynomialFeatures(12, include_bias=False),
                               StandardScaler(), Ridge(alpha=penalty))
        fitted.fit(t_fit.reshape(-1, 1), u_fit)
        train.append(mean_squared_error(u_fit, fitted.predict(t_fit.reshape(-1, 1))) ** 0.5)
        test.append(mean_squared_error(u_held, fitted.predict(t_held.reshape(-1, 1))) ** 0.5)
    train, test = np.array(train), np.array(test)

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(penalties, train, "o-", lw=2.2, color=TEAL, label="training error")
    ax.plot(penalties, test, "o-", lw=2.2, color=RUST, label="test error")
    ax.set_xscale("log"); ax.set_yscale("log")

    best = penalties[np.argmin(test)]
    ax.axvline(best, ls=":", lw=1.5, color=SLATE)
    ax.text(best * 1.6, test.max() * 0.55, f"best: λ = {best:g}",
            fontsize=10, color=SLATE)
    ax.annotate("too flexible", xy=(penalties[0], test[0]), xytext=(1.4e-8, test[0] * 0.35),
                fontsize=9.5, color=SLATE)
    ax.annotate("too rigid", xy=(penalties[-1], test[-1]),
                xytext=(2.0, test[-1] * 0.45), fontsize=9.5, color=SLATE)

    ax.set_xlabel("penalty λ (log scale)")
    ax.set_ylabel("RMSE (kWh, log scale)")
    ax.set_title("The penalty buys generalisation, until it does not",
                 fontsize=12.5, weight="bold")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "alpha_trade_off.png")


if __name__ == "__main__":
    print("generating lesson 3 conceptual figures:")
    projection_picture()
    ridge_lasso_geometry()
    learning_rate_regimes()
    coefficient_trust()
    bias_variance_trade()
