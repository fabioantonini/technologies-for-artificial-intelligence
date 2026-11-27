"""Worked examples for the lesson's commentary, Docs/*_commentary.md.

Every number in the examples is computed here and written into the text by this
script, so nothing is transcribed by hand. Run it from the lesson's Docs/
folder; it prints each example's markdown; with --write it splices them into
the commentary under the slide they belong to, and with --check (run by
tools/verify_lesson.py) it fails if the commentary holds anything else.

The values are invented and deliberately small - sums a student can follow by
hand - and each example checks itself against the claim the slide makes before
it is printed.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from topics_examples import example, fmt, run  # noqa: E402


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# --------------------------------------------------------------- slide 9
@example(9)
def one_unit_more_points():
    w1, w2, b = 1.0, 2.0, -1.0
    def term(v):                                # (−0.5) for a negative factor
        return f"({v:g})".replace("-", "−") if v < 0 else f"{v:g}"

    rows, sides = [], []
    for x1, x2 in ((0.5, 0.5), (2.0, -0.5), (0.0, 0.5)):
        z = w1 * x1 + w2 * x2 + b
        p = sigmoid(z)
        side = "class 1" if p > 0.5 else "class 0" if p < 0.5 else "on the line"
        sides.append(side)
        rows.append(f"| ({x1:g}, {x2:g}) | 1 × {term(x1)} + 2 × {term(x2)} − 1 = {z:g} | "
                    f"σ({z:g}) = {fmt(p, 3)} | {side} |".replace("(2, -0.5)", "(2, −0.5)"))
    # The sign of z decides the class; z = 0 is the line itself.
    assert sides == ["class 1", "on the line", "on the line"]
    table = "\n".join(rows)
    return f"""
### Worked example — three more points for the unit of slide 9

The same unit, $w = (1, 2)$, $b = -1$; the rule is "class 1 when $z > 0$".

| point $(x_1, x_2)$ | $z = w_1x_1 + w_2x_2 + b$ | $\\hat{{y}} = \\sigma(z)$ | class |
|---|---|---|---|
{table}

Two of the three land exactly on the line $x_1 + 2x_2 - 1 = 0$, although they look
nothing alike: the line runs through every point whose weighted sum is zero.
"""


# --------------------------------------------------------------- slide 34
@example(34)
def one_step_with_a_zero_input():
    w1, w2, b, x1, x2, y, alpha = 1.0, 2.0, -1.0, 0.0, 1.0, 0, 0.5
    z = w1 * x1 + w2 * x2 + b
    p = sigmoid(z)
    blame = p - y
    g1, g2, gb = blame * x1, blame * x2, blame
    n1, n2, nb = w1 - alpha * g1, w2 - alpha * g2, b - alpha * gb
    p_new = sigmoid(n1 * x1 + n2 * x2 + nb)
    assert g1 == 0 and n1 == w1 and p_new < p
    loss, loss_new = -math.log(1 - p), -math.log(1 - p_new)
    assert loss_new < loss
    return f"""
### Worked example — one step where the other input is 0

The unit of slide 9 sees the point $(0, 1)$, true label $y = 0$, with $\\alpha = 0.5$.

| | weight from $x_1$ | weight from $x_2$ | bias |
|---|---|---|---|
| input | {x1:g} | {x2:g} | 1 |
| blame = $(\\hat{{y}} - y) \\times$ input | {fmt(blame, 3)} × {x1:g} = {fmt(g1, 3)} | {fmt(blame, 3)} × {x2:g} = {fmt(g2, 3)} | {fmt(blame, 3)} × 1 = {fmt(gb, 3)} |
| new value = old − {alpha:g} × blame | {w1:g} − {alpha:g} × {fmt(g1, 3)} = {fmt(n1, 3)} | {w2:g} − {alpha:g} × {fmt(g2, 3)} = {fmt(n2, 3)} | −1 − {alpha:g} × {fmt(gb, 3)} = {fmt(nb, 3).replace("-", "−")} |

Forward first: $z = 0 + 2 - 1 = {z:g}$ and $\\hat{{y}} = \\sigma({z:g}) = {fmt(p, 3)}$. After the
step, $\\hat{{y}} = {fmt(p_new, 3)}$ and the loss falls from ${fmt(loss, 3)}$ to ${fmt(loss_new, 3)}$. The weight
from $x_1$ has not moved: its input was 0.
"""


# --------------------------------------------------------------- slide 44
@example(44)
def softmax_three_scores():
    z = (1.0, 1.0, 0.0)
    e = [math.exp(v) for v in z]
    total = sum(e)
    p = [v / total for v in e]
    assert abs(sum(p) - 1) < 1e-12 and abs(p[0] - p[1]) < 1e-12
    shifted = [math.exp(v + 3) for v in z]
    p_shift = [v / sum(shifted) for v in shifted]
    assert all(abs(a - b) < 1e-12 for a, b in zip(p, p_shift))
    rows = "\n".join(f"| {k} | {z[k]:g} | {fmt(e[k], 3)} | {fmt(e[k], 3)} ÷ {fmt(total, 3)} = {fmt(p[k], 3)} |"
                     for k in range(3))
    return f"""
### Worked example — softmax with a tie

Three scores, two of them equal.

| class | score $z$ | $e^z$ | probability $= e^z \\div$ total |
|---|---|---|---|
{rows}

The total is $2.718 + 2.718 + 1 = {fmt(total, 3)}$. The two tied scores share the top
equally, {fmt(p[0], 3)} each, and the probabilities add up to 1 (0.999 here only
because each is rounded to three places). Add 3 to every score and
nothing changes: each $e^z$, and the total, is multiplied by $e^3$.
"""


if __name__ == "__main__":
    run(__file__)
