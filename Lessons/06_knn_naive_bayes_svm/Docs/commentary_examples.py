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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from topics_examples import example, fmt, run  # noqa: E402


# --------------------------------------------------------------- slide 7
@example(7)
def knn_vote():
    # distance from the new pump to five stored pumps, and their labels
    pumps = [("A", 0.4, "faulty"), ("B", 0.7, "healthy"), ("C", 0.9, "healthy"),
             ("D", 1.3, "faulty"), ("E", 1.6, "faulty")]
    def vote(k):
        near = sorted(pumps, key=lambda p: p[1])[:k]
        labels = [lab for _, _, lab in near]
        return max(set(labels), key=labels.count), near
    results = {k: vote(k) for k in (1, 3, 5)}
    assert results[1][0] == "faulty" and results[3][0] == "healthy" and results[5][0] == "faulty"
    rows = "\n".join(f"| {n} | {d} | {lab} |" for n, d, lab in pumps)
    lines = "\n".join(
        f"- $k = {k}$: " + ("the nearest, " if k == 1 else f"the {k} nearest, ")
        + ", ".join(f"{n} ({lab})" for n, _, lab in v[1]) + f" → **{v[0]}**"
        for k, v in results.items())
    return f"""
### Worked example — five stored pumps, three values of k

Distances from a new pump to the five stored ones (standardised units):

| pump | distance | label |
|---|---|---|
{rows}

{lines}

Nothing was fitted and nothing is estimated: the answer is a vote, and $k$ decides
who gets one. The same pump changes class twice as $k$ grows — which is why $k$ is
chosen by cross-validation and not by taste.
"""


# --------------------------------------------------------------- slide 9
@example(9)
def scale_first():
    query = np.array([42.0, 5.6])
    a = np.array([45.0, 5.6])
    b = np.array([42.5, 6.3])
    spread = np.array([6.0, 0.3])           # typical spread of each reading
    raw = [np.linalg.norm(p - query) for p in (a, b)]
    std = [np.linalg.norm((p - query) / spread) for p in (a, b)]
    assert raw[1] < raw[0] and std[0] < std[1]
    return f"""
### Worked example — who is nearest depends on the units

A new pump at 42 Hz, 5.6 bar. Two stored pumps: A at 45 Hz, 5.6 bar; B at 42.5 Hz,
6.3 bar. Suppose vibration typically spreads by about 6 Hz and pressure by about
0.3 bar.

| distance to | A | B | nearest |
|---|---|---|---|
| raw units | {fmt(raw[0], 2)} | {fmt(raw[1], 2)} | **B** |
| standardised | {fmt(std[0], 2)} | {fmt(std[1], 2)} | **A** |

Raw, A is $\\sqrt{{3^2 + 0^2}}$ away and B $\\sqrt{{0.5^2 + 0.7^2}}$; standardised, A is
$\\sqrt{{(3/6)^2 + 0^2}}$ and B $\\sqrt{{(0.5/6)^2 + (0.7/0.3)^2}}$.

In raw units 0.7 bar looks smaller than 3 Hz, so B wins — although 0.7 bar is more
than two typical spreads of pressure and 3 Hz is half a spread of vibration.
Standardised, each reading is measured against its own spread and A is the real
neighbour.
"""


# --------------------------------------------------------------- slide 18
@example(18)
def edge_length():
    rows = []
    for d in (1, 2, 10, 100):
        edge = 0.01 ** (1 / d)
        rows.append(f"| {d} | $0.01^{{1/{d}}}$ | **{fmt(edge, 3)}** |")
    assert 0.01 ** (1 / 10) > 0.6
    table = "\n".join(rows)
    return f"""
### Worked example — how big a neighbourhood holds 1% of the data

Points spread evenly in a cube of side 1. A smaller cube that contains 1% of them
must have volume 0.01, so its side is $0.01^{{1/d}}$:

| dimensions $d$ | side of the 1% cube | as a fraction of each axis |
|---|---|---|
{table}

In two dimensions the 1% neighbourhood spans a tenth of each axis — local. In ten it
spans {fmt(0.01 ** 0.1 * 100, 0)}% of every axis, and in a hundred nearly all of it. A "nearest
neighbour" found that way is not near in any useful sense; it is just the least far.
"""


# --------------------------------------------------------------- slide 26
@example(26)
def bayes_rule():
    prior = {"faulty": 735 / 1200, "healthy": 465 / 1200}
    like = {"faulty": 0.02, "healthy": 0.05}
    num = {c: prior[c] * like[c] for c in prior}
    px = sum(num.values())
    post = {c: num[c] / px for c in num}
    assert max(num, key=num.get) == max(post, key=post.get) == "healthy"
    return f"""
### Worked example — one pump, two classes, with and without the denominator

Priors from the 1,200 pumps: $P(\\text{{faulty}}) = 735/1200 = {fmt(prior['faulty'], 4)}$,
$P(\\text{{healthy}}) = {fmt(prior['healthy'], 4)}$. Suppose the readings $x$ of one pump have
likelihood 0.02 under faulty and 0.05 under healthy.

| class | prior × likelihood | ÷ $P(x)$ = posterior |
|---|---|---|
| faulty | ${fmt(prior['faulty'], 4)} \\times 0.02 = {fmt(num['faulty'], 5)}$ | {fmt(post['faulty'], 3)} |
| healthy | ${fmt(prior['healthy'], 4)} \\times 0.05 = {fmt(num['healthy'], 5)}$ | **{fmt(post['healthy'], 3)}** |

$P(x) = {fmt(num['faulty'], 5)} + {fmt(num['healthy'], 5)} = {fmt(px, 5)}$, the same divisor for both rows.
Dividing changes the numbers into probabilities but cannot change which is larger:
the decision is already made in the middle column. Here the rarer class wins,
because its likelihood is two and a half times larger.
"""


# --------------------------------------------------------------- slide 28
@example(28)
def parameter_count():
    rows = []
    for n in (2, 10, 30):
        rows.append(f"| {n} | $2^{{{n}}} - 1 = {2 ** n - 1:,}$ | {n} |")
    assert 2 ** 30 - 1 > 10 ** 9
    table = "\n".join(rows)
    return f"""
### Worked example — what the assumption saves, counted

Take $n$ yes/no readings per pump. Describing $P(x \\mid y = c)$ exactly needs a
probability for every combination of answers; under the naive assumption it needs
one probability per reading:

| readings $n$ | joint distribution, per class | naive, per class |
|---|---|---|
{table}

With 30 readings the joint table has over a billion cells per class, almost all of
them never seen in any training set. The naive model needs 30 numbers, each
estimated from every row of the class. That is the bargain: it is false, and it is learnable.
"""


# --------------------------------------------------------------- slide 41
@example(41)
def xor_blind():
    # 100 pumps per cell of (vibration high?, pressure high?); faulty iff exactly one is high
    cells = {(0, 0): "healthy", (0, 1): "faulty", (1, 0): "faulty", (1, 1): "healthy"}
    n = 25
    p_vib_high = {c: sum(n for (v, _), cl in cells.items() if cl == c and v == 1) /
                  sum(n for cl in cells.values() if cl == c) for c in ("faulty", "healthy")}
    p_pre_high = {c: sum(n for (_, q), cl in cells.items() if cl == c and q == 1) /
                  sum(n for cl in cells.values() if cl == c) for c in ("faulty", "healthy")}
    assert p_vib_high["faulty"] == p_vib_high["healthy"] == 0.5
    assert p_pre_high["faulty"] == p_pre_high["healthy"] == 0.5
    return f"""
### Worked example — the interaction, counted

100 pumps, 25 in each cell. A pump is faulty when **exactly one** reading is high:

| | pressure low | pressure high |
|---|---|---|
| **vibration low** | 25 healthy | 25 faulty |
| **vibration high** | 25 faulty | 25 healthy |

What Naive Bayes estimates is one column at a time, within each class:

- $P(\\text{{vibration high}} \\mid \\text{{faulty}}) = 25/50 = {fmt(p_vib_high['faulty'], 1)}$, and
  $\\mid \\text{{healthy}}$ also {fmt(p_vib_high['healthy'], 1)};
- $P(\\text{{pressure high}} \\mid \\text{{faulty}}) = {fmt(p_pre_high['faulty'], 1)}$, and
  $\\mid \\text{{healthy}}$ also {fmt(p_pre_high['healthy'], 1)}.

Every factor is the same for both classes, so every pump gets the same score and
the model can only fall back on the prior. The rule lives entirely in the
combination, and the combination is exactly what the assumption throws away.
"""


# --------------------------------------------------------------- slide 45
@example(45)
def margin_distance():
    w, b = np.array([3.0, 4.0]), -10.0
    x = np.array([2.0, 3.0])
    norm = np.linalg.norm(w)
    dist = abs(w @ x + b) / norm
    assert norm == 5 and abs(dist - 1.6) < 1e-12
    return f"""
### Worked example — distance to a boundary, and the width of the slab

Boundary $3x_1 + 4x_2 - 10 = 0$, so $w = (3, 4)$ and $\\lVert w \\rVert = 5$. A pump at
$(2, 3)$:

$$\\text{{distance}} = \\frac{{|3 \\cdot 2 + 4 \\cdot 3 - 10|}}{{5}} = \\frac{{8}}{{5}} = {fmt(dist, 1)}$$

If the closest pumps sit where $|w^\\top x + b| = 1$, they are $1/5 = 0.2$ from the
boundary and the slab is $2/\\lVert w \\rVert = 0.4$ wide. A boundary whose closest pumps
were twice as far off, at 0.4, would satisfy the same rule only with
$\\lVert w \\rVert = 1/0.4 = 2.5$, and its slab would be 0.8 wide. Under that rule a shorter
$w$ **is** a wider margin, which is why the SVM minimises $\\lVert w \\rVert$.
"""


# --------------------------------------------------------------- slide 48
@example(48)
def hinge():
    rows = []
    for m in (2.0, 1.0, 0.5, 0.0, -1.0):
        h = max(0.0, 1 - m)
        lg = math.log(1 + math.exp(-m))
        where = ("outside the slab, right side" if m > 1 else "on the edge" if m == 1 else
                 "inside the slab" if m > 0 else "on the boundary" if m == 0 else "wrong side")
        rows.append(f"| {m} | {where} | **{fmt(h, 2)}** | {fmt(lg, 3)} |")
    assert max(0, 1 - 2.0) == 0
    table = "\n".join(rows)
    return f"""
### Worked example — five pumps, two losses

The margin of a pump is $y \\cdot f(x)$: positive when it is classified correctly,
larger the further it sits on its own side.

| margin $y f(x)$ | where it sits | hinge $\\max(0, 1 - y f)$ | log loss $\\log(1 + e^{{-y f}})$ |
|---|---|---|---|
{table}

Past a margin of 1 the hinge is exactly zero: that pump stops influencing the
boundary at all. The log loss never reaches zero, so every pump keeps a vote. That
zero is where support vectors come from.
"""


# --------------------------------------------------------------- slide 50
@example(50)
def c_tradeoff():
    wide = {"w": 1.0, "xi": 1.5}             # wide slab, two violations costing 1.5 in total
    narrow = {"w": 3.0, "xi": 0.0}           # narrow slab, no violations
    rows = []
    for C in (0.1, 1, 10):
        cw = 0.5 * wide["w"] ** 2 + C * wide["xi"]
        cn = 0.5 * narrow["w"] ** 2 + C * narrow["xi"]
        rows.append(f"| {C} | ${fmt(0.5 * wide['w'] ** 2, 1)} + {C} \\times 1.5 = {fmt(cw, 2)}$ | "
                    f"${fmt(0.5 * narrow['w'] ** 2, 1)} + 0 = {fmt(cn, 2)}$ | **{'wide' if cw < cn else 'narrow'}** |")
    assert rows[0].endswith("**wide** |") and rows[2].endswith("**narrow** |")
    table = "\n".join(rows)
    return f"""
### Worked example — two candidate boundaries, three prices

Boundary W is wide, $\\lVert w \\rVert = 1$, but two pumps violate it, with slack totalling
1.5. Boundary N is narrow, $\\lVert w \\rVert = 3$, and violates nothing. The SVM minimises
$\\tfrac12 \\lVert w \\rVert^2 + C \\sum \\xi_i$:

| $C$ | cost of W | cost of N | chosen |
|---|---|---|---|
{table}

Cheap violations buy the wide, calm boundary; expensive ones force the narrow
boundary that bends to every training point. Large $C$ means **less**
regularisation — the direction that catches people out.
"""


# --------------------------------------------------------------- slide 58
@example(58)
def rbf_values():
    gamma = 0.5
    half = math.sqrt(math.log(2) / gamma)
    rows = []
    for d in (0, 0.5, 1, half, 2, 3):
        k = math.exp(-gamma * d * d)
        label = f"{fmt(d, 3)} (half-way)" if d == half else fmt(d, 1)
        rows.append(f"| {label} | {fmt(k, 3)} |")
    assert abs(math.exp(-gamma * half ** 2) - 0.5) < 1e-12
    table = "\n".join(rows)
    return f"""
### Worked example — how similar two pumps are, by distance

$K(x, x') = e^{{-\\gamma \\lVert x - x' \\rVert^2}}$ with scikit-learn's default on the
standardised pumps, $\\gamma = 0.5$:

| distance (standard deviations) | similarity $K$ |
|---|---|
{table}

Identical pumps score 1; at {fmt(half, 2)} standard deviations the similarity has
halved; at 3 it is about 1%. Every vote in the SVM's prediction is weighted by one
of these numbers, so a pump three spreads away is, in effect, not consulted.
"""


if __name__ == "__main__":
    run(__file__)
