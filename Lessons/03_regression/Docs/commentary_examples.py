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
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from topics_examples import example, fmt, run  # noqa: E402


# --------------------------------------------------------------- slide 8
@example(8)
def squared_vs_absolute():
    y = np.array([150, 200, 700])           # prices in k EUR; the third is a mansion
    mean, median = y.mean(), np.median(y)
    def sq(c): return float(((y - c) ** 2).mean())
    def ab(c): return float(np.abs(y - c).mean())
    grid = np.arange(100, 801)
    assert grid[np.argmin([sq(c) for c in grid])] == mean
    assert grid[np.argmin([ab(c) for c in grid])] == median
    return f"""
### Worked example — the best single price for three houses

Three houses at 150, 200 and 700 thousand euros. Predict one number for all three:
which number is best depends on how errors are charged.

| guess | mean squared error | mean absolute error |
|---|---|---|
| {int(median)} (the median) | {fmt(sq(median), 0)} | **{fmt(ab(median), 1)}** |
| {int(mean)} (the mean) | **{fmt(sq(mean), 0)}** | {fmt(ab(mean), 1)} |

Squared error is minimised by the **mean**, {int(mean)}; absolute error by the
**median**, {int(median)}. One mansion drags the squared-error answer {int(mean - median)}
thousand above the median, past two of the three houses — the outlier sensitivity
on the slide, in one line. That is the price paid for differentiability and for the Gaussian story.
"""


# --------------------------------------------------------------- slide 12
@example(12)
def r_squared():
    def report(y, yhat):
        y, yhat = np.array(y, float), np.array(yhat, float)
        ss_res = ((y - yhat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return ss_res, ss_tot, 1 - ss_res / ss_tot, np.sqrt(ss_res / len(y)), np.sqrt(ss_tot / len(y))
    a = report([200, 250, 300, 450], [210, 240, 320, 430])
    b = report([290, 300, 310, 300], [295, 305, 305, 300])
    assert a[2] > b[2] and b[3] < a[3]
    return f"""
### Worked example — the same model, two test sets

Four houses (k EUR), and the model's predictions:

| test set | prices | predictions | RMSE, model | RMSE, predict the mean | $R^2$ |
|---|---|---|---|---|---|
| varied | 200, 250, 300, 450 | 210, 240, 320, 430 | {fmt(a[3], 1)} | {fmt(a[4], 1)} | **{fmt(a[2], 3)}** |
| look-alike | 290, 300, 310, 300 | 295, 305, 305, 300 | {fmt(b[3], 1)} | {fmt(b[4], 1)} | **{fmt(b[2], 3)}** |

$R^2 = 1 - \\text{{SS}}_{{\\text{{res}}}}/\\text{{SS}}_{{\\text{{tot}}}}$: on the varied set
$1 - {fmt(a[0], 0)}/{fmt(a[1], 0)}$, on the look-alikes $1 - {fmt(b[0], 0)}/{fmt(b[1], 0)}$.
The model's error **fell** from {fmt(a[3], 1)} to {fmt(b[3], 1)}, and yet $R^2$ fell too —
because the look-alike houses leave the mean-predicting baseline almost nothing to
get wrong. $R^2$ is a ratio to the test set's own spread; RMSE is in thousands of
euros.
"""


# --------------------------------------------------------------- slide 17
@example(17)
def convex_bowl():
    x, y = np.array([1, 2, 3], float), np.array([2, 3, 6], float)
    m = len(x)
    def J(w): return float(((w * x - y) ** 2).sum() / (2 * m))
    w_star = float((x * y).sum() / (x * x).sum())
    curv = float((x * x).sum() / m)
    d = 0.3
    rise = J(w_star + d) - J(w_star)
    assert abs(rise - (J(w_star - d) - J(w_star))) < 1e-12
    assert abs(rise - curv * d * d / 2) < 1e-12
    rows = "\n".join(f"| {fmt(w, 3)} | {fmt(J(w), 4)} |" for w in
                     (w_star - 2 * d, w_star - d, w_star, w_star + d, w_star + 2 * d))
    return f"""
### Worked example — one slope, no intercept, three houses

Areas 1, 2, 3 (hundreds of m²), prices 2, 3, 6 (hundreds of k EUR), model
$\\hat y = w x$. Setting the derivative to zero gives
$w^* = \\sum xy / \\sum x^2 = 26/14 = {fmt(w_star, 3)}$. The cost around it:

| $w$ | $J(w)$ |
|---|---|
{rows}

Step 0.3 either side and the cost rises by exactly the same amount,
{fmt(rise, 4)} $= \\tfrac12 \\cdot \\tfrac{{\\sum x^2}}{{m}} \\cdot 0.3^2$: the cost is a parabola
whose curvature $\\sum x^2/m = {fmt(curv, 3)}$ can never be negative. That is the
one-feature case of $\\lVert Xv \\rVert^2 \\ge 0$ — so the flat spot is the bottom.
"""


# --------------------------------------------------------------- slide 19
@example(19)
def condition_number():
    rows = []
    for r in (0.0, 0.5, 0.9, 0.99):
        ev = np.linalg.eigvalsh(np.array([[1, r], [r, 1]]))
        kappa = ev.max() / ev.min()
        assert abs(kappa - (1 + r) / (1 - r)) < 1e-9
        rows.append(f"| {r} | {fmt(1 + r, 2)} | {fmt(1 - r, 2)} | **{fmt(kappa, 0)}** |")
    table = "\n".join(rows)
    return f"""
### Worked example — two standardised features, rising correlation

Two features, each standardised, correlated $r$. The curvatures of the cost are the
eigenvalues of $\\begin{{pmatrix}}1 & r \\\\ r & 1\\end{{pmatrix}}$, which are $1 + r$ and
$1 - r$, so $\\kappa = (1 + r)/(1 - r)$:

| $r$ | steepest | shallowest | $\\kappa$ |
|---|---|---|---|
{table}

Standardising fixes the units, not the correlation: two features that nearly say
the same thing still make a stretched valley. And gradient descent needs roughly
$\\kappa$ steps, so at $r = 0.99$ about two hundred times as many as with
independent features.
"""


# --------------------------------------------------------------- slide 21
@example(21)
def weighted_vote():
    area = np.array([80, 120, 200])         # m²
    err = np.array([10, -5, -20])           # prediction minus truth, k EUR
    pulls = err * area
    grad = pulls.mean()
    share = abs(pulls[2]) / abs(pulls).sum()
    assert grad < 0 and share > 0.7
    rows = "\n".join(f"| {a} | {e:+d} | {p:+,d} |" for a, e, p in zip(area, err, pulls))
    return f"""
### Worked example — three houses voting on the price of a square metre

The gradient for the area coefficient is the average of (error × area):

| area (m²) | error $\\hat y - y$ (k EUR) | pull = error × area |
|---|---|---|
{rows}

$\\partial J/\\partial w = ({pulls[0]:+,d} {pulls[1]:+,d} {pulls[2]:+,d})/3 = {grad:+,.1f}$: negative,
so the update **raises** $w$ — the model was charging too little per m². The big
house supplies {share:.0%} of the total pull: it was the most wrong *and* the
largest. A one-room flat with the same error would barely be heard.
"""


# --------------------------------------------------------------- slide 24
@example(24)
def learning_rates():
    c, w0 = 2.0, 1.0
    rows = []
    for alpha, label in ((0.1, "small"), (0.5, "exactly 1/c"), (0.9, "large"), (1.1, "too large")):
        k = 1 - alpha * c
        path = [w0 * k ** t for t in range(1, 4)]
        rows.append(f"| {alpha} | {label} | $\\times ({fmt(k, 1)})$ | {', '.join(fmt(v, 3) for v in path)} |")
    assert abs(1 - 1.1 * c) > 1 > abs(1 - 0.9 * c)
    table = "\n".join(rows)
    return f"""
### Worked example — a bowl of curvature 2

$J(w) = \\tfrac12 \\cdot 2\\,w^2$, minimum at 0, start at $w = 1$. The gradient is $2w$,
so each step multiplies $w$ by $1 - 2\\alpha$:

| $\\alpha$ | | each step | first three steps |
|---|---|---|---|
{table}

Below $\\alpha = 1/c$ the path creeps in from one side; between $1/c$ and $2/c$ it
overshoots and oscillates but still closes in; past $2/c = 1$ every step lands
further out than the last. With several features the steepest direction sets $c$ —
which is why one badly scaled feature limits the step for all of them.
"""


# --------------------------------------------------------------- slide 29
@example(29)
def linear_in_w():
    t = np.array([-1.0, 0.0, 1.0])
    y = np.array([3.0, 1.0, 3.0])
    X = np.column_stack([np.ones(3), t, t ** 2])
    theta = np.linalg.solve(X.T @ X, X.T @ y)
    assert np.allclose(theta, [1, 0, 2])
    return f"""
### Worked example — a parabola by the normal equation

Three days: temperature $t$ = −1, 0, 1 (standardised), energy 3, 1, 3. Give the
design matrix a column of $t^2$:

$$X = \\begin{{pmatrix}} 1 & -1 & 1 \\\\ 1 & 0 & 0 \\\\ 1 & 1 & 1 \\end{{pmatrix}}, \\qquad
\\theta = (X^\\top X)^{{-1}} X^\\top y = ({fmt(theta[0], 0)}, {fmt(theta[1], 0)}, {fmt(theta[2], 0)})$$

so $\\hat y = {fmt(theta[0], 0)} + {fmt(theta[2], 0)}\\,t^2$ — the U, through all three points. Nothing in the
method changed: same equation, same code, one more column. The curve is in the
**columns**; the model is still a straight line in $(1, t, t^2)$.
"""


# --------------------------------------------------------------- slide 42
@example(42)
def ridge_one_feature():
    sxy, sxx = 26, 14                       # x = 1, 2, 3; y = 2, 3, 6
    rows = "\n".join(f"| {lam} | $26/({sxx} + {lam})$ | **{fmt(sxy / (sxx + lam), 3)}** |"
                     for lam in (0, 1, 14, 100))
    assert sxy / (sxx + 100) > 0
    return f"""
### Worked example — one coefficient, four penalties

The three houses of slide 17 ($\\sum xy = 26$, $\\sum x^2 = 14$, no intercept). With one
feature, $(X^\\top X + \\lambda I)^{{-1}} X^\\top y$ is just a division:

| $\\lambda$ | $w = \\sum xy / (\\sum x^2 + \\lambda)$ | $w$ |
|---|---|---|
{rows}

$\\lambda = 0$ is least squares. Adding $\\lambda$ to the denominator shrinks the
coefficient; $\\lambda = \\sum x^2$ halves it; no finite $\\lambda$ makes it zero.
"Tilting the floor" is literally this: the same bowl, pulled towards $w = 0$.
"""


# --------------------------------------------------------------- slide 43
@example(43)
def singular_fixed():
    xtx = np.array([[14.0, 14.0], [14.0, 14.0]])
    det0 = np.linalg.det(xtx)
    lam = 1.0
    det1 = np.linalg.det(xtx + lam * np.eye(2))
    ev = np.linalg.eigvalsh(xtx)
    assert abs(det0) < 1e-9 and abs(det1 - 29) < 1e-9
    return f"""
### Worked example — the same column twice

Put the area of slide 17 in twice, $x_1 = x_2 = (1, 2, 3)$:

$$X^\\top X = \\begin{{pmatrix}} 14 & 14 \\\\ 14 & 14 \\end{{pmatrix}}, \\quad
\\det = 14 \\cdot 14 - 14 \\cdot 14 = 0$$

No inverse: eigenvalues {fmt(ev.max(), 0)} and {fmt(abs(ev.min()), 0)}, and the zero is the flat
direction — raise $w_1$ and lower $w_2$ by the same amount and nothing changes. Add
$\\lambda = 1$ down the diagonal:

$$\\begin{{pmatrix}} 15 & 14 \\\\ 14 & 15 \\end{{pmatrix}}, \\quad
\\det = 225 - 196 = {fmt(det1, 0)}$$

Both eigenvalues moved up by 1 — to 29 and 1 — so neither is zero and the inverse
exists. Every $\\lambda > 0$ does the same.
"""


# --------------------------------------------------------------- slide 45
@example(45)
def shrink_vs_select():
    w = np.array([5.0, 2.0, 0.5])
    def ridge(lam): return w / (1 + lam)
    def lasso(lam): return np.sign(w) * np.maximum(np.abs(w) - lam, 0)
    def show(v): return ", ".join(fmt(x, 3) for x in v)
    assert (ridge(3) > 0).all() and list(lasso(3)) == [2, 0, 0]
    return f"""
### Worked example — three coefficients, both penalties

In the simplest case — features uncorrelated and of unit length, penalty written
so the formulas come out clean — each penalty acts on each least-squares
coefficient separately. Ridge **divides**, $w/(1+\\lambda)$; Lasso **subtracts and
stops at zero**, $\\operatorname{{sign}}(w)\\max(|w| - \\lambda, 0)$. Least squares gives
5, 2, 0.5:

| $\\lambda$ | Ridge | Lasso |
|---|---|---|
| 0 | {show(ridge(0))} | {show(lasso(0))} |
| 1 | {show(ridge(1))} | {show(lasso(1))} |
| 3 | {show(ridge(3))} | {show(lasso(3))} |

Ridge keeps all three, smaller. Lasso drops the smallest first and, by $\\lambda = 3$,
keeps one feature. The order it drops them in is the order of how much each was
worth — which is slide 46's table.
"""


# --------------------------------------------------------------- slide 47
@example(47)
def duplicate_split():
    # x1 = x2 = (1, 2, 3), y = (2, 3, 6): least squares only fixes w1 + w2.
    total = F(26, 14)
    candidates = [(total, F(0)), (total / 2, total / 2), (total + 5, F(-5))]
    ridge_w = F(26, 29)                      # (X'X + I)^{-1} X'y, symmetric
    assert ridge_w * 2 < total
    rows = "\n".join(f"| {fmt(float(a), 3)} | {fmt(float(b), 3)} | {fmt(float(a + b), 3)} | "
                     f"{fmt(float(a * a + b * b), 2)} |" for a, b in candidates)
    return f"""
### Worked example — one effect, two columns

The duplicated area of slide 43. Least squares only pins down the **sum**
$w_1 + w_2 = 26/14 = {fmt(float(total), 3)}$; every split predicts the same prices:

| $w_1$ | $w_2$ | $w_1 + w_2$ | $w_1^2 + w_2^2$ |
|---|---|---|---|
{rows}

Same fit, very different coefficients — and a solver will report whichever the
arithmetic happens to land on. Ridge charges $w_1^2 + w_2^2$, which is smallest for
the even split, so it chooses it: with $\\lambda = 1$, $w_1 = w_2 = 26/29 = {fmt(float(ridge_w), 3)}$,
a sum of {fmt(float(2 * ridge_w), 3)} — shrunk a little, and divided fairly.
"""


# --------------------------------------------------------------- slide 49
@example(49)
def what_a_coefficient_says():
    x1 = np.array([1.0, 2, 3, 4])
    x2 = np.array([1.0, 3, 3, 5])
    y = 2 * x1 + 3 * x2
    both = np.linalg.lstsq(np.column_stack([np.ones(4), x1, x2]), y, rcond=None)[0]
    alone = np.linalg.lstsq(np.column_stack([np.ones(4), x1]), y, rcond=None)[0]
    r = np.corrcoef(x1, x2)[0, 1]
    assert np.allclose(both[1:], [2, 3]) and abs(alone[1] - 5.6) < 1e-9
    return f"""
### Worked example — the same column, two coefficients

Price $= 2\\,x_1 + 3\\,x_2$ exactly, no noise, where $x_1$ = 1, 2, 3, 4 and $x_2$ = 1, 3, 3, 5
(correlated, $r = {fmt(r, 2)}$).

- Fit both: coefficients **{fmt(both[1], 1)}** and **{fmt(both[2], 1)}** — the truth.
- Leave $x_2$ out: the coefficient of $x_1$ becomes **{fmt(alone[1], 1)}**.

Neither fit is wrong. With $x_2$ in the model, 2 answers "one more unit of $x_1$,
$x_2$ held fixed". Without it, {fmt(alone[1], 1)} answers "one more unit of $x_1$, *and
whatever $x_2$ usually does alongside it*". The lesson's 2,785 €/m² (area alone) against
2,421 (all six features) is this, on the real data.
"""


if __name__ == "__main__":
    run(__file__)
