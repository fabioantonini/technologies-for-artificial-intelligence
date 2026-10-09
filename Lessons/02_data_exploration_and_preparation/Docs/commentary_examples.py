"""Worked examples for the lesson's commentary, Docs/*_commentary.md.

Every number in the examples is computed here and written into the text by this
script, so nothing is transcribed by hand. Run it from the lesson's Docs/
folder; it prints each example's markdown; with --write it splices them into
the commentary under the slide they belong to, and with --check (run by
tools/verify_lesson.py) it fails if the commentary holds anything else.

The examples are deliberately tiny - five to ten values, sums a student can
follow by hand - and each one checks itself against the claim the slide
makes before it is printed.
"""
from __future__ import annotations

import itertools
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from topics_examples import example, fmt, run  # noqa: E402

# --------------------------------------------------------------- slide 11
@example(11)
def mean_imputation():
    # Ten customers: age and monthly charges. Four ages go missing (p = 0.4).
    # The set of four is chosen so that, on this tiny sample, the variance keeps
    # exactly (1 - p) of its value - the formula's average case made exact - and
    # the correlation lands within a hair of sqrt(1 - p).
    age = np.array([24, 28, 31, 35, 38, 42, 45, 49, 52, 56], float)
    charges = np.array([31, 50, 41, 62, 47, 70, 58, 66, 83, 72], float)
    p = 0.4
    best = None
    for missing in itertools.combinations(range(10), 4):
        obs = [i for i in range(10) if i not in missing]
        filled = age.copy()
        filled[list(missing)] = age[obs].mean()
        vr = filled.var() / age.var()
        cr = np.corrcoef(filled, charges)[0, 1] / np.corrcoef(age, charges)[0, 1]
        score = abs(vr - (1 - p)) + abs(cr - np.sqrt(1 - p))
        if best is None or score < best[0]:
            best = (score, missing, filled, vr, cr)
    _, missing, filled, vr, cr = best
    obs_mean = age[[i for i in range(10) if i not in missing]].mean()
    r0 = np.corrcoef(age, charges)[0, 1]
    r1 = np.corrcoef(filled, charges)[0, 1]
    assert abs(vr - 0.6) < 0.03 and abs(cr - np.sqrt(0.6)) < 0.03, (vr, cr)
    gone = ", ".join(str(int(age[i])) for i in missing)
    lost = 1 - cr
    return f"""
### Worked example — four ages out of ten

Ten customers, ages {", ".join(str(int(a)) for a in age)}. Lose four of them —
{gone}, a typical draw of four — and fill each with the mean of the six that
remain, **{fmt(obs_mean, 1)}**. That is $p = 0.4$.

| | before | after imputation | ratio |
|---|---|---|---|
| variance of age | {fmt(age.var(), 1)} | {fmt(filled.var(), 1)} | **{fmt(vr, 2)}** |
| correlation with monthly charges | {fmt(r0, 3)} | {fmt(r1, 3)} | **{fmt(cr, 2)}** |

The formula says $1 - p = 0.60$ for the variance and $\\sqrt{{0.6}} = 0.77$ for the
correlation. The average age barely shifted ({fmt(age.mean(), 1)} before, {fmt(filled.mean(), 1)}
after, and on average over many draws it does not shift at all), yet about
**{lost:.0%}** of the correlation with charges is gone: four points now sit exactly on
the average, where they say nothing about how age and charges move together. On
ten values the ratios depend on which four go missing; the formula is their
average.
"""


# --------------------------------------------------------------- slides 16-17
@example(16)
def z_score_masking():
    x = np.array([40, 45, 50, 55, 60, 65, 70, 500], float)
    n = len(x)
    mean, sd = x.mean(), x.std(ddof=1)
    z = (x - mean) / sd
    bound = (n - 1) / np.sqrt(n)
    assert abs(z).max() < 3 and abs(z.max() - bound) < 0.3
    sd_clean = x[:-1].std(ddof=1)
    return f"""
### Worked example — the ruler made of the outlier

Eight monthly charges: 40, 45, 50, 55, 60, 65, 70 and a data-entry error, **500**.

- mean $\\bar x = {fmt(mean, 1)}$, standard deviation $s = {fmt(sd, 1)}$ — the seven
  ordinary values alone have $s = {fmt(sd_clean, 1)}$;
- $z_{{500}} = (500 - {fmt(mean, 1)}) / {fmt(sd, 1)} = \\mathbf{{{fmt(z[-1], 2)}}}$ — below 3,
  so the rule flags nothing.

This is masking in its purest form, and it is not bad luck: with $n$ values no
z-score can exceed $(n-1)/\\sqrt n$, which for $n = 8$ is **{fmt(bound, 2)}**. On a sample
this small the $3\\sigma$ rule cannot fire whatever the data. The error inflated
the very ruler it was measured with.
"""


@example(17)
def iqr_rule():
    x = np.array([40, 45, 50, 55, 60, 65, 70, 500], float)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    flagged = x[(x < lo) | (x > hi)]
    assert list(flagged) == [500]
    q1c, q3c = np.percentile(x[:-1], [25, 75])
    sd_ratio = x.std(ddof=1) / x[:-1].std(ddof=1)
    return f"""
### Worked example — the same eight values

Quartiles (NumPy's default, linear interpolation): $Q_1 = {fmt(q1, 2)}$,
$Q_3 = {fmt(q3, 2)}$, so $IQR = {fmt(iqr, 2)}$ and the fences are

$$Q_1 - 1.5\\,IQR = {fmt(lo, 2)}, \\qquad Q_3 + 1.5\\,IQR = {fmt(hi, 2)}$$

500 is far outside: **flagged**, where the z-score saw nothing. The reason is
the robustness: drop the 500 and the quartiles of the other seven are
{fmt(q1c, 2)} and {fmt(q3c, 2)} — the error moved them by a few units, while it
multiplied the standard deviation by **{fmt(sd_ratio, 1)}**.
"""


# --------------------------------------------------------------- slide 26
@example(26)
def learning_rate():
    a, b = 100.0, 1.0           # curvature along each axis: a 100:1 ratio
    eta_max = 2 / a
    eta = 0.019
    shrink_b = 1 - eta * b
    steps = int(np.ceil(np.log(0.01) / np.log(shrink_b)))
    assert eta < eta_max
    return f"""
### Worked example — one step size for two directions

Take the simplest ravine, $J(w) = \\tfrac12 (100\\,w_1^2 + 1\\,w_2^2)$: one direction
100 times steeper than the other, the ratio on the slide. A gradient step
multiplies $w_1$ by $(1 - 100\\eta)$ and $w_2$ by $(1 - \\eta)$.

- Stability along the steep axis needs $|1 - 100\\eta| < 1$, so
  $\\eta < 2/100 = \\mathbf{{{fmt(eta_max, 2)}}}$. Above that, $w_1$ overshoots further every step.
- At $\\eta = {eta}$, safely inside, the shallow axis shrinks by only
  ${fmt(shrink_b, 3)}$ per step: **{steps} steps** to fall to 1% of where it started.

Standardise, so both curvatures are 1, and $\\eta = 1$ lands on the minimum in a
**single step**. Nothing about the model changed — only the units of the inputs.
"""


# --------------------------------------------------------------- slide 27
@example(27)
def scalers():
    x = np.array([20, 50, 80, 120, 3300], float)
    mm = (x - x.min()) / (x.max() - x.min())
    z = (x - x.mean()) / x.std()
    assert mm[3] < 0.04
    rows = "\n".join(f"| {int(v)} | {fmt(m, 3)} | {fmt(s, 2)} |" for v, m, s in zip(x, mm, z))
    return f"""
### Worked example — five customers, one outlier

Monthly charges 20, 50, 80, 120 and 3300 (mean {fmt(x.mean(), 0)}, standard deviation
{fmt(x.std(), 0)}):

| charge | MinMax | Standard |
|---|---|---|
{rows}

MinMax packs the four ordinary customers into the bottom **{fmt(mm[3] * 100, 1)}%** of
$[0, 1]$: 20 and 120 end up {fmt(mm[3] - mm[0], 3)} apart. Standard scaling squeezes them
too — all four sit between {fmt(z[0], 2)} and {fmt(z[3], 2)} — but they stay distinct
on a scale where the outlier is "two units away", not "the whole range".
"""


# --------------------------------------------------------------- slides 29-30
# Rates 3/4, 1/4, 0: the first step twice the second, the shape of the real
# churn rates (0.266, 0.137, 0.071: steps 0.129 and 0.066).
CUSTOMERS = [("month-to-month", 1), ("month-to-month", 1), ("month-to-month", 1),
             ("month-to-month", 0), ("one-year", 1), ("one-year", 0),
             ("one-year", 0), ("one-year", 0), ("two-year", 0), ("two-year", 0)]


@example(29)
def encodings():
    cats = ["month-to-month", "one-year", "two-year"]
    rate = {c: F(sum(y for k, y in CUSTOMERS if k == c), sum(1 for k, _ in CUSTOMERS if k == c))
            for c in cats}
    rows = "\n".join(
        f"| {c} | {', '.join(str(y) for k, y in CUSTOMERS if k == c)} | "
        f"{'[' + ', '.join('1' if c == d else '0' for d in cats) + ']'} | {cats.index(c)} | "
        f"{rate[c]} = {fmt(float(rate[c]), 3)} |" for c in cats)
    return f"""
### Worked example — ten customers, three encodings

Churn (1) or stay (0), by contract:

| contract | labels | one-hot | ordinal | target (churn rate) |
|---|---|---|---|---|
{rows}

One-hot and ordinal are computed from the category alone. The target column is
computed **from the labels** — every number in it is an average of $y$. That is
what makes it powerful, and it is the column the second half of the lesson
breaks.
"""


@example(30)
def unequal_steps():
    cats = ["month-to-month", "one-year", "two-year"]
    rate = [F(sum(y for k, y in CUSTOMERS if k == c), sum(1 for k, _ in CUSTOMERS if k == c))
            for c in cats]
    d1, d2 = rate[0] - rate[1], rate[1] - rate[2]
    assert d1 == 2 * d2
    return f"""
### Worked example — the steps ordinal encoding assumes equal

On the ten customers of the previous example the churn rates are
{rate[0]}, {rate[1]}, {rate[2]}. The two steps:

- month-to-month → one-year: ${rate[0]} - {rate[1]} = {d1}$ ({fmt(float(d1), 3)})
- one-year → two-year: ${rate[1]} - {rate[2]} = {d2}$ ({fmt(float(d2), 3)})

Ordinal encoding gives both steps the same length, 1, so a linear model must
predict the same change for both. The data asks for one step twice the other —
the same shape as the real rates on the slide, 0.129 and 0.066. One-hot gives
each contract its own coefficient and asks nothing.
"""


# --------------------------------------------------------------- slide 33
@example(33)
def rank_deficient():
    cats = ["month-to-month", "one-year", "two-year", "one-year"]
    names = ["month-to-month", "one-year", "two-year"]
    X = np.array([[1] + [1 if c == d else 0 for d in names] for c in cats])
    rank = np.linalg.matrix_rank(X)
    Xd = X[:, [0, 2, 3]]
    assert rank == 3 and np.linalg.matrix_rank(Xd) == 3
    rows = "\n".join(f"| {c} | " + " | ".join(str(v) for v in r) + " |" for c, r in zip(cats, X))
    return f"""
### Worked example — four rows, four columns, rank three

Four customers, an intercept and all three dummies:

| customer | intercept | $d_1$ | $d_2$ | $d_3$ |
|---|---|---|---|---|
{rows}

In every row $d_1 + d_2 + d_3 = 1$ = the intercept, so the four columns are not
independent: rank **{rank}**, not 4. Concretely, adding 5 to the intercept and
subtracting 5 from all three dummy coefficients changes no prediction at all —
infinitely many coefficient vectors fit equally well. Drop $d_1$ and the three
remaining columns have rank 3: unique again, with month-to-month as the
reference the other coefficients are measured from.
"""


# --------------------------------------------------------------- slide 44
@example(44)
def knn_leak():
    # tenure for six customers; one training row has age missing; k = 2.
    rows = [("A", "train", 10, 34), ("B", "train", 30, 52), ("C", "train", 41, None),
            ("D", "train", 70, 61), ("E", "test", 42, 45), ("F", "test", 39, 47)]
    c_tenure = 41

    def donors(pool):
        cands = [(abs(t - c_tenure), name, age) for name, split, t, age in pool
                 if age is not None]
        return sorted(cands)[:2]

    leaky = donors(rows)
    honest = donors([r for r in rows if r[1] == "train"])
    leaky_age = np.mean([a for _, _, a in leaky])
    honest_age = np.mean([a for _, _, a in honest])
    assert {n for _, n, _ in leaky} == {"E", "F"}
    table = "\n".join(f"| {n} | {s} | {t} | {a if a is not None else '**missing**'} |"
                      for n, s, t, a in rows)
    return f"""
### Worked example — the two nearest neighbours are both in the test set

Six customers; C, in training, has no age. `KNNImputer` with $k = 2$ looks for the
two customers closest in tenure:

| customer | split | tenure (months) | age |
|---|---|---|---|
{table}

- Fitted on **all six**, C's nearest are {leaky[0][1]} and {leaky[1][1]} (tenure 42 and 39):
  C gets age **{fmt(leaky_age, 1)}** — built entirely from **test** rows.
- Fitted on the **training rows only**, C's nearest are {honest[0][1]} and
  {honest[1][1]}: age **{fmt(honest_age, 1)}**.

No test label was touched, and no error was raised. The test customers' features
went into a training row, and the model will be graded on those same customers.
"""


# --------------------------------------------------------------- slides 49-51
ZIP = [("A", 1), ("A", 1), ("B", 0), ("B", 0), ("C", 1), ("C", 0), ("D", 0), ("D", 1)]


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


@example(49)
def target_leak():
    labels = [y for _, y in ZIP]
    groups = {}
    for z, y in ZIP:
        groups.setdefault(z, []).append(y)
    leaky = [F(sum(groups[z]), len(groups[z])) for z, _ in ZIP]
    a = auc([float(v) for v in leaky], labels)
    # the derivation's identity on a group of four
    ys = [1, 0, 0, 1]
    full = F(sum(ys), 4)
    loo = F(sum(ys) - 1, 3)
    assert full - loo == (1 - loo) / 4
    assert a > 0.8
    rows = "\n".join(f"| {z} | {y} | {v} |" for (z, y), v in zip(ZIP, leaky))
    return f"""
### Worked example — a zip code with no signal, and an AUC of {fmt(a, 3)}

Eight customers in four zip codes, two per code, labels from a coin flip — the
code is irrelevant by construction:

| zip | churned | target-encoded (all rows) |
|---|---|---|
{rows}

Score each customer by that column and compute the AUC against their own labels:
**{fmt(a, 3)}**. A column that carries nothing ranks churners above stayers, because
with two customers per code each customer's own label is half of their own
feature.

The slide's identity on one group of four, labels 1, 0, 0, 1, for a churner:
with its own label the group mean is ${full}$; without it, ${loo}$; the
difference is ${full - loo}$ — and indeed $(y_i - \\bar y^{{(-i)}})/n_c =
(1 - {loo})/4 = {(1 - loo) / 4}$.
"""


@example(51)
def cross_fit():
    labels = [y for _, y in ZIP]
    # two folds: rows 0,2,4,6 and rows 1,3,5,7 (one of each pair per fold)
    folds = [[0, 2, 4, 6], [1, 3, 5, 7]]
    enc = [None] * len(ZIP)
    for k, fold in enumerate(folds):
        other = folds[1 - k]
        for i in fold:
            z = ZIP[i][0]
            same = [ZIP[j][1] for j in other if ZIP[j][0] == z]
            enc[i] = F(sum(same), len(same))
    a = auc([float(v) for v in enc], labels)
    leaky_auc = auc([float(F(sum(y for zz, y in ZIP if zz == z), sum(1 for zz, _ in ZIP if zz == z)))
                     for z, _ in ZIP], labels)
    assert a <= 0.5
    rows = "\n".join(f"| {z} | {y} | {i % 2 + 1} | {v} |" for i, ((z, y), v) in enumerate(zip(ZIP, enc)))
    return f"""
### Worked example — the same eight customers, cross-fitted

Two folds, one customer of each pair in each. Each customer's code is encoded
using **only the other fold** — so only the other member of its pair:

| zip | churned | fold | encoded from the other fold |
|---|---|---|---|
{rows}

AUC against the labels: **{fmt(a, 3)}**. With no shared label, the column is worth
nothing — as it should be — where the leaky version scored {fmt(leaky_auc, 3)}. On groups
this small the honest score can even fall below 0.5: each customer is described by
a neighbour that disagrees with it. That is noise, not signal, and it averages out
on real group sizes.
"""



# --------------------------------------------------------------- slide 39
@example(39)
def auc_by_pairs():
    # Two churners and three non-churners, with the model's churn scores.
    churners = [0.7, 0.4]
    stayers = [0.6, 0.3, 0.2]
    pairs = [(c, s) for c in churners for s in stayers]
    right = sum(c > s for c, s in pairs)
    auc = F(right, len(pairs))
    # The same number by the library's route, from labels and scores.
    from sklearn.metrics import roc_auc_score
    labels = [1] * len(churners) + [0] * len(stayers)
    assert abs(roc_auc_score(labels, churners + stayers) - float(auc)) < 1e-12
    # A model that gives everyone the same score ranks nobody: ties count half.
    assert roc_auc_score(labels, [0.5] * len(labels)) == 0.5
    rows = "\n".join(f"| churner, {c} | " + " | ".join("yes" if c > s else "no" for s in stayers)
                     + f" | {sum(c > s for s in stayers)} |" for c in churners)
    return f"""
### Worked example — the AUC by counting pairs

Two churners and three customers who stayed, with the score the model gives each
of them. Every pair of one churner and one stayer is a small test: did the model
score the churner higher? There are {len(churners)} × {len(stayers)} = {len(pairs)}
pairs.

| churner's score | above stayer 0.6? | above 0.3? | above 0.2? | pairs right |
|---|---|---|---|---|
{rows}

{right} of {len(pairs)} pairs are ordered correctly, so the AUC is {right}/{len(pairs)}
= **{fmt(float(auc), 2)}**. scikit-learn's `roc_auc_score`, given the five labels and
the five scores, returns the same number.

Two readings follow. A model that ordered the customers at random would get about
half the pairs right: 0.5. And the majority baseline, which answers "stays" for
everyone, gives every customer the same score, so it orders no pair at all: its
AUC is 0.5, although its accuracy on the churn data is 0.806. That is why the AUC
shows signal that accuracy hides.
"""


if __name__ == "__main__":
    run(__file__)
