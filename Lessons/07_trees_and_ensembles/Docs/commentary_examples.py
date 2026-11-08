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
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from topics_examples import example, fmt, run  # noqa: E402


def gini(counts):
    total = sum(counts)
    return 1 - sum((c / total) ** 2 for c in counts)


# --------------------------------------------------------------- slide 10
@example(10)
def gini_by_hand():
    groups = [("A", 8, 0), ("B", 6, 2), ("C", 4, 4)]
    rows = []
    values = {}
    for name, d, r in groups:
        p = d / (d + r)
        g = gini([d, r])
        assert abs(g - 2 * p * (1 - p)) < 1e-12        # slide 11's shortcut agrees
        values[name] = g
        rows.append(f"| {name} | {d} | {r} | {d} ÷ {d + r} = {fmt(p, 2)} | "
                    f"1 − ({fmt(p, 2)}² + {fmt(1 - p, 2)}²) = {fmt(g, 3)} |")
    three = gini([3, 3, 3])
    assert values["A"] == 0 and values["C"] == 0.5 and abs(three - (1 - 1 / 3)) < 1e-12
    table = "\n".join(rows)
    return f"""
### Worked example — the Gini impurity of three groups of eight

| group | defaulted | repaid | $p$ = defaulted ÷ size | $G = 1 - (p^2 + (1-p)^2)$ |
|---|---|---|---|---|
{table}

A pure group scores 0, an even one 0.5, the two-class maximum. Each line also equals
$2p(1-p)$, slide 11's shortcut. With three classes the ceiling moves: a group of nine
with three of each class scores $1 - 3 \\times (1/3)^2 = {fmt(three, 3)}$.
"""


# --------------------------------------------------------------- slide 13
@example(13)
def gain_by_hand():
    parent = (4, 6)                    # 4 defaulted, 6 repaid among 10
    left, right = (3, 1), (1, 5)       # 4 loans go left, 6 go right
    gp, gl, gr = gini(parent), gini(left), gini(right)
    wl, wr = sum(left) / 10, sum(right) / 10
    avg = wl * gl + wr * gr
    gain = gp - avg
    assert 0 < gain < gp and abs(sum(left) + sum(right) - 10) < 1e-12
    wrong = 1 - avg
    assert wrong > gp                  # the "1 minus" slip gives an impossible gain
    return f"""
### Worked example — the gain of one split of ten loans

Ten loans, 4 defaulted. A threshold sends 4 loans left (3 defaulted) and 6 right
(1 defaulted).

| group | loans | defaulted | $G = 2p(1-p)$ | weight = loans ÷ 10 |
|---|---|---|---|---|
| parent | 10 | 4 | 2 × 0.40 × 0.60 = {fmt(gp, 3)} | |
| left | 4 | 3 | 2 × 0.75 × 0.25 = {fmt(gl, 3)} | {fmt(wl, 1)} |
| right | 6 | 1 | 2 × {fmt(1 / 6, 3)} × {fmt(5 / 6, 3)} = {fmt(gr, 3)} | {fmt(wr, 1)} |

Weighted average of the children: {fmt(wl, 1)} × {fmt(gl, 3)} + {fmt(wr, 1)} × {fmt(gr, 3)} =
{fmt(avg, 3)}. Gain: $\\Delta G$ = {fmt(gp, 3)} − {fmt(avg, 3)} = **{fmt(gain, 3)}**. The slip of
subtracting from 1 would give {fmt(wrong, 3)}, larger than the parent's whole impurity,
{fmt(gp, 3)}: impossible, which is how to catch it.
"""


# --------------------------------------------------------------- slide 25
@example(25)
def leaf_budget():
    m = 1200
    leaves = [("A", 900, 0.020), ("B", 30, 0.300)]
    rows, weighted = [], {}
    for name, size, gain in leaves:
        w = size / m
        weighted[name] = gain * w
        rows.append(f"| {name} | {size} | {fmt(gain, 3)} | {size} ÷ {m:,} = {fmt(w, 3)} | "
                    f"{fmt(gain, 3)} × {fmt(w, 3)} = {fmt(gain * w, 4)} |")
    assert leaves[1][2] > leaves[0][2] and weighted["A"] > weighted["B"]
    table = "\n".join(rows)
    return f"""
### Worked example — which leaf a budget of leaves splits next

Two leaves of a tree trained on {m:,} applicants could be split next. Leaf B's best
split has the larger gain inside its own group; leaf A's group is much larger.

| leaf | applicants | best $\\Delta G$ in the leaf | share of all applicants | weighted gain |
|---|---|---|---|---|
{table}

`max_leaf_nodes` spends the next leaf on **A**: its gain is smaller, but it improves the
purity of {fmt(900 / m * 100, 0)}% of the applicants, against {fmt(30 / m * 100, 1)}% for B.
Judged by the local gain alone, B would win; that is the slip the slide warns about.
"""


# --------------------------------------------------------------- slide 30
@example(30)
def vote_by_hand():
    truth = {"P": "D", "Q": "R", "S": "D"}
    # five trees' verdicts; S is a flipped label: every tree says "R" there
    votes = {"P": "DDRDD", "Q": "RDRRR", "S": "RRRRR"}
    rows, results = [], {}
    for a in "PQS":
        v = votes[a]
        maj = "D" if v.count("D") > v.count("R") else "R"
        wrong = sum(1 for x in v if x != truth[a])
        results[a] = (maj == truth[a], wrong)
        rows.append(f"| {a} | {' '.join(v)} | {v.count('D')} D, {v.count('R')} R | {maj} | "
                    f"{truth[a]} | {'right' if maj == truth[a] else 'wrong'} |")
    assert results["P"][0] and results["Q"][0] and not results["S"][0]
    per_tree = sum(results[a][1] for a in "PQS") / 5
    table = "\n".join(rows)
    return f"""
### Worked example — five trees vote on three applicants

| applicant | trees 1 to 5 | count | vote | label | |
|---|---|---|---|---|---|
{table}

On P and Q one tree each is wrong, a different tree each time: they are outvoted. On S
every tree is wrong the same way, because S's label was flipped and its neighbours follow
the rule; there is no minority to outvote. Each tree alone is wrong on {fmt(per_tree, 1)}
of the three applicants on average; the vote is wrong on 1. The vote removes the
mistakes that differ, not the one they share.
"""


# --------------------------------------------------------------- slide 32
@example(32)
def bootstrap_by_hand():
    m = 5
    draws = [2, 4, 2, 1, 2]                       # one bootstrap sample of rows 1-5
    counts = {r: draws.count(r) for r in range(1, m + 1)}
    distinct = sum(1 for c in counts.values() if c > 0)
    never = [r for r, c in counts.items() if c == 0]
    repeats = m - distinct
    assert repeats == len(never)                  # every repeat replaces a row left out
    miss = (1 - 1 / m) ** m
    return f"""
### Worked example — one bootstrap sample of five rows

Five training rows, numbered 1 to 5; five draws with replacement: {", ".join(map(str, draws))}.

- Times each row is drawn: {", ".join(f"row {r}: {c}" for r, c in counts.items())}.
- Different rows drawn: {distinct}. Never drawn, the out-of-bag rows of this tree:
  rows {" and ".join(map(str, never))}.
- Repeated draws: {m} − {distinct} = {repeats}, exactly as many as the rows never drawn:
  each repeat took the place of a row left out.

The theory for $m = 5$: a row is missed by all five draws with probability
$(1 - 1/5)^5 = {fmt(miss, 3)}$, so on average {fmt(m * miss, 2)} rows of 5 are out of bag.
For large $m$ the share tends to 0.368.
"""


# --------------------------------------------------------------- slide 51
@example(51)
def boosting_by_hand():
    y = [1.0, 1.0, 3.0, 3.0]
    alpha = 0.5
    F = [sum(y) / len(y)] * 4
    start = F[0]
    rows = []
    for t in range(1, 4):
        res = [yi - fi for yi, fi in zip(y, F)]
        # a depth-1 tree on x = 1, 2 | 3, 4 predicts each half's mean residual
        left, right = sum(res[:2]) / 2, sum(res[2:]) / 2
        h = [left, left, right, right]
        F = [fi + alpha * hi for fi, hi in zip(F, h)]
        rows.append(f"| {t} | {fmt(res[0], 3)}, {fmt(res[2], 3)} | {fmt(left, 3)}, {fmt(right, 3)} | "
                    f"{fmt(F[0], 3)}, {fmt(F[2], 3)} |")
    assert abs(F[0] - (1 + (start - 1) * (1 - alpha) ** 3)) < 1e-12
    table = "\n".join(rows)
    return f"""
### Worked example — three boosting steps on four points

Four points: $x$ = 1, 2, 3, 4 with $y$ = 1, 1, 3, 3. Learning rate $\\alpha = {alpha}$; each
tree has one split, between $x = 2$ and $x = 3$, and predicts the mean residual on each
side. Start: the mean, $F_0 = {fmt(start, 0)}$ for every point.

| tree | residuals left, right | tree's output left, right | new prediction left, right |
|---|---|---|---|
{table}

Each tree is trained on what the sum so far still gets wrong, and each step covers half
of the remaining distance, because $\\alpha = {alpha}$: the residual goes 1, 0.5, 0.25. The
prediction approaches 1 and 3 without ever jumping there in one step.
"""


# --------------------------------------------------------------- slide 56
@example(56)
def classification_start():
    defaulted, total = 3, 10
    p = defaulted / total
    F0 = math.log(p / (1 - p))
    back = 1 / (1 + math.exp(-F0))
    assert abs(back - p) < 1e-12
    return f"""
### Worked example — where classification boosting starts

Ten loans, {defaulted} defaulted. The start is the same for every loan:

- $p = {defaulted}/{total} = {fmt(p, 1)}$, the share who defaulted;
- the score $F_0 = \\log\\big(p/(1-p)\\big) = \\log({fmt(p, 1)}/{fmt(1 - p, 1)}) = {fmt(F0, 3)}$, and the
  logistic function turns it back into $1/(1 + e^{{{fmt(-F0, 3)}}}) = {fmt(back, 1)}$;
- the first tree is trained on $y - p$: $1 - {fmt(p, 1)} = +{fmt(1 - p, 1)}$ for each
  defaulter and $0 - {fmt(p, 1)} = -{fmt(p, 1)}$ for each loan repaid.

The tree's output moves the score $F$, not $p$; the next pseudo-residuals are computed
from the new $p$, so they change from round to round.
"""


# --------------------------------------------------------------- slide 65
@example(65)
def menu_by_hand():
    real, noise, offered = 2, 6, 2
    cols = list(range(real + noise))           # 0 and 1 are the real columns
    menus = list(combinations(cols, offered))
    no_real = [c for c in menus if not set(c) & {0, 1}]
    p = len(no_real) / len(menus)
    assert abs(p - math.comb(noise, offered) / math.comb(real + noise, offered)) < 1e-12
    return f"""
### Worked example — how often the menu holds no real column

A smaller version of slide 65: {real} real columns and {noise} random ones, {real + noise} in all;
each split is offered {offered} of them at random.

- Possible menus: $\\binom{{{real + noise}}}{{{offered}}} = {len(menus)}$.
- Menus with no real column, both picked among the {noise} random ones:
  $\\binom{{{noise}}}{{{offered}}} = {len(no_real)}$.
- Share: {len(no_real)} ÷ {len(menus)} = **{fmt(p, 3)}**: more than half of the splits must pick a
  random column.

The lesson's case is the same count with 20 random columns out of 22 and 4 offered:
4,845 ÷ 7,315 = 0.662.
"""


if __name__ == "__main__":
    run(__file__)
