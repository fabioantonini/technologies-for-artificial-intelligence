"""Worked examples for the lesson's commentary, Docs/*_commentary.md.

Every number in the examples is computed here and written into the text by this
script, so nothing is transcribed by hand. Run it from the lesson's Docs/
folder; it prints each example's markdown; with --write it splices them into
the commentary under the slide they belong to, and with --check (run by
tools/verify_lesson.py) it fails if the commentary holds anything else.

The values are invented and deliberately small - sums a student can follow by
hand - and each example checks itself against the claim the slide makes before
it is printed. Where a second route to the same number exists, it is taken too.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from topics_examples import example, fmt, run  # noqa: E402


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


# --------------------------------------------------------------- slide 9
@example(9)
def odds_and_log_odds():
    rows, values = [], {}
    for p in (0.05, 0.2, 0.5, 0.8, 0.95):
        odds = p / (1 - p)
        logit = math.log(odds)
        values[p] = logit
        assert abs(sigmoid(logit) - p) < 1e-12          # the sigmoid undoes it
        rows.append(f"| {p} | {fmt(p, 2)} ÷ {fmt(1 - p, 2)} = {fmt(odds, 3)} "
                    f"| {fmt(logit, 2)} |")
    assert abs(values[0.2] + values[0.8]) < 1e-12        # symmetric around 0.5
    table = "\n".join(rows)
    return f"""
### Worked example — five probabilities, their odds and their log-odds

| probability $p$ | odds $p \\div (1 - p)$ | log-odds $\\log$(odds) |
|---|---|---|
{table}

Read the columns from left to right. The probability is squeezed into $[0, 1]$.
The odds have no ceiling (0.95 already gives {fmt(0.95 / 0.05, 0)}) but still stop at 0.
The log-odds have neither limit, and they are symmetric: 0.2 and 0.8 sit at
{fmt(values[0.2], 2)} and {fmt(values[0.8], 2)}, the same distance either side of
0, which is where $p = 0.5$ lands.
"""


# --------------------------------------------------------------- slide 14
@example(14)
def odds_not_probability():
    w = 0.7
    factor = math.exp(w)
    rows = []
    after = {}
    for p in (0.01, 0.5, 0.9):
        odds = p / (1 - p)
        new_odds = odds * factor
        new_p = new_odds / (1 + new_odds)
        after[p] = new_p
        rows.append(f"| {p} | {fmt(odds, 3)} | {fmt(new_odds, 3)} | {fmt(new_p, 3)} "
                    f"| {fmt(new_p / p, 2)} |")
    assert after[0.01] / 0.01 > 1.9 and after[0.9] / 0.9 < 1.1
    table = "\n".join(rows)
    return f"""
### Worked example — one coefficient, three drives

Take a coefficient $w = {w}$. One unit more of its feature adds {w} to the log-odds,
so it multiplies the odds by $e^{{{w}}} = {fmt(factor, 2)}$: roughly doubles them.

| probability before | odds before | odds after, $\\times {fmt(factor, 2)}$ | probability after | probability after ÷ before |
|---|---|---|---|---|
{table}

The odds double for all three drives. The probability nearly doubles only for the
rare event; at 0.5 it rises by a third, and at 0.9 it moves five points. "Twice the
odds" and "twice as likely" agree only when the event is rare, which is why the
habit of confusing them survives on this fleet and fails elsewhere.
"""


# --------------------------------------------------------------- slide 24
@example(24)
def log_loss_four_drives():
    drives = [("A", 1, 0.9), ("B", 0, 0.2), ("C", 0, 0.1), ("D", 1, 0.05)]
    rows, losses = [], []
    for name, y, p in drives:
        happened = p if y == 1 else 1 - p
        loss = -(y * math.log(p) + (1 - y) * math.log(1 - p))
        assert abs(loss + math.log(happened)) < 1e-12   # the bracket is an if
        losses.append(loss)
        rows.append(f"| {name} | {y} | {p} | {fmt(happened, 2)} | {fmt(loss, 3)} |")
    cost = sum(losses) / len(losses)
    assert max(losses) == losses[3] and losses[3] > sum(losses[:3])
    table = "\n".join(rows)
    return f"""
### Worked example — four drives, one confident mistake

| drive | $y$ | $p$, the P(fail) | probability of what happened | loss $-\\log$ of it |
|---|---|---|---|---|
{table}

The cost is the average: ({' + '.join(fmt(v, 3) for v in losses)}) ÷ 4 =
**{fmt(cost, 3)}**. Drive D failed after the model gave it 0.05, and its loss alone,
{fmt(losses[3], 3)}, is larger than the other three together. Log loss is dominated by
confident mistakes, which is exactly where a fitting procedure should be pushed.
"""


# --------------------------------------------------------------- slide 25
@example(25)
def one_gradient_step():
    xs, ys = [2.0, -1.0, 0.5], [1, 0, 0]
    w = b = 0.0
    ps = [sigmoid(w * x + b) for x in xs]
    errors = [p - y for p, y in zip(ps, ys)]
    grad_w = sum(e * x for e, x in zip(errors, xs)) / 3
    grad_b = sum(errors) / 3
    # the same derivative, numerically, from the cost itself
    def cost(w_, b_):
        return -sum(y * math.log(sigmoid(w_ * x + b_)) + (1 - y) * math.log(1 - sigmoid(w_ * x + b_))
                    for x, y in zip(xs, ys)) / 3
    h = 1e-6
    assert abs((cost(h, 0) - cost(-h, 0)) / (2 * h) - grad_w) < 1e-8
    alpha = 0.5
    new_w, new_b = w - alpha * grad_w, b - alpha * grad_b
    rows = "\n".join(f"| {i + 1} | {x} | {y} | {fmt(p, 1)} | {fmt(e, 1)} | {fmt(e * x, 2)} |"
                     for i, (x, y, p, e) in enumerate(zip(xs, ys, ps, errors)))
    return f"""
### Worked example — one step of gradient descent on three drives

One standardised feature $x$, start at $w = 0$, $b = 0$, so every drive gets
$p = \\sigma(0) = 0.5$:

| drive | $x$ | $y$ | $p$ | error $p - y$ | error × $x$ |
|---|---|---|---|---|---|
{rows}

$\\partial J / \\partial w$ = average of the last column = **{fmt(grad_w, 4)}**;
$\\partial J / \\partial b$ = average of the errors = **{fmt(grad_b, 4)}**. With a
learning rate $\\alpha = {alpha}$ the step is $w = 0 - {alpha} \\times ({fmt(grad_w, 4)}) =
{fmt(new_w, 4)}$ and $b = 0 - {alpha} \\times {fmt(grad_b, 4)} = {fmt(new_b, 4)}$.

The failed drive, with a large positive $x$, pulls $w$ up. So does the healthy drive
at $x = -1$: a larger $w$ lowers its $p$, which is what a healthy drive needs. Only the
healthy drive at $x = 0.5$ pulls $w$ down, and weakly, because its $x$ is small. The
intercept goes down because two of the three drives are healthy. That is lesson 3's
weighted vote, with $p$ in place of the straight line.
"""


# --------------------------------------------------------------- slide 36
@example(36)
def twenty_drive_confusion():
    tp, fn, fp, tn = 3, 1, 2, 14
    m = tp + fn + fp + tn
    accuracy = (tp + tn) / m
    baseline = (fp + tn) / m
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    assert abs(f1 - 2 * tp / (2 * tp + fp + fn)) < 1e-12
    specificity = tn / (tn + fp)
    return f"""
### Worked example — twenty drives, four of which fail

A model flags 5 of 20 test drives; 3 of the 5 really fail, and 1 failure is missed:
TP = {tp}, FP = {fp}, FN = {fn}, TN = {tn}.

| metric | formula | value |
|---|---|---|
| accuracy | (TP + TN) ÷ {m} = ({tp} + {tn}) ÷ {m} | {fmt(accuracy, 2)} |
| always "healthy" | (FP + TN) ÷ {m} = {fp + tn} ÷ {m} | {fmt(baseline, 2)} |
| precision | TP ÷ (TP + FP) = {tp} ÷ {tp + fp} | {fmt(precision, 2)} |
| recall | TP ÷ (TP + FN) = {tp} ÷ {tp + fn} | {fmt(recall, 2)} |
| F1 | 2TP ÷ (2TP + FP + FN) = {2 * tp} ÷ {2 * tp + fp + fn} | {fmt(f1, 3)} |
| specificity | TN ÷ (TN + FP) = {tn} ÷ {tn + fp} | {fmt(specificity, 3)} |
| false positive rate | 1 − specificity | {fmt(1 - specificity, 3)} |

The model beats the do-nothing baseline by {fmt((accuracy - baseline) * 100, 0)} points of accuracy, and the
other rows say what those points are made of: three failures caught out of four, at
the price of two needless replacements. F1 from the counts and from $2PR / (P + R)$ give
the same {fmt(f1, 3)}.
"""


# --------------------------------------------------------------- slide 45
@example(45)
def auc_by_pairs():
    failing = [0.9, 0.6, 0.3]
    healthy = [0.7, 0.3, 0.2, 0.1]
    wins = []
    for f in failing:
        score = sum(1.0 if f > h else 0.5 if f == h else 0.0 for h in healthy)
        wins.append(score)
    auc = sum(wins) / (len(failing) * len(healthy))
    # second route: the area under the ROC curve, by the trapezoid rule
    thresholds = sorted(set(failing + healthy), reverse=True)
    points = [(0.0, 0.0)]
    for t in thresholds:
        tpr = sum(f >= t for f in failing) / len(failing)
        fpr = sum(h >= t for h in healthy) / len(healthy)
        points.append((fpr, tpr))
    area = sum((x1 - x0) * (y0 + y1) / 2 for (x0, y0), (x1, y1) in zip(points, points[1:]))
    assert abs(area - auc) < 1e-12
    rows = "\n".join(f"| {f} | {fmt(w, 1)} of 4 |" for f, w in zip(failing, wins))
    return f"""
### Worked example — the AUC of seven drives, by counting pairs

Three failing drives scored 0.9, 0.6, 0.3; four healthy ones scored 0.7, 0.3, 0.2,
0.1. There are 3 × 4 = 12 (failing, healthy) pairs. For each failing drive, count the
healthy drives it outscores, a tie counting half:

| failing drive's score | healthy drives below it |
|---|---|
{rows}

AUC = ({' + '.join(fmt(w, 1) for w in wins)}) ÷ 12 = **{fmt(auc, 3)}**. Drawing the ROC
curve threshold by threshold and measuring its area gives the same {fmt(auc, 3)}: the
probability reading and the geometric one are one number. The drive at 0.6 loses one
pair, to the healthy drive at 0.7; the drive at 0.3 loses one and ties one.
"""


# --------------------------------------------------------------- slide 50
@example(50)
def average_precision_by_steps():
    ranking = "FHFHHFH"            # riskiest first: F failed, H healthy
    positives = ranking.count("F")
    caught, steps, ap = 0, [], 0.0
    for k, label in enumerate(ranking, start=1):
        if label == "F":
            caught += 1
            precision = caught / k
            recall_step = 1 / positives
            ap += recall_step * precision
            steps.append((k, caught, precision))
    baseline = positives / len(ranking)
    assert ap > baseline
    rows = "\n".join(f"| {k} | {c} of {k} = {fmt(p, 3)} | {c} of {positives} | 1/{positives} |"
                     for k, c, p in steps)
    terms = " + ".join(f"\\tfrac{{1}}{{{positives}}} \\times {fmt(p, 3)}" for _, _, p in steps)
    return f"""
### Worked example — average precision of a seven-drive ranking

Seven drives sorted by score, riskiest first: failed, healthy, failed, healthy,
healthy, failed, healthy. Recall rises only when the cut passes a failed drive, so
only those cuts add a step:

| cut after drive | precision there | recall there | recall step |
|---|---|---|---|
{rows}

$\\text{{AP}} = {terms} = {fmt(ap, 3)}$. A random ranking would have precision
equal to the positive rate at every cut, {positives} ÷ {len(ranking)} = {fmt(baseline, 3)},
and so an AP of {fmt(baseline, 3)}: that, not 0.5, is the line this ranking has to beat.
"""


# --------------------------------------------------------------- slide 53
@example(53)
def threshold_from_costs():
    c_fp, c_fn = 50, 200
    t_star = c_fp / (c_fp + c_fn)
    rows = []
    for p in (0.15, 0.3):
        flag = (1 - p) * c_fp
        leave = p * c_fn
        decision = "flag" if flag < leave else "leave"
        assert (decision == "flag") == (p > t_star)
        rows.append(f"| {p} | (1 − {p}) × {c_fp} = {fmt(flag, 1)} | {p} × {c_fn} = "
                    f"{fmt(leave, 1)} | **{decision}** |")
    assert c_fp / (c_fp + c_fp) == 0.5
    table = "\n".join(rows)
    return f"""
### Worked example — a false alarm at {c_fp}, a miss at {c_fn}

$t^* = {c_fp} \\div ({c_fp} + {c_fn}) = {fmt(t_star, 2)}$. Check it on two drives,
comparing the two expected costs directly:

| drive's $p$ | expected cost of flagging | expected cost of leaving | decision |
|---|---|---|---|
{table}

The drive at 0.15 is below $t^*$ and is left alone; the one at 0.3 is above and is
flagged, and the direct comparison agrees both times. Make the miss cost {c_fp} as well
and $t^* = {c_fp} \\div ({c_fp} + {c_fp}) = 0.5$, the default.
"""


# --------------------------------------------------------------- slide 55
@example(55)
def class_weights_as_threshold():
    k = 4
    shift = math.log(k)
    threshold = 1 / (1 + k)
    p_plain = threshold
    odds = p_plain / (1 - p_plain)
    p_weighted = odds * k / (1 + odds * k)
    assert abs(p_weighted - 0.5) < 1e-12
    return f"""
### Worked example — a weight ratio of {k}

Give every failure {k} times the weight of a healthy drive. For a correctly specified
logistic model the main effect is on the intercept, which rises by $\\log {k} =
{fmt(shift, 3)}$: every odds is multiplied by {k}.

A drive the plain model scores $p = {fmt(p_plain, 2)}$ has odds ${fmt(p_plain, 2)} \\div
{fmt(1 - p_plain, 2)} = {fmt(odds, 2)}$. Multiplied by {k}, the odds become 1, a probability of
0.5. So the weighted model's threshold of 0.5 falls where the plain model's
$1 \\div (1 + {k}) = {fmt(threshold, 2)}$ was: the same decisions, reached by moving the
line at training time instead of after it. With a misspecified model, little data
or a strong penalty, the weights move the slopes too, and the match is only
approximate.
"""


# --------------------------------------------------------------- slide 58
@example(58)
def softmax_three_scores():
    scores = [2.0, 1.0, 0.1]
    exps = [math.exp(z) for z in scores]
    total = sum(exps)
    probs = [e / total for e in exps]
    assert abs(sum(probs) - 1) < 1e-12
    # two classes: softmax of (0, z) is the sigmoid of z
    z = 1.5
    two = math.exp(z) / (math.exp(0) + math.exp(z))
    assert abs(two - sigmoid(z)) < 1e-12
    rows = "\n".join(f"| {name} | {s} | {fmt(e, 3)} | {fmt(e, 3)} ÷ {fmt(total, 3)} = {fmt(p, 3)} |"
                     for name, s, e, p in zip(("healthy", "degraded", "failed"), scores, exps, probs))
    return f"""
### Worked example — softmax on three scores

| class | score $z_k$ | $e^{{z_k}}$ | probability |
|---|---|---|---|
{rows}

The three probabilities add up to 1, and the largest score keeps the largest share:
the drive is called healthy with {fmt(probs[0], 3)}. With two classes and scores 0 and
{z}, softmax gives $e^{{{z}}} \\div (1 + e^{{{z}}}) = {fmt(two, 3)}$, which is
$\\sigma({z})$: the sigmoid is softmax with one score fixed at 0.
"""


if __name__ == "__main__":
    run(__file__)
