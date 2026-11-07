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
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from topics_examples import example, fmt, run  # noqa: E402


def auc_by_pairs(failing, healthy):
    """The share of failing-healthy pairs ranked the right way round."""
    right = sum(1 for f, h in product(failing, healthy) if f > h)
    return right, len(failing) * len(healthy)


# --------------------------------------------------------------- slide 8
@example(8)
def auc_from_pairs():
    failing = [0.90, 0.70]
    healthy = [0.80, 0.60, 0.40, 0.30, 0.20]
    right, pairs = auc_by_pairs(failing, healthy)
    # One failing drive slips below two more healthy ones.
    slipped = [0.90, 0.50]
    right_after, _ = auc_by_pairs(slipped, healthy)
    assert pairs == 10 and right - right_after == 1
    rows = []
    for f in failing:
        beaten = [h for h in healthy if f > h]
        rows.append(f"| {fmt(f, 2)} | {', '.join(fmt(h, 2) for h in beaten)} | {len(beaten)} of {len(healthy)} |")
    table = "\n".join(rows)
    return f"""
### Worked example — the AUC as a count of pairs

A test set of seven drives, two failing and five healthy, with the model's predicted
probabilities. Healthy: {', '.join(fmt(h, 2) for h in healthy)}. The pairs are
$2 \\times 5 = {pairs}$, and a pair is ranked right when the failing drive scores higher.

| failing drive's probability | healthy drives it outscores | pairs ranked right |
|---|---|---|
{table}

AUC = pairs ranked right ÷ pairs = {right} ÷ {pairs} = {fmt(right / pairs, 2)}. Now let the second
failing drive score {fmt(slipped[1], 2)} instead of {fmt(failing[1], 2)}: it drops below one more
healthy drive, {right - right_after} pair turns wrong, and the AUC becomes {right_after} ÷ {pairs} =
{fmt(right_after / pairs, 2)}. With so few pairs, one drive moving a little moves the AUC by a tenth:
slide 8's 1,351 pairs are many more, and still only seven drives carry them.
"""


# --------------------------------------------------------------- slide 16
@example(16)
def naive_interval():
    scores = [0.90, 0.94, 0.86, 0.92, 0.88]
    k = len(scores)
    mean = sum(scores) / k
    squares = [(a - mean) ** 2 for a in scores]
    s = math.sqrt(sum(squares) / k)
    se = s / math.sqrt(k)
    half = 1.96 * se
    assert abs(mean - 0.90) < 1e-12
    rows = "\n".join(f"| {fmt(a, 2)} | {fmt(a, 2)} − {fmt(mean, 2)} = {a - mean:+.2f} "
                     f"| {fmt(q, 4)} |" for a, q in zip(scores, squares))
    return f"""
### Worked example — five fold scores, their spread and the tempting interval

Five invented round AUCs.

| round AUC $a_i$ | deviation $a_i - \\bar a$ | squared |
|---|---|---|
{rows}

The mean is $({' + '.join(fmt(a, 2) for a in scores)}) \\div 5 = {fmt(mean, 2)}$. The squares add to
{fmt(sum(squares), 4)}; divided by 5 that is {fmt(sum(squares) / k, 4)}, and its square root is the
spread $s = {fmt(s, 4)}$. The tempting standard error is $s \\div \\sqrt{{5}} = {fmt(s, 4)} \\div {fmt(math.sqrt(k), 3)} =
{fmt(se, 4)}$, and the tempting interval $\\pm 1.96 \\times {fmt(se, 4)} = \\pm {fmt(half, 3)}$. Report
"{fmt(mean, 2)} ± {fmt(s, 3)}, the spread over 5 folds", not "{fmt(mean, 2)} ± {fmt(half, 3)}": on the
drives, slide 17 measured the real spread at twice the tempting one.
"""


# --------------------------------------------------------------- slide 31
@example(31)
def decomposition_by_hand():
    truth = 10.0
    predictions = [8.0, 9.0, 7.0, 10.0, 6.0]
    n = len(predictions)
    average = sum(predictions) / n
    bias2 = (average - truth) ** 2
    variance = sum((p - average) ** 2 for p in predictions) / n
    direct = sum((truth - p) ** 2 for p in predictions) / n
    noise = 1.0
    assert abs(bias2 + variance - direct) < 1e-12        # the identity, without noise
    rows = "\n".join(f"| {fmt(p, 0)} | ({fmt(p, 0)} − {fmt(average, 0)})² = {fmt((p - average) ** 2, 0)} "
                     f"| ({fmt(truth, 0)} − {fmt(p, 0)})² = {fmt((truth - p) ** 2, 0)} |"
                     for p in predictions)
    return f"""
### Worked example — the decomposition on five models

Five models, each trained on its own data, asked about the same point, where the truth is
$f = {fmt(truth, 0)}$. The noise of a new measurement has variance $\\sigma^2 = {fmt(noise, 0)}$.

| prediction $\\hat f$ | for the variance: $(\\hat f - \\bar f)^2$ | for the check: $(f - \\hat f)^2$ |
|---|---|---|
{rows}

Average prediction $\\bar f = ({' + '.join(fmt(p, 0) for p in predictions)}) \\div 5 = {fmt(average, 0)}$.
Bias $= {fmt(average, 0)} - {fmt(truth, 0)} = {fmt(average - truth, 0)}$, so bias² $= {fmt(bias2, 0)}$.
Variance = middle column summed ÷ 5 = {fmt(variance * n, 0)} ÷ 5 = {fmt(variance, 0)}; the truth does not enter it.
Check: the last column, the squared misses from the truth, summed ÷ 5 = {fmt(direct * n, 0)} ÷ 5 = {fmt(direct, 0)},
exactly bias² + variance = {fmt(bias2, 0)} + {fmt(variance, 0)}. Against a new noisy measurement add the
noise: expected error = {fmt(bias2, 0)} + {fmt(variance, 0)} + {fmt(noise, 0)} = {fmt(bias2 + variance + noise, 0)}.
"""


# --------------------------------------------------------------- slide 51
@example(51)
def best_of_many():
    scores = [0.48, 0.55, 0.51, 0.46, 0.53]
    best = max(scores)
    average = sum(scores) / len(scores)
    assert best > average > 0.5 - 0.01
    return f"""
### Worked example — the best of five worthless candidates

Five candidates whose true AUC is 0.50, each measured once with some noise:
{', '.join(fmt(a, 2) for a in scores)}. Their average is $({' + '.join(fmt(a, 2) for a in scores)}) \\div 5 =
{fmt(average, 3)}$, close to the truth. Their maximum is {fmt(best, 2)}, and it is the number a search
reports. Nothing was done dishonestly; the maximum of noisy numbers is pushed up by the
noise, by {fmt(best - average, 3)} here. With 25 candidates instead of 5 the luckiest is luckier still.
"""


# --------------------------------------------------------------- slide 55
@example(55)
def threshold_sweep():
    miss, alarm = 2600, 140

    def bill(drives, t):
        missed = sum(1 for p, failed in drives if failed and p < t)
        alarms = sum(1 for p, failed in drives if not failed and p >= t)
        return missed, alarms, missed * miss + alarms * alarm

    validation = [(0.70, 1), (0.08, 1), (0.30, 0), (0.12, 0), (0.06, 0), (0.04, 0)]
    test = [(0.65, 1), (0.15, 1), (0.40, 0), (0.09, 0), (0.07, 0), (0.03, 0)]
    candidates = [0.05, 0.10, 0.50]
    valid_bills = {t: bill(validation, t) for t in candidates}
    chosen = min(candidates, key=lambda t: valid_bills[t][2])
    test_bills = {t: bill(test, t) for t in candidates}
    cheat = min(candidates, key=lambda t: test_bills[t][2])
    assert chosen != cheat and test_bills[chosen][2] > test_bills[cheat][2]
    rows = "\n".join(
        f"| {fmt(t, 2)} | {valid_bills[t][0]} × 2,600 + {valid_bills[t][1]} × 140 = {valid_bills[t][2]:,} "
        f"| {test_bills[t][0]} × 2,600 + {test_bills[t][1]} × 140 = {test_bills[t][2]:,} |"
        for t in candidates)
    return f"""
### Worked example — three thresholds, six validation drives, six test drives

One model has given probabilities to six validation drives, two of them failing, and to
six test drives, two failing. A drive is flagged when its probability is at least the
threshold; a missed failure costs 2,600 EUR, a false alarm 140.

| threshold | bill on the validation drives (EUR) | bill on the test drives (EUR) |
|---|---|---|
{rows}

On the validation drives the cheapest threshold is {fmt(chosen, 2)}: that is the choice, and the
test set then reports {test_bills[chosen][2]:,} EUR. Had the threshold been chosen on the test drives
themselves, {fmt(cheat, 2)} would have won with {test_bills[cheat][2]:,} EUR: cheaper, and not reportable,
because it was picked by looking at the bill being reported.
"""


if __name__ == "__main__":
    run(__file__)
