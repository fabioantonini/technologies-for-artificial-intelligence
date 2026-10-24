"""Recompute every number lesson 4 works out by hand.

    python Lessons/04_classification_and_metrics/Docs/worked_examples.py

Run by ``tools/verify_lesson.py``. The confusion-matrix figures come from
notebook 2 and are recomputed here from the four counts alone, so a metric
quoted in the handout cannot drift away from the matrix printed beside it.
"""

import math
import sys

HANDOUT = "Lessons/04_classification_and_metrics/Docs/classification_and_metrics.md"
checks = 0


def same(name: str, computed, printed, tolerance=5e-3) -> None:
    global checks
    checks += 1
    if abs(computed - printed) > tolerance:
        raise SystemExit(
            f"{HANDOUT}: {name}\n"
            f"    the handout prints {printed}\n"
            f"    recomputing gives  {computed}")


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


# --------------------------------- Section 1, the logarithmic scale

# Section 1.1 claims 0 -> 4 reallocated sectors is a 17 times larger move than
# 40 -> 44. Recomputed from the transform itself, not from the handout's logs.
same("1.1 how much larger the first four sectors are",
     (math.log1p(4) - math.log1p(0)) / (math.log1p(44) - math.log1p(40)), 17,
     tolerance=0.3)

# --------------------------------- Section 2, the sigmoid and the odds

same("2.2 the sigmoid's slope at zero", sigmoid(0) * (1 - sigmoid(0)), 0.25)

# Section 2.2 contrasts two units of log-odds spent in two places. The four
# probabilities are asserted individually as well as by their differences, so
# each digit the handout prints is checked rather than only their gap.
same("2.2 the probability at z = 0", sigmoid(0), 0.50, tolerance=5e-5)
same("2.2 the probability at z = 2", sigmoid(2), 0.88, tolerance=5e-3)
same("2.2 the probability at z = 6", sigmoid(6), 0.9975, tolerance=5e-5)
same("2.2 the probability at z = 8", sigmoid(8), 0.9997, tolerance=5e-5)
same("2.2 two units of evidence from z = 0", sigmoid(2) - sigmoid(0), 0.38,
     tolerance=5e-3)
same("2.2 the same two units from z = 6", sigmoid(8) - sigmoid(6), 0.0021,
     tolerance=5e-5)
same("2.2 how much less the second pair buys",
     (sigmoid(2) - sigmoid(0)) / (sigmoid(8) - sigmoid(6)), 178, tolerance=1.5)
same("2.4 the base rate implied by the intercept", sigmoid(-6.09), 0.0023,
     tolerance=5e-5)
same("2.4 the odds multiplier for a coefficient of 1.80",
     math.exp(1.80), 6.0, tolerance=0.06)

# The caution about odds ratios: odds of 9, multiplied by 5, as a probability.
odds = 9 * 5
same("2.4 odds of 9 multiplied by five, as a probability",
     odds / (1 + odds), 0.978, tolerance=5e-4)

# Section 2.3's table: three drives, the same evidence (odds x 6), against
# adding 0.10 to the probability. The average drive's risk is rebuilt from the
# intercept, not read off the table.
average = sigmoid(-6.20)
for p, odds_before, odds_after, risk_after, step in (
        (average, 0.002, 0.012, 0.012, 0.102),
        (0.50, 1, 6, 0.86, 0.60),
        (0.95, 19, 114, 0.991, 1.05)):
    o = p / (1 - p)
    same(f"2.3 odds before at p = {p:.4f}", o, odds_before, tolerance=5e-4 * max(1, o))
    same(f"2.3 odds after at p = {p:.4f}", 6 * o, odds_after,
         tolerance=5e-4 * max(1, 6 * o))
    same(f"2.3 risk after at p = {p:.4f}", 6 * o / (1 + 6 * o), risk_after,
         tolerance=5e-4 if risk_after > 0.9 else 5e-3)
    same(f"2.3 adding 0.10 at p = {p:.4f}", p + 0.10, step, tolerance=5e-4)
same("2.3 the average drive's risk", average, 0.002, tolerance=5e-4)
same("2.3 the factor behind 1.80", math.exp(1.80), 6.05, tolerance=5e-3)
for p, points in ((average, 1), (0.50, 36), (0.95, 4)):
    o = 6 * p / (1 - p)
    same(f"2.3 points added at p = {p:.4f}", 100 * (o / (1 + o) - p), points,
         tolerance=0.5)

# --------------------------------- Section 3.1, three drives, two settings

# A failed, B and C did not. Each factor is the probability of the outcome that
# occurred, built from P(fail) rather than read off the handout's table.
outcome = [1, 0, 0]
for name, p_fail, product, surprise in (("setting 1", [0.8, 0.1, 0.2], 0.576, 0.55),
                                        ("setting 2", [0.3, 0.6, 0.5], 0.060, 2.81)):
    factors = [p if y else 1 - p for p, y in zip(p_fail, outcome)]
    same(f"3.1 the likelihood under {name}", math.prod(factors), product, tolerance=5e-4)
    same(f"3.1 the summed surprise under {name}",
         sum(-math.log(f) for f in factors), surprise)
    same(f"3.1 the summed surprise under {name}, as minus the log of the product",
         -math.log(math.prod(factors)), surprise)
# The per-drive surprises the handout adds up, each from its own P(fail).
for p_fail, y, printed in ((0.8, 1, 0.22), (0.1, 0, 0.11), (0.2, 0, 0.22),
                           (0.3, 1, 1.20), (0.6, 0, 0.92), (0.5, 0, 0.69)):
    same(f"3.1 the surprise of a drive given P(fail) {p_fail}, y = {y}",
         -math.log(p_fail if y else 1 - p_fail), printed)
same("3.1 how many times more probable under setting 1",
     math.prod([0.8, 0.9, 0.8]) / math.prod([0.3, 0.4, 0.5]), 9.6, tolerance=0.05)

# --------------------------------- Section 4, the two losses

# A drive failed and the model said 0.0001.
p = 0.0001
same("4.1 squared error for a confident falsehood", (p - 1) ** 2, 0.9998)
same("4.1 log loss for the same prediction", -math.log(p), 9.21, tolerance=5e-3)
same("4.1 log loss for an honest half", -math.log(0.5), 0.69, tolerance=5e-3)
# The likelihood underflows: 6,000 training drives, each at p = 0.5 at the start,
# and at the fitted model's mean log loss 0.0692 (notebook 1's printed history).
same("3.3 the starting likelihood, as a power of ten", 6000 * math.log10(0.5), -1806,
     tolerance=0.5)
# The float limits the handout quotes, read off the machine rather than recalled:
# the smallest subnormal (5e-324) and the smallest normal (2.2e-308).
# Compared as powers of ten, since 1e324 itself overflows.
same("3.3 smallest positive 64-bit float, about 5e-324",
     math.log10(math.ulp(0.0)), math.log10(5) - 324, tolerance=0.01)
same("3.3 smallest full-precision 64-bit float, about 2e-308",
     math.log10(sys.float_info.min), math.log10(2) - 308, tolerance=0.05)
same("3.3 the starting likelihood is stored as exactly zero", 0.5 ** 6000, 0.0,
     tolerance=0.0)
same("3.3 the fitted likelihood, as a power of ten", -6000 * 0.0692 / math.log(10), -180,
     tolerance=0.5)
same("3.3 the cost of a one-in-a-thousand surprise", math.log(1000), 6.9,
     tolerance=0.01)   # the handout says "about 6.9"; it is 6.9078

# The gradient table: squared error carries a factor p(1-p), log loss does not.
for probability, squared, logloss in ((0.5, 0.2500, 0.500),
                                      (0.1, 0.1620, 0.900),
                                      (0.01, 0.0196, 0.990),
                                      (0.001, 0.0020, 0.999)):
    same(f"4.2 squared-error gradient at p={probability}",
         abs(2 * (probability - 1) * probability * (1 - probability)),
         squared, tolerance=5e-4)
    same(f"4.2 log-loss gradient at p={probability}",
         abs(probability - 1), logloss, tolerance=5e-4)

# --------------------------------- Section 6, the confusion matrix

# The only inputs: the four counts from notebook 2.
tn, fp, fn, tp = 1911, 13, 33, 43

same("6.1 the test set adds up", tn + fp + fn + tp, 2000)
same("6.2 precision", tp / (tp + fp), 0.768, tolerance=5e-4)
same("6.2 recall", tp / (tp + fn), 0.566, tolerance=5e-4)
same("6.2 specificity", tn / (tn + fp), 0.993, tolerance=5e-4)
same("6.2 the false positive rate", 1 - tn / (tn + fp), 0.007, tolerance=5e-4)

precision, recall = tp / (tp + fp), tp / (tp + fn)
same("6.3 F1", 2 * precision * recall / (precision + recall), 0.652,
     tolerance=5e-4)

same("5.1 the always-healthy baseline", (tn + fp) / 2000, 0.9620, tolerance=5e-5)
same("5.1 the model's accuracy", (tn + tp) / 2000, 0.9770, tolerance=5e-5)

# Flag every drive: the model F1 exists to punish.
all_precision, all_recall = 76 / 2000, 1.0
same("6.3 precision when everything is flagged", all_precision, 0.038,
     tolerance=5e-4)
same("6.3 the arithmetic mean flatters it",
     (all_precision + all_recall) / 2, 0.519, tolerance=5e-4)
same("6.3 the harmonic mean does not",
     2 * all_precision * all_recall / (all_precision + all_recall), 0.073,
     tolerance=5e-4)

# --------------------------------- Section 7, the cost-optimal threshold

cost_fp, cost_fn = 140, 2600
same("7.2 the cost-optimal threshold", cost_fp / (cost_fp + cost_fn), 0.051,
     tolerance=5e-4)
same("7.2 how many false alarms a miss is worth", cost_fn / cost_fp, 19,
     tolerance=0.6)

# The three policies, priced from the counts notebook 3 reports.
same("7.2 the cost of doing nothing", 76 * cost_fn, 197_600)
same("7.2 the cost at threshold 0.50", 13 * cost_fp + 33 * cost_fn, 87_620)
same("7.2 the cost at threshold 0.08", 121 * cost_fp + 11 * cost_fn, 45_540)
same("7.2 the saving, as a fraction",
     (87_620 - 45_540) / 87_620, 0.48, tolerance=5e-3)

# --------------------------------- Section 9, class weights

# balanced weighting makes each failure count m / (K m_k).
m, positives = 6000, 230
weight_positive = m / (2 * positives)
weight_negative = m / (2 * (m - positives))
same("9 how many healthy drives a failure is worth under balanced weights",
     weight_positive / weight_negative, 25, tolerance=0.6)

# --------------------------------- Sections 1.1 and 2.2, on the real data
#
# These are the only numbers here that need the dataset. Both sections make a
# claim about *scale* - what four reallocated sectors are worth - and the two
# claims disagree on purpose, because one is read off a model fitted to the raw
# counter and the other off the log scale the labels were generated on. Refit
# from the raw drives rather than trusting either.

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "Notebooks"))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from disk_data import (TRUE_COEFFICIENTS, TRUE_INTERCEPT, load_drives,
                       transform_features)

drives = load_drives()
raw = drives[["reallocated_sectors"]]
fit = make_pipeline(StandardScaler(),
                    LogisticRegression(max_iter=1000)).fit(raw, drives["failed"])
scaler, model = fit[0], fit[1]

# Put the fitted curve back into sectors, so its steepest point is a count.
slope = model.coef_[0][0] / scaler.scale_[0]
intercept = model.intercept_[0] - model.coef_[0][0] * scaler.mean_[0] / scaler.scale_[0]
curve = lambda sectors: sigmoid(slope * sectors + intercept)

same("2.2 where the raw-counter curve is steepest", -intercept / slope, 13.2,
     tolerance=0.05)
same("2.2 what the first four sectors are worth to it",
     curve(4) - curve(0), 0.03, tolerance=5e-3)
same("2.2 what sectors 10 to 14 are worth to it",
     curve(14) - curve(10), 0.31, tolerance=5e-3)

# The table of the six counters in section 1.1: median, and 5th to 95th percentile.
for column, (low, median, high), tolerance in (
        ("reallocated_sectors", (0, 2, 6), 0.5),
        ("spin_retry_count", (0, 0, 1), 0.5),
        ("read_error_rate", (0.0004, 0.0017, 0.0070), 5e-5),
        ("seek_error_rate", (0.0005, 0.0022, 0.0097), 5e-5),
        ("temperature_c", (28.3, 35.2, 42.3), 0.05),
        ("power_on_hours", (5_100, 16_100, 37_000), 50)):
    for quantile, printed in zip((0.05, 0.5, 0.95), (low, median, high)):
        same(f"1.1 {column}, {quantile:.0%} quantile",
             drives[column].quantile(quantile), printed, tolerance=tolerance)
same("1.1 drives with no spin retries", (drives["spin_retry_count"] == 0).mean(),
     0.84, tolerance=5e-3)
same("1.1 the median drive's service, in months",
     drives["power_on_hours"].median() / (24 * 365.25 / 12), 22, tolerance=0.5)

same("2.2 the raw-count model's slope, per sector", slope, 0.336, tolerance=5e-4)
same("2.2 its z with no sectors", intercept, -4.43, tolerance=5e-3)
same("2.2 the first four sectors, in its z", 4 * slope, 1.34, tolerance=0.01)
same("2.2 its drive with no sectors", curve(0), 0.012, tolerance=5e-4)
same("2.2 its drive with four", curve(4), 0.044, tolerance=5e-4)
same("2.2 drives with at most four sectors",
     (drives["reallocated_sectors"] <= 4).mean(), 0.90, tolerance=5e-3)

# The same four sectors on the scale the labels were actually generated on.
transformed = transform_features(drives)["reallocated_sectors"]
mean, deviation = transformed.mean(), transformed.std(ddof=0)
weight = TRUE_COEFFICIENTS["reallocated_sectors"]
same("2.3 the intercept quoted for an average drive", TRUE_INTERCEPT, -6.20, tolerance=1e-9)
same("2.3 the sectors coefficient behind the factor 6", weight, 1.80, tolerance=1e-9)
severity = lambda sectors: (TRUE_INTERCEPT
                            + weight * (np.log1p(sectors) - mean) / deviation)

same("1.1 the first four sectors, in log-odds", severity(4) - severity(0), 4.7,
     tolerance=0.05)
same("1.1 sectors 40 to 44, in log-odds", severity(44) - severity(40), 0.27,
     tolerance=5e-3)
same("1.1 a drive with no reallocated sectors", sigmoid(severity(0)), 0.00012,
     tolerance=5e-6)
same("1.1 the same drive with four", sigmoid(severity(4)), 0.013, tolerance=5e-4)
same("1.1 ... about a hundredfold increase", sigmoid(severity(4)) / sigmoid(severity(0)),
     100, tolerance=10)
same("1.1 the same drive with forty", sigmoid(severity(40)), 0.85, tolerance=5e-3)
same("1.1 the same drive with forty-four", sigmoid(severity(44)), 0.89,
     tolerance=5e-3)
same("2.2 the true model's hundredfold rise, in probability",
     sigmoid(severity(4)) - sigmoid(severity(0)), 0.013, tolerance=5e-4)

# --------------------------------- Section 3.4, the descent itself
#
# Re-run the fifteen lines rather than reading the notebook's printout, so the
# figure the handout quotes is checked against a second execution of the same
# arithmetic from the same raw drives.

from sklearn.model_selection import train_test_split

split = train_test_split(transform_features(drives), drives["failed"],
                         test_size=0.25, random_state=42,
                         stratify=drives["failed"])
Z = StandardScaler().fit(split[0]).transform(split[0])
labels = split[2].to_numpy()

weights, offset_b, curve = np.zeros(Z.shape[1]), 0.0, []
for _ in range(4000):
    probability = 1 / (1 + np.exp(-(Z @ weights + offset_b)))
    curve.append(-np.mean(labels * np.log(probability + 1e-12)
                          + (1 - labels) * np.log(1 - probability + 1e-12)))
    residual = probability - labels
    weights -= 0.5 * (Z.T @ residual) / len(labels)
    offset_b -= 0.5 * residual.mean()
curve = np.array(curve)

# w = b = 0 gives every drive p = 0.5, so the first loss is log 2 by hand.
same("3.4 the log loss a zero model starts at", curve[0], math.log(2),
     tolerance=5e-5)
same("3.4 the handout's printed starting value", curve[0], 0.6931,
     tolerance=5e-5)
same("3.4 where 4,000 iterations end", curve[-1], 0.0692, tolerance=5e-5)
same("3.4 the share of the fall in the first 100 iterations",
     (curve[0] - curve[100]) / (curve[0] - curve[-1]), 0.967, tolerance=5e-4)
same("3.4 what the last 100 iterations are worth",
     curve[3900] - curve[-1], 0.0, tolerance=5e-6)

# ------------- Sections 4.3, 6.3 and 7.2, the three steps now written out

# 4.3: the Hessian and its positive semi-definiteness. The formula is not
# evaluated against itself - the second derivatives are measured numerically
# from the cost, and the curvature is probed in random directions.
import numpy as np                                                      # noqa: E402

rng4 = np.random.default_rng(20261009)
Xh = rng4.normal(size=(60, 3))
yh = (rng4.random(60) < 0.4).astype(float)


def cost(weights):
    z = Xh @ weights
    prob = 1 / (1 + np.exp(-z))
    return -np.mean(yh * np.log(prob) + (1 - yh) * np.log(1 - prob))


point = rng4.normal(size=3) * 0.5
step = 1e-4
numeric = np.empty((3, 3))
for a in range(3):
    for b_ in range(3):
        ea, eb = np.zeros(3), np.zeros(3)
        ea[a] = eb[b_] = step
        numeric[a, b_] = (cost(point + ea + eb) - cost(point + ea)
                          - cost(point + eb) + cost(point)) / step ** 2

probs = 1 / (1 + np.exp(-(Xh @ point)))
formula = Xh.T @ np.diag(probs * (1 - probs)) @ Xh / len(yh)
same("4.3 the Hessian of the cost is X'SX/m",
     float(np.abs(numeric - formula).max()), 0.0, tolerance=1e-4)
same("4.3 and it has no negative eigenvalue",
     float(np.linalg.eigvalsh(formula).min() >= 0), 1.0, tolerance=0)
directions = rng4.normal(size=(2_000, 3))
same("4.3 no direction curves downwards",
     float(min((u @ formula @ u) for u in directions) >= 0), 1.0, tolerance=0)

# 6.3: the harmonic mean written as an average of reciprocals, and F-beta
# reducing to F1. Checked against sklearn rather than against the formula.
from sklearn.metrics import fbeta_score                                 # noqa: E402

same("6.3 the harmonic mean is the inverted average of the inverses",
     2 / (1 / precision + 1 / recall),
     2 * precision * recall / (precision + recall), tolerance=1e-12)

predicted = np.array([1] * tp + [1] * fp + [0] * fn + [0] * tn)
actual = np.array([1] * tp + [0] * fp + [1] * fn + [0] * tn)
same("6.3 F-beta at beta = 1 is F1",
     float(fbeta_score(actual, predicted, beta=1)),
     2 * precision * recall / (precision + recall), tolerance=1e-9)
for beta in (0.5, 2.0):
    weighted = (1 + beta ** 2) / (1 / precision + beta ** 2 / recall)
    same(f"6.3 the weighted-reciprocal form of F-beta at beta = {beta}",
         float(fbeta_score(actual, predicted, beta=beta)), weighted,
         tolerance=1e-9)

# 7.2: the rearrangement, checked as the statement it makes about decisions -
# below t* leaving the drive alone really is the cheaper of the two.
star = cost_fp / (cost_fp + cost_fn)
for probability, cheaper_to_flag in ((star - 0.01, False), (star + 0.01, True)):
    flag = (1 - probability) * cost_fp
    leave = probability * cost_fn
    same(f"7.2 at p = {probability:.3f} flagging is the cheaper action: {cheaper_to_flag}",
         float(flag < leave), float(cheaper_to_flag), tolerance=0)

print(f"lesson 4: {checks} hand-worked numbers recomputed, all agree")
