"""Recompute every number lesson 4 works out by hand.

    python Lessons/04_classification_and_metrics/Docs/worked_examples.py

Run by ``tools/verify_lesson.py``. The confusion-matrix figures come from
notebook 2 and are recomputed here from the four counts alone, so a metric
quoted in the handout cannot drift away from the matrix printed beside it.
"""

import math

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

# --------------------------------- Section 4, the two losses

# A drive failed and the model said 0.0001.
p = 0.0001
same("4.1 squared error for a confident falsehood", (p - 1) ** 2, 0.9998)
same("4.1 log loss for the same prediction", -math.log(p), 9.21, tolerance=5e-3)
same("4.1 log loss for an honest half", -math.log(0.5), 0.69, tolerance=5e-3)
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

# The same four sectors on the scale the labels were actually generated on.
transformed = transform_features(drives)["reallocated_sectors"]
mean, deviation = transformed.mean(), transformed.std(ddof=0)
weight = TRUE_COEFFICIENTS["reallocated_sectors"]
severity = lambda sectors: (TRUE_INTERCEPT
                            + weight * (np.log1p(sectors) - mean) / deviation)

same("1.1 the first four sectors, in log-odds", severity(4) - severity(0), 4.7,
     tolerance=0.05)
same("1.1 sectors 40 to 44, in log-odds", severity(44) - severity(40), 0.27,
     tolerance=5e-3)
same("1.1 a drive with no reallocated sectors", sigmoid(severity(0)), 0.0001,
     tolerance=5e-5)
same("1.1 the same drive with four", sigmoid(severity(4)), 0.013, tolerance=5e-4)
same("1.1 the same drive with forty", sigmoid(severity(40)), 0.85, tolerance=5e-3)
same("1.1 the same drive with forty-four", sigmoid(severity(44)), 0.89,
     tolerance=5e-3)

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

print(f"lesson 4: {checks} hand-worked numbers recomputed, all agree")
