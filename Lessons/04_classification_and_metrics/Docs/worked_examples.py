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


# --------------------------------- Section 2, the sigmoid and the odds

same("2.2 the sigmoid's slope at zero", sigmoid(0) * (1 - sigmoid(0)), 0.25)
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

print(f"lesson 4: {checks} hand-worked numbers recomputed, all agree")
