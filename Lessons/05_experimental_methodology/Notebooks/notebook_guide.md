---
title: "Lesson 5 — What Each Notebook Shows"
subtitle: "Notebook guide, lesson 5 — Technologies for Artificial Intelligence"
author: "Fabio Antonini — Università degli Studi dell'Aquila"
date: "30 October 2026 · three notebooks"
---

> A short guide to this lesson's notebooks. For each one: what it is for, the
> concepts in the order they appear, the numbers worth noticing, and **the two-minute
> version**, a summary to read before running it or to come back to afterwards.
> Every number here comes from the committed output of the notebook it belongs to.

---

# Notebook 01 — `01_why_one_split_lies`

## Why one train/test split lies

**What it is for.** Lesson 1 gave the rule: hold data out, never look at it, report the
score. This notebook shows what the rule does **not** give you — the precision of the
number it produces.

**The data.** Lesson 4's drives, cut down to **800 with 29 failures (3.6%)** — because
800 rows with 29 positives is the ordinary case, not an exaggerated one.

**The concepts, in order.**

1. **Two honest experiments, two answers.** Five seeds give test AUCs of 0.961, 0.966,
   0.980, **1.000** and 0.990, each with 7 failures in the test set. Everything done
   correctly; only the rows held out changed.
2. **How wide is it?** 200 splits: worst **0.885**, best **1.000**, spread **0.115**,
   mean 0.955 with a standard deviation of 0.024, and **10 of 200 land above 0.99**.
   The right-hand tail is the dangerous end — a disappointing score makes you keep
   working, a delightful one makes you stop and publish.
3. **Choosing with a noisy number.** Three logistic regressions differing only in
   penalty: mean area under the receiver operating characteristic curve (AUC) **0.9549, 0.9549, 0.9544** — identical to three decimals — and
   they are declared best **91, 71 and 38 times** out of 200. The winner is noise
   wearing the clothes of a result.
4. **k-fold cross-validation**, implemented by hand and matched against scikit-learn
   **exactly** (agreement 0.00e+00), giving **0.9514 ± 0.0187**. Report the spread: five
   fold scores are five measurements of the same quantity.
5. **Stratification is not optional here.** Plain `KFold` gives folds with 1, 5, 7, 7
   and 9 failures out of 160; `StratifiedKFold` gives 5, 6, 6, 6, 6 against a
   fleet-wide rate of 3.62%.
6. **How much it helps**, the comparison across 40 seeds:

   | | single 75/25 split | 5-fold CV |
   |---|---|---|
   | worst | 0.9119 | 0.9439 |
   | best | 1.0000 | 0.9588 |
   | mean | 0.9600 | 0.9532 |
   | spread | 0.0881 | 0.0149 |
   | std | 0.0221 | 0.0032 |

   **Cross-validation is 6.8× more stable**, and both centre in about the same place —
   it is not more pessimistic, it is less arbitrary.

**The two-minute version.** *Eight hundred drives, twenty-nine failures, and one model.
Split it five different ways and the AUC comes out 0.961, 0.966, 0.980, 1.000 and
0.990 — every experiment done correctly. Over 200 splits the score runs from 0.885 to
a perfect 1.000, and ten of the 200 clear 0.99. Then the expensive part: three models
whose honest performance is identical to three decimals each get declared the winner a
share of the time, purely by luck of the split. Cross-validation replaces one
measurement with five, is nearly seven times more stable across seeds, and comes with
a spread you are supposed to report.*

---

# Notebook 02 — `02_bias_variance_and_learning_curves`

## What the error is made of

**What it is for.** Notebook 1 established that a test score moves. This one asks
**why**, and answers with a decomposition that is checked, not asserted — possible only
because the true function is ours to choose.

**The data.** Energy against outdoor temperature, 25 training observations per
universe, 300 universes, irreducible noise of **22 kWh**, so a variance of **484** no
model can remove.

**The concepts, in order.**

1. **Three hundred parallel universes.** Three panels — degree 1 stable and wrong,
   degree 3 about right, degree 12 flying apart at the edges. In real life you drew one
   of those lines and never saw the rest.
2. **Measuring the three parts**, and the identity holding to floating point:

   | degree | bias² | variance | noise | total |
   |---|---|---|---|---|
   | 1 | 5,250.8 | 723.8 | 484.0 | 6,458.6 |
   | 2 | 59.6 | 70.8 | 484.0 | **614.3** |
   | 5 | 8.9 | 368.9 | 484.0 | 861.8 |
   | 12 | 153,537.3 | 32,174,344.9 | 484.0 | 32,328,366.2 |

   Largest disagreement between the sum and the measured mean squared error (MSE): **2.7 × 10⁻¹²**.
3. **Why degree 1's variance is not the smallest** — the entry the textbook picture
   does not predict. Each universe redraws its 25 temperatures, and a straight line
   that cannot follow the curve pivots according to where they fell. Hold the
   temperatures fixed and the variance column climbs from the first row: **33.9, 55.3,
   81.9, 154.5, 709.2, 2,669.8.**
4. **More data moves the answer.** The best degree is 2 at $n = 25$ and **5** at
   $n = 60$ and $n = 150$: the optimum moves right as the data grows, because there is
   finally enough of it to pin down the extra coefficients.
5. **Learning curves**, the diagnostic that needs no truth. High bias: the curves meet
   **low**, 0.715 against 0.718, flat from the first point to the last. High variance:
   training **1.000** against validation 0.847, a gap of 0.153.
6. **And read the slope to the end.** The high-variance curve climbs to **0.857 at 251
   examples** and then moves sideways — 0.834, 0.847, 0.835, 0.832, 0.847 — ending
   *below* its own peak. The last step rises by 0.014, but per fold it is −0.030,
   +0.050, +0.006, −0.009, +0.053: **a slope smaller than its own band is not a
   slope.**

**The two-minute version.** *This notebook does something impossible with real data: it
draws 300 training sets from a known curve and measures bias, variance and noise
separately. The three add up to the measured error to twelve decimal places, so the
decomposition is an identity, not a metaphor. Degree 12 has a variance of 32 million.
The optimal degree moves from 2 to 5 as the data grows from 25 points to 60, so "the
right model" depends on how much data you have. Then learning curves, which give the
same diagnosis without knowing the truth: curves that meet low mean bias and more data
will not help; a wide gap means variance. And read the slope to the end — ours peaks
at 251 examples and then goes flat, which says more rows would buy nothing.*

---

# Notebook 03 — `03_leakage_and_honest_search`

## Leakage that survives cross-validation

**What it is for.** Notebook 1 gave them cross-validation. This one shows what it does
**not** protect against — because a tool trusted wrongly is more dangerous than no tool.

**The data.** **2,000 columns of pure noise, 800 rows, 29 positives.** The honest AUC
of any model built on it is **0.500**.

**The concepts, in order.**

1. **The catastrophe.** Select the 10 most predictive columns, then cross-validate:
   **AUC 0.931 ± 0.053**, with four of five folds between 0.93 and 0.98 — which is
   exactly what a trustworthy result looks like. **Cross-validation did not fail. It
   was lied to.**
2. **Doing it properly — and the surprise.** Selection inside the pipeline, over 20 CV
   seeds: mean **0.658**, min 0.576, max 0.759. Better, **and not 0.500**. Searching a
   large space on a small sample leaves optimism that no rearrangement of the same 800
   rows can remove.
3. **Hyperparameter search is the same problem wearing a hat.** 25 combinations on the
   signal-free table: best candidate **0.7999**, average candidate 0.7265, worst 0.6840.
   What a practitioner reports is the **maximum of 25 noisy estimates**.
4. **Nested cross-validation.** Reported by the search 0.7999; nested **0.6699 ±
   0.1318**; optimism **+0.130** — larger than most differences anyone publishes
   between methods. Four numbers describing the same signal-free data, each step of
   discipline costing a chunk of the score.
5. **Doing lesson 4's threshold honestly.** Three-way split: threshold chosen on
   validation **0.050**, cost on the test set **4,700 €** against **10,540 €** at 0.50.
   The cheating threshold gives 3,300 € — **and is not reportable**, because it was
   chosen on the test set.
6. **Seeds.** What to fix and where; and then 30 perfectly reproducible runs scoring
   from **0.913 to 1.000**. **A fixed seed makes a number repeatable, not correct.**

**The two-minute version.** *Two thousand columns of noise, eight hundred rows, nothing
to learn. Select the ten best features and then cross-validate, and you get 0.93 with
a small standard deviation — cross-validation confirms the mistake instead of catching
it. Put the selection inside the pipeline and it falls to 0.66, which is better and
still not 0.5, because searching a big space on a small sample is optimistic by itself.
Then the same thing for hyperparameters: the best of 25 candidates reports 0.80, nested
cross-validation says 0.67, so the search was optimistic by thirteen points. And the
last section pays lesson 4's debt: choosing the threshold on a validation set instead
of the test set costs more on paper and is the only number you are allowed to report.*
