---
title: "Exercise 4 — Classification and Evaluation Metrics"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini — Università degli Studi dell'Aquila"
date: "Set 16 October 2026 · due Friday 23 October 2026, 23:59"
---

## What this exercise is for

Lesson 4 argued that fitting a classifier is the easy part and judging one is
not. This exercise is built so that the code takes an hour and the **judgement
takes the rest** — and the marks follow the judgement.

You will build a classifier on a dataset you have not seen, report it honestly,
and defend one decision that the data cannot make for you.

**Submit** a single notebook, `exercise04_<surname>.ipynb`, that runs top to
bottom in the course container with no manual steps. Written answers go in
markdown cells, next to the evidence for them.

---

## The data

A hospital wants to predict which discharged patients will be readmitted within
thirty days — the same shape of problem as the disk fleet, and none of the same
numbers. The data ships with this exercise and needs no network:

```python
from patient_data import load_patients, transform_features

discharges = load_patients()
X = transform_features(discharges)
y = discharges["readmitted"]
```

9,000 discharge records, six columns, and a `readmitted` label. As with the disk
fleet, the coefficients that generated the labels are written down in
`patient_data.py` — **do not read them before attempting question 13.** Reading
them afterwards, to mark yourself, is why they are there.

You may substitute a different binary classification dataset with a positive
rate below 20% if you prefer, provided you state what it is, where it came from,
what the two classes mean, and that it runs offline in the container.

Whatever you use, the first thing your notebook must print is the **base
rate**.

---

## Part 1 — Build the thing (25 marks)

1. Load the data, split it into training and test sets **stratified by the
   label**, and say in one sentence why stratification matters here.
2. Build a scikit-learn `Pipeline` containing every preprocessing step from
   lesson 2 and a `LogisticRegression`. Nothing may be fitted before the split.
3. Report the **accuracy of the always-negative baseline** and the accuracy of
   your model, side by side, and comment on the gap in one sentence.
4. Report the confusion matrix at threshold 0.5.

**What is being marked:** that the pipeline is genuinely leak-free, and that the
baseline is present. A submission without the baseline cannot score full marks
however good the model is.

---

## Part 2 — Report it honestly (25 marks)

5. Report precision, recall and F1 **for the positive class**, and state each as
   an English sentence containing the actual counts — not just the ratio.
6. Plot the **precision-recall curve** with its no-skill baseline drawn on it,
   and report the average precision.
7. Plot the **ROC curve** and report the AUC.
8. In **three or four sentences**, say which of these numbers you would put in
   an email to the hospital's operations manager, and which you would leave out.
   Justify both choices.

**What is being marked:** question 8. Questions 5 to 7 are the evidence for it.

---

## Part 3 — Choose a threshold, and defend it (30 marks)

The hospital gives you two figures:

- A patient flagged for follow-up who would not have been readmitted costs
  **€180** — a nurse's call and a clinic slot.
- A patient not flagged who is readmitted costs **€4,400** — the readmission
  itself, plus the penalty the hospital pays.

9. Compute the theoretical cost-optimal threshold from these two numbers, using
   the formula from the lesson. Show the arithmetic.
10. Sweep the threshold empirically and plot the total cost. Report the cheapest
    threshold and the cost at it, against the cost at 0.5.
11. State the size of the saving, in euros and as a percentage.
12. **In a paragraph of five to eight sentences**, defend the threshold you would
    actually deploy. You are not required to choose the cheapest one. You *are*
    required to say what you chose, why, what it costs, and what you would need
    to know to be more confident.

**What is being marked:** question 12, and it is worth more than any other
single answer on this sheet. A defensible paragraph attached to a merely
adequate model scores higher than a silent optimum.

Things a good paragraph tends to mention: how flat the cost curve is near its
minimum; that the two costs are estimates and what happens if they are wrong;
that a threshold optimised on the test set is optimistic; and whether the
hospital can absorb the number of follow-ups your threshold implies, since a
clinic has a finite capacity that no cost formula knows about.

---

## Part 4 — Two short questions (20 marks)

13. **The decoy.** Exactly one column in `patient_data.py` was generated with a
    coefficient of zero. Identify it *before* opening the file, present the
    evidence that led you there, and say how confident you were.

    Then open `TRUE_COEFFICIENTS` and mark yourself. If you were right, say what
    would have happened had the column been weakly predictive rather than
    useless; if you were wrong, say what misled you. **Both outcomes score the
    same** — the marks are for the evidence and the honesty, not the guess.

14. **The imbalance test.** Take your fitted model, and *without retraining it*,
    construct a version of your test set in which the positive rate is roughly a
    quarter of what it is now — by removing positives at random, keeping every
    negative. Recompute AUC, average precision and precision at your chosen
    threshold.

    Report the three numbers before and after, and explain in two or three
    sentences which of them moved, which did not, and why. Then say what this
    implies for anyone reporting AUC alone.

---

## Marking

| Part | Marks |
|---|---|
| 1 — Build the thing | 25 |
| 2 — Report it honestly | 25 |
| 3 — Choose a threshold, and defend it | 30 |
| 4 — Two short questions | 20 |
| **Total** | **100** |

Marks are lost for:

- any preprocessing fitted before the split, or a threshold chosen on the test
  set without saying so (**−10**, and say so and lose nothing);
- reporting the weighted average without stating that it is weighted
  (**−5**);
- reporting AUC as the only summary of an imbalanced problem (**−5**);
- a notebook that does not run top to bottom in the container (**−10**).

Marks are **not** lost for a model that performs poorly, provided the report is
honest about it. A well-documented mediocre model is worth more here than an
undocumented good one, which is the same standard the final project will apply.

---

## Getting help

Bring questions to the start of lesson 5 on 23 October, or email
fabio.antonini.1969@gmail.com. If you are stuck on the mechanics of Part 3,
re-read notebook 3 section 4 — the cost sweep there is the same computation on
different numbers.
