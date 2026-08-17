---
title: "Exercise 5 — Experimental Methodology"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini — Università degli Studi dell'Aquila"
date: "Set 23 October 2026 · due Friday 30 October 2026, 23:59"
---

## What this exercise is for

Every previous exercise asked you to build something. This one asks you to
**refuse to sign something off**.

A colleague has left the company, leaving behind `suspicious_result.ipynb`: a
gearbox failure model reporting an area under the ROC curve (**AUC**) of
**0.92** under five-fold cross-validation,
with tight agreement between folds. The maintenance team wants to start
scheduling inspections from it next month.

The notebook runs. Every number in it is real. It is also wrong, in more than
one way.

Your job is to find out how, fix it, and report the number that should have been
there — which will be worse, and is the point.

**Submit** a single notebook, `exercise05_<surname>.ipynb`, that runs top to
bottom in the course container. Written answers go in markdown cells beside the
evidence for them.

---

## The data

`turbine_data.py` ships with this exercise and needs no network.

```python
from turbine_data import load_turbines, FEATURES, METADATA, ALL_COLUMNS

data = load_turbines()
```

**480 weekly readings from 40 wind turbines.** Each turbine contributes twelve
readings. The `needs_repair` label belongs to the *turbine* — a gearbox either
needed an unplanned repair that quarter or it did not — so every reading from
the same turbine carries the same label.

The columns come in two kinds, and the distinction matters:

- `FEATURES` — weekly telemetry: vibration, oil temperature, particle count,
  power output, ambient humidity. These vary from week to week.
- `METADATA` — the installation record: hub height, rotor diameter,
  commissioning year and so on. **Recorded once per turbine and repeated on
  every one of its rows.**

`TRUE_COEFFICIENTS` in that file records how the labels were generated. Do not
open it until Part 4.

---

## Part 1 — Read the evidence before touching anything (20 marks)

1. Run `suspicious_result.ipynb` unchanged and confirm the reported figure.
2. Then, **without running anything new**, list every methodological concern you
   can identify by reading it. For each, say in one sentence what you think it
   does to the reported number and in which direction.
3. Rank your concerns by how much you expect each to matter, and say why.

**What is being marked:** the reading, not the ranking. A concern that turns out
to be minor still scores if the reasoning was sound. **State your predictions
before you test them** — several are wrong in instructive ways, and a
prediction you got wrong and then investigated is worth more here than a
cautious list.

There are at least three distinct problems. Finding two of them well is worth
more than listing six vaguely.

---

## Part 2 — Test each concern in isolation (30 marks)

For each concern from Part 1, run the experiment that isolates it: change that
one thing, hold everything else fixed, and measure the difference.

Present the results as a **single table** with one row per concern, showing the
reported figure, the corrected figure, and the difference.

You should end up able to say, for example: *"this flaw was worth +0.11 AUC;
this one was worth +0.01; this one changed nothing measurable."*

**What is being marked:** that each experiment changes exactly one thing. A
corrected pipeline that fixes four problems at once tells you the total and
nothing about the parts.

Two hints, because they are methodology rather than puzzle:

- One concern needs `GroupKFold` and the `turbine` column.
- One concern needs the preprocessing moved inside a `Pipeline`.

---

## Part 3 — Produce the honest number (25 marks)

4. Build the analysis as it should have been done, and report the result with
   its spread.
5. State plainly how much lower it is than the headline, in AUC points.
6. **In a paragraph of five to eight sentences**, write what you would tell the
   maintenance team. They are not statisticians; they want to know whether they
   can schedule inspections from this model, and if so with what expectation.

   Say what the model is worth, what it cannot do, and what you would need in
   order to give them a better answer.

**What is being marked:** question 6. It is the closest thing in this course to
what the job actually consists of.

A good paragraph tends to note: that the honest figure still describes a useful
model; that 40 turbines is a small sample and the spread across folds is wide;
what "AUC 0.88" does and does not promise about any individual turbine; and that
a decision threshold has not been chosen yet, which is a separate conversation
about the cost of a needless inspection against a missed failure.

---

## Part 4 — What was actually in the data (25 marks)

Now open `turbine_data.py` and read `TRUE_COEFFICIENTS`.

7. Which columns genuinely carry signal, and which do not? Compare against what
   your corrected model relies on.
8. One telemetry column was generated with a coefficient of **exactly zero**.
   Did your analysis identify it? If not, what would have been needed?
9. The metadata columns are not in `TRUE_COEFFICIENTS` at all — they play no
   causal role whatsoever. Explain, in two or three sentences, **how a model
   using them scored well under the original evaluation**, and why that is a
   property of the split rather than of the columns.
10. Finally: your corrected model may score *better* without the metadata than
    with it. If that happened, explain why. If it did not, report what you found.

**What is being marked:** question 9. It is the central idea of lesson 5, and
being able to explain it in plain language is the test of having understood it.

---

## Marking

| Part | Marks |
|---|---|
| 1 — Read the evidence | 20 |
| 2 — Test each concern in isolation | 30 |
| 3 — Produce the honest number | 25 |
| 4 — What was actually in the data | 25 |
| **Total** | **100** |

Marks are lost for:

- fixing several problems at once and reporting only the total (**−10**);
- a corrected analysis that still leaks somewhere (**−10**, or **−0** if you
  spot it yourself and say so);
- reporting a figure without its spread (**−5**);
- a notebook that does not run top to bottom in the container (**−10**).

Marks are **not** lost for a corrected figure that is much lower than the
headline. That is the expected outcome and the reason the exercise exists.

---

## Getting help

Bring questions to the start of lesson 6 on 30 October, or email
fabio.antonini.1969@gmail.com. If you are stuck on Part 2, notebook 3 of this
lesson does the same kind of isolation on a different dataset.
