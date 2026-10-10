---
title: "Lessons 1 and 2: What to Keep"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini, Università degli Studi dell'Aquila"
date: "A recap before lesson 3"
---

# Lesson 1: what to keep

- We minimise the error on the sample but care about new data: so hold data out
- Baseline first: always "benign" scores **62.9%**
- With 1% positives: **0.986** accuracy, **0** recall
- **77%** on coin-flip labels, from choosing features before the split
- Four traps raise no error: leakage, imbalance, shortcuts, one split

::: notes
A two-minute recap at the start of lesson 3, one line per idea. The 77% in the
fourth bullet is why every step that learns goes inside the pipeline. Read the first
bullet as one argument: we minimise the empirical risk, we care about the
expected risk, and the held-out set is the only honest bridge between them.

The 77% is the number lesson 1 left behind: pure noise, 5,000 random features,
and a model that looks good only because the feature selection had seen the
test rows. Ask the room which line of code caused it before saying it.
Lesson 1 handout, sections 2 and 7.
:::

# Lesson 2: what to keep

- Look before you touch: types, gaps, shapes
- Ask **why** a value is missing, not how many
- A tenure of **−3** passes both outlier rules
- Every encoding makes a claim about categories
- **94 of 128** imputed rows borrowed from the test set
- What learns from data is fitted on training rows only

::: notes
On the second bullet, name the three reasons: missing completely at random
(MCAR), at random (MAR, explained by other columns), not at random (MNAR,
explained by the missing value itself). Filling with the mean weakens every
correlation the column has, by about the square root of one minus the missing
fraction.

The 94 of 128 is lesson 2's number: imputing with nearest neighbours before
the split let most of the filled rows copy a test customer. Nothing raised an
error, and the score looked good.

Close on the last bullet, because it is the bridge to today: the models of
lesson 3 go inside the same pipeline, after the same split. Lesson 2 handout,
sections 3, 4 and 9.
:::
