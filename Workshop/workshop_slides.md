---
title: "Workshop: Late Deliveries"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini, Università degli Studi dell'Aquila"
date: "Two hours in class, after lesson 5"
---

# Two hours, one dataset, six steps

- 1,200 parcels from a courier company: which ones arrive late?
- You take them through the whole workflow of lessons 1 to 5
- Each step: fill the blanks `___`, run the cell, run the check
- The check prints ✓, or ✗ with a hint
- Each step ends with one question, answered in a sentence

::: notes
Say what the session is for: practice, and a quick, honest signal of what each of
them can already do alone. Nothing is collected and nothing is marked.

Insist on the questions. The code is short on purpose; the sentence under each
step is the part that trains them for the exam, where they explain their own
notebook.
:::

# The six steps

| Step | What you do |
|---|---|
| 1 | look at the data, and compute the baseline |
| 2 | split, keeping the share of late parcels equal |
| 3 | one pipeline: fill the gaps, scale, logistic regression |
| 4 | cross-validation, on the training set only |
| 5 | could a curve do better? Same folds |
| 6 | test the winner, once |

::: notes
The six steps are the workflow of lessons 1 to 5 in order, and every one of
them has a mistake the course warned about: the wrong baseline, a split that is
not stratified, imputing before splitting, the test set inside the
cross-validation, choosing on the test set.

The checks catch each of those. When someone gets a cross, point back to the
lesson that covered it rather than giving the fix.
:::

# Before you start

- Start the course container and open `Workshop/workshop.ipynb`
- Run the first cell: it loads the data
- `NameError: name '___' is not defined` means a blank is still empty
- Do **not** open `delivery_data.py`: it holds the answer

::: notes
Wait until every laptop has run the first cell before starting step 1. A
container that does not start is the one problem that cannot be solved later in
the session.

Pairs are fine, but both people should type: the point is for each of them to
see what they can do.
:::

# How did it go?

- Hands up for each step that printed ✓
- Where the crosses were, and which lesson that step came from
- The ± of step 4 and step 5: is the difference real?

::: notes
Go through the six steps and ask for a show of hands at each, so that both you
and they see where the room stands. The step with the fewest ticks is the one
to discuss.

Then the comparison of step 5: 0.858 against 0.762, with spreads of about 0.02.
The gain is several times the spread, so it is real. Ask what they would have
needed to see to call it noise.
:::

# The rule, revealed

- Longer routes, fuller vans and rain make a parcel **later**
- Driver experience makes it **earlier**
- Departure time works through rush hour: the further from 13:00, the later
- The age of the van has **no effect at all**

::: notes
The rule is a logistic one, with the departure hour entering as (hour - 13)
squared. The guide has the formula with its coefficients.

Two points to make. The model with squared terms has exactly the shape of the
rule, which is why it won; on real data nobody hands you the rule, and the
cross-validation of step 5 is how you would have found out. And the errors that
remain, about 13%, are not a failure: the rule gives a probability, so even a
perfect model gets some parcels wrong.

Close by saying that what they did today, from baseline to one look at the test
set, is the shape of every exercise they will talk through at the exam.
:::
