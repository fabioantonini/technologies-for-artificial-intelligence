---
title: "Exercise 10 — Two Stations, One Photograph"
subtitle: "Technologies for Artificial Intelligence — Lesson 10"
author: "Fabio Antonini — Università degli Studi dell'Aquila"
date: "Set 27 November 2026 · the last one, and the only one no lesson follows: work it by Friday 4 December 2026"
---

## What this is

The last exercise of the course. It is deliberately smaller than exercise 9,
because the week it falls in is the week of the project peer review and that
has the stronger claim on your time. Budget **two to three hours**, and if it
is taking longer than that you have misread something — say so in your report
rather than pushing on.

Everything runs on the container's central processing unit (CPU) in well under
five minutes of fitting. Nothing here needs a graphics card, a download, or a
pre-trained network from anywhere but your own script.

**The dataset:** `window_inspection_data.py`, beside this file. Read its module
docstring before anything else — for this exercise that docstring is half the
brief, and Part 1 is answerable from it alone.

**One rule about the file, and it matters:** the module has a banner reading
`PART 4 STARTS HERE`. Everything below that line gives away the answer to
station 2. **Do not read past it until you reach Part 4.** You are on your
honour; the exercise is worth nothing to you if you break it, and it will be
obvious in your Part 1 reasoning if you did.

---

## The setting

Meridian Instruments coats the glass windows its sensors look out through.
Every window is photographed after coating, and the photograph is cropped to a
24 × 24 patch covering the whole disc.

**Two stations on the same line ask two different questions of that one
photograph.**

- `load_coating_grade()` — station 1, goods-inward: *is this window blemished
  at all?*
- `load_aperture_grade()` — station 2, salvage: *the window is blemished; can
  it still be shipped?*

Both return `(X, y, truth)`. `X` is the photographs, `y` is the **recorded**
grade — the only labels a model may ever see — and `truth` is the noise-free
grade, which exists for Part 4 and for nothing else. **Never fit on `truth`,
never select on it, and never score against it before Part 4.**

The two batches are matched on purpose: same size, same image, same base rate,
same 3% grading error, the same blemish drawn by the same code. The only thing
that differs is what the station is asking.

---

## Part 1 — Predict, before you fit anything (20 marks)

1. Load both batches. Report, for each: the number of images, the reject rate,
   the majority-class baseline, and — using `truth` **only for this one
   count** — the fraction of grades the bench recorded wrongly, and therefore
   the ceiling on that batch. (4 marks)

2. Look at the data without fitting any model. At minimum, display a handful
   of accepted and rejected windows from **each** station side by side. Say in
   two or three sentences what visibly distinguishes the two classes at
   station 1, and what visibly distinguishes them at station 2. One of these is
   much harder to see than the other, and noticing that is part of the answer.
   (6 marks)

3. Now commit, in writing, to a prediction for **each station**, answering all
   three of these:

   - Will a dense network of the kind lesson 9 built beat the majority
     baseline? By how much, roughly?
   - Will the convolutional network of handout section 5 — two convolutional
     layers, then a **global** max pooling — beat the dense one?
   - Is there any reason the answer to the second question might differ
     between the two stations?

   (10 marks)

**What is being marked here is the reasoning, not the prediction.** A
confident wrong prediction that argues from what the two questions actually
ask earns nearly full marks. A hedged "it depends on the data" earns very few.
Commit.

---

## Part 2 — Measure it properly (25 marks)

4. For each station, train and score three models on the recorded labels:

   - the dense network of lesson 9 — flatten, two layers of 256, one sigmoid
     output;
   - the convolutional network of handout section 5, ending in
     `GlobalMaxPooling2D`;
   - **the same convolutional network with `GlobalMaxPooling2D` replaced by
     `Flatten`**, and nothing else changed.

   Use one held-out test batch per station, generated with a seed you did not
   train on. Report accuracy for all six combinations. (12 marks)

5. **Run each of the six at three network initialisations** and report the mean
   and the spread, not a single number. One of the six has a spread several
   times larger than the others; find it and say which. (7 marks)

6. State plainly, station by station, which model wins and by how much
   **against the spread you just measured**. Where two are not distinguishable,
   say so. (6 marks)

**What is being marked:** that the test batch is genuinely held out, that the
three architectures differ only where the brief says they differ, and that
every comparison is made against a measured spread rather than by eye.

---

## Part 3 — Explain it (30 marks)

Something inverts between the two stations, and it is not the thing most people
expect. Explain it, in six to ten sentences per station.

**Support every claim with a measurement from your own run rather than with an
assertion.** A paragraph that says "station 2 depends on position, so pooling
hurts" is worth almost nothing on its own: it restates the outcome in different
words. The measurements below are worth what they cost, and a better instrument
you thought of yourself is worth more than three of these.

- *For a claim about which assumption is doing the work.* Run the permutation
  test of handout section 9 on **both** stations, using
  `pixel_permutation()`. Apply the same permutation to every image, retrain,
  and report the cost to each model at each station. Handout section 9 found
  that permutation cost the convolution nothing on one task and 4.5 points on
  another; say which of your two stations behaves which way, and why.

- *For a claim about position.* Station 1's loader takes a `band` argument, so
  the translation test of handout section 6 is available there: train with
  blemishes confined to `TRUE_TOP_BAND`, test on `TRUE_BOTTOM_BAND`. Run it for
  the dense network and the pooled convolutional one. Then say — in one
  sentence — why the same experiment **cannot** be run at station 2, and what
  that impossibility already tells you.

- *For a claim about what pooling costs.* You have already measured the pooled
  and the flattened head. Report the difference at each station and explain the
  sign of it. Handout section 4.2 lists three things pooling does; say which of
  the three is responsible.

- *For a claim about capacity.* Count the parameters of all three
  architectures. Two of them differ by a factor of about seven and the larger
  is not always better; use that to argue that what separates the models here
  is **not** how much they can express.

**What is being marked:** this part carries the most marks, and the
measurements are what separate an explanation from a plausible story. The
strongest available answer identifies **which single layer** is responsible for
the inversion and demonstrates it by changing that layer alone.

---

## Part 4 — Mark yourself against the rule that made the data (25 marks)

Now, and not before, read past the `PART 4 STARTS HERE` banner in the module.

7. `TRUE_APERTURE_CENTRE` and `TRUE_APERTURE_RADIUS_PX`, with
   `_inside_aperture`, give station 2's rule exactly. Write it out in one
   sentence. Then say whether your Part 1 prediction was right, wrong, or right
   for the wrong reason — being specific about which of the three questions in
   Part 1.3 you got right and which you did not. (8 marks)

8. `aperture_mask()` returns the rule as a 24 × 24 boolean image. Use it to
   build a station-2 classifier that has **no parameters and needs no
   training**: locate the brightest local-contrast response in each photograph,
   look up that position in the mask, and grade accordingly. The contrast
   kernel of handout section 3.3 is the tool. Report its accuracy against the
   recorded grade and against `truth`, and compare both with your best trained
   model.

   **If your accuracy comes out near chance, do not conclude the rule is
   wrong — inspect where your classifier thinks the blemish is.** A
   correlation function that pads the border with zeros can find a stronger
   "response" at a corner than at an actual blemish, on an image whose
   brightness is never negative. `TRUE_MARK_MARGIN`, already used above, is
   the fix. Diagnosing this yourself is worth real credit; being stuck on it
   for more than fifteen minutes is a sign to re-read that constant's
   docstring. (10 marks)

9. Check the ceiling identity of handout section 5.1 on your best model at each
   station. If a model agrees with the true grade on a fraction $q$ and the
   bench records the wrong grade with probability $e = 0.03$, its accuracy
   against the recorded grade should be $q(1-e) + (1-q)e$. Report predicted
   and observed, per station. Then answer in one sentence: at the station where
   your best model scores lowest, how much of the shortfall belongs to the
   model and how much to the bench? (7 marks)

**What is being marked:** question 8, which is the whole lesson in one
construction — a rule you can write down beat everything you trained, because
it encodes the right assumption exactly rather than approximately.

---

## Marks, and what loses them

| Part | Marks |
|---|---|
| 1 — predict before fitting | 20 |
| 2 — measure it properly | 25 |
| 3 — explain it | 30 |
| 4 — against the generating rule | 25 |
| | **100** |

**You lose marks for:**

- fitting or selecting on `truth`, or scoring against it before Part 4 (**−15**);
- reading past the `PART 4 STARTS HERE` banner before Part 4, which will show
  in Part 1 (**−15**);
- reporting a single run's accuracy where Part 2 asks for a mean and a spread;
- asserting a claim in Part 3 with no supporting number from your own run;
- changing more than one thing between two architectures you then compare.

**You do not lose marks for:**

- a wrong prediction in Part 1, provided it engaged with all three questions;
- **reporting that two models are not distinguishable at this sample size and
  declining to name a winner.** If the spreads genuinely overlap, that is the
  correct answer and it earns full marks. This course has spent ten weeks on
  the difference between a result and a draw; do not throw that away in the
  last week to make a tidier table;
- an architecture that underperforms, if you say so and diagnose it.

## If you are stuck

Handout section 5 has the architecture, layer by layer, with its parameter
counts. Section 6 is the translation test. Section 9 is the permutation test
and what it separates. Section 4.2 is what pooling does and what it costs.
Notebook 02 runs every one of these on the wafer images, and the code
transfers with the dataset swapped.

Keep one notebook, `exercise10.ipynb`, executed top to bottom, with the prose in
markdown cells. Nothing is handed in, and no lesson follows to discuss this one:
it is worked for the exam, where one of the ten exercises is drawn for you to talk
through your own notebook.
