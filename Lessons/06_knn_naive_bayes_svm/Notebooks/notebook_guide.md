---
title: "Lesson 6 — What Each Notebook Shows"
subtitle: "Notebook guide, lesson 6 — Technologies for Artificial Intelligence"
author: "Fabio Antonini — Università degli Studi dell'Aquila"
date: "6 November 2026 · three notebooks"
---

> A short guide to this lesson's notebooks. For each one: what it is for, the
> concepts in the order they appear, the numbers worth noticing, and **the two-minute
> version**, a summary to read before running it or to come back to afterwards.
> Every number here comes from the committed output of the notebook it belongs to.

---

# Notebook 01 — `01_knn_and_the_curse`

## k-nearest neighbours, and where distance stops working

**What it is for.** Every model so far learned a rule — coefficients, a boundary, a
threshold. This one learns nothing at all, wins anyway, and is then broken on purpose.

**The data.** 1,200 pumps, two readings each, **735 faulty (61.3%)**, with the healthy
ones in a central blob and the faulty ones surrounding it.

**The concepts, in order.**

1. **Look at it first**, and see that **no straight line separates the classes**. A
   pump can fail by running too slow *or* too fast.
2. **What that costs a linear model.** Majority baseline **0.613**; logistic regression
   **0.613 ± 0.000**. Not slightly beaten — it learned *nothing*, and predicts "faulty"
   for every pump.
3. **k-NN written out in eight lines**, agreeing with scikit-learn on **100% of the
   test points**, accuracy 0.937.
4. **Choosing $k$**, the bias-variance dial made visible:

   | $k$ | training | cross-validated |
   |---|---|---|
   | 1 | **1.000** | 0.912 |
   | 5 | 0.953 | **0.944** |
   | 51 | 0.940 | 0.933 |
   | 401 | 0.828 | 0.708 |

   At $k = 1$ the training accuracy is exactly 1.000 **by construction** — every point
   is its own nearest neighbour — which is the clearest demonstration that a training
   score can measure nothing at all.
5. **The three boundaries.** Ragged with islands at $k = 1$, a clean disc at $k = 15$,
   and swollen past the true envelope at $k = 401$.
6. **Now break it.** Add columns of pure noise, leaving both real readings untouched:

   | noise columns | accuracy | above baseline |
   |---|---|---|
   | 0 | 0.938 | +0.325 |
   | 10 | 0.762 | +0.149 |
   | 50 | 0.602 | **−0.011** |
   | 100 | 0.578 | −0.035 |

   **The signal never left**, and ten useless columns cost eighteen points.
7. **Why: geometry, not k-NN.** The ratio of nearest to farthest distance —
   **0.016** in 2 dimensions, 0.263 at 10, **0.701 at 100**, 0.855 at 500. Once every
   point is about as far as every other, "nearest" stops being a statement.
8. **What it costs to use.** No training at all, and $O(mn)$ per prediction — plus the
   dataset itself has to travel with the model.

**The two-minute version.** *Two readings per pump, healthy in the middle and faulty
all around, so no line works: logistic regression scores exactly the base rate, 0.613.
k-NN, which learns nothing, scores 0.944 with k = 5. At k = 1 the training accuracy is
1.000 by construction and the honest score is 0.912 — a perfect training score meaning
nothing at all. Then we break it: adding ten columns of pure noise, with both real
readings untouched, costs eighteen points of accuracy, and by fifty columns k-NN is
worse than always guessing "faulty". The mechanism is geometric — in two dimensions the
nearest point is 2% as far as the farthest, in a hundred it is 70%.*

---

# Notebook 02 — `02_naive_bayes`

## Naive Bayes, and the assumption it is named for

**What it is for.** k-NN made no assumptions and paid for it in dimensions. This is the
opposite bargain: one very strong assumption, in exchange for training in a single pass
and working in thousands of dimensions. The notebook shows it succeeding, then failing
**one step away** — and explains both.

**The concepts, in order.**

1. **Bayes' rule, and where the difficulty is.** The prior and the denominator are
   easy; the joint likelihood $P(x \mid y = c)$ is the problem.
2. **The naive assumption** — given the class, the features are independent — which
   replaces one $n$-dimensional estimate with $n$ one-dimensional ones.
3. **On the pumps it does very well:** **0.933 ± 0.010**, one point behind k-NN and a
   long way ahead of logistic regression's 0.613.
4. **And then the check that makes it a lesson rather than a result.** The assumption
   is about *within*-class dependence: overall −0.046, **within healthy −0.006, within
   faulty −0.049.** Uncorrelated — but not independent. Healthy pumps fill a disc and
   faulty ones a ring, and on a ring a pump far out in vibration must be central in
   pressure; squaring the distances from the design point shows it, **+0.168** healthy
   and **−0.420** faulty. **The assumption is false here**, and why it works anyway is
   item 6.
5. **What training actually computed.** Gaussian Naive Bayes puts a bell curve on
   every feature in every class, so training is priors plus a mean and a variance per
   feature per class — **ten numbers** on the pumps, one pass, no gradient descent. The
   table carries the surprise: the classes have **the same centre** (41.986 against
   42.015 Hz) and different **widths** (sd 1.695 against 4.636 Hz; 0.219 against 0.576
   bar). That is why logistic regression is stuck at the baseline and Naive Bayes is
   not. One pump taken apart term by term: at 48 Hz and 5.6 bar the log scores are
   −8.095 healthy and −4.144 faulty, **P(faulty) = 0.981**, matching `predict_proba`;
   at the design point, **P(healthy) = 0.820**. The positive log P(pressure), +0.598,
   is a density above 1, not an error.
6. **Wrong densities, nearly right boundary.** Walking out from the centre until the
   prediction flips puts the model's edge at 3.152 and 3.191 Hz, 0.413 and 0.411 bar,
   against a true envelope of 3.50 Hz and 0.45 bar: an axis-aligned ellipse of the true
   shape, about a tenth small. With shared centres, comparing a narrow bell to a wide
   one can only give such an ellipse — which is why the false assumption costs almost
   nothing: 0.933 against k-NN's 0.944, under the 0.96 noise ceiling.
7. **One step away, worse than guessing.** A second dataset where the pump is faulty
   when **exactly one** reading is high. Each sensor alone correlates +0.081 and −0.064
   with the label — essentially nothing. The redrawn figure shows why: per sensor both
   classes have the same two humps (quadrants 273/278 healthy, 359/253 faulty), and the
   one bell per class Naive Bayes fits sits in the valley, nearly identical for the two
   classes (sensor A: centre −0.012 and +0.167, sd 1.090 and 1.112). The 0.17 between
   centres is a sampling accident. Results: baseline 0.523, **Naive Bayes 0.404**, logistic regression 0.393, linear support vector machine (SVM)
   0.606, k-NN 0.967, RBF SVM 0.972. k-NN and the RBF SVM prove the data is easily
   learnable: **the problem is the hypothesis class, not the data.**
8. **The assumption failing, measured.** Section 3's test on these sensors gives the
   mirror image: overall **−0.057**, but **+0.821 within healthy** and **−0.826 within
   faulty**. Independent-looking overall, tightly tied inside each class — exactly what
   the model assumes away.
9. **It cannot even tell that it is lost.** Mean confidence **0.567 when right, 0.555
   when wrong**, highest 0.673. Equally unsure either way, which is honest and useless:
   the reported probability carries no information about whether to trust the
   prediction.
10. **When to reach for it anyway.** Very high dimensions with little data — text
   classification being the canonical case.

**The two-minute version.** *Naive Bayes takes the opposite bargain from k-NN: one very
strong assumption, and in exchange it trains in a single pass. On the pumps it scores
0.933, and rather than leave that as luck we check the assumption — within each class
the two readings correlate −0.006 and −0.049, but squared distances correlate −0.420
among the faulty pumps: uncorrelated, not independent, so the assumption is false.
Training is ten numbers, and they show why it works anyway: the two classes share a
centre and differ only in width, so the boundary comes out an ellipse of the true
shape, 3.17 Hz by 0.41 bar against 3.5 by 0.45. Then
one step away: a dataset where the fault is "exactly one
reading is high". Each sensor alone says nothing, and Naive Bayes only ever sees the
sensors one at a time — inside each class the two correlate +0.821 and −0.826 — so
it scores 0.404, below chance, while k-NN gets 0.967 on the
same data. Structural, not a data shortage. And its confidence is 0.567 when right and
0.555 when wrong, so it cannot tell you when to doubt it.*

---

# Notebook 03 — `03_svm_margins_and_kernels`

## Support vector machines: the widest street, and a change of coordinates

**What it is for.** The third answer to the same question. k-NN remembered, Naive Bayes
assumed; the SVM **changes the criterion** and then, with kernels, **changes the
space**.

**The concepts, in order.**

1. **The widest street.** On separable data every separating line has zero error, so
   the error cannot choose between them. The SVM takes the widest slab — and of **80
   points, 3 are support vectors**. Only a handful of points matter.
2. **Soft margins and the one knob.** Violations are allowed and charged for; $C$ sets
   the price, and note the direction — large $C$ means *less* tolerance and less
   regularisation.
3. **On the pumps, a straight line still fails.** Linear SVM **0.613 ± 0.000**, RBF
   **0.947 ± 0.005**. And the support-vector counts say why in a second language:
   **947 of 1,200 (79%)** for the linear kernel against **278 (23%)** for the RBF. A
   boundary pushing through a crowd is a boundary in the wrong space.
4. **The kernel trick.** Add a third coordinate — distance from the design point — and
   the disc-inside-a-ring becomes two layers a plane can separate. Then the idea that
   makes it famous: the dual depends on the data only through inner products, so a
   kernel computes those in the transformed space **without ever constructing the
   coordinates**. Checked by hand: for $\phi(x) = (x_1^2, \sqrt{2}x_1x_2, x_2^2)$,
   $a = (1, 2)$, $b = (3, 1)$, lifting gives $9 + 12 + 4 = 25$ and never lifting gives
   $(a \cdot b)^2 = 5^2 = 25$.
5. **Training and inference, step by step.** Training solves for one weight per pump,
   two at a time, no learning rate: **922** at weight 0, **268** at the ceiling $C = 1$,
   **10** on the edge — 278 support vectors — and $b = +1.105$. Inference on the 48 Hz,
   5.6 bar pump: faulty votes **+37.455**, healthy **−36.677**, intercept +1.105, total
   **+1.883**, exactly `decision_function`: faulty. The 56 support vectors within half
   similarity give a net +1.141, the other 222 −0.363 — the neighbourhood decides,
   though the five nearest disagree. k-NN with a smooth similarity and learned weights.
6. **Gamma, C, and what they do to the boundary.** Similarity halves at 2.63, 1.18,
   0.83 and 0.12 standard deviations for $\gamma$ = 0.1, 0.5, 1 and 50:

   | setting | training | cross-validated |
   |---|---|---|
   | $\gamma$ = 0.1, C = 1 | 0.936 | 0.929 |
   | $\gamma$ = 1, C = 1 | 0.950 | **0.944** |
   | $\gamma$ = 50, C = 1000 | **0.995** | 0.902 |

   The best training score is the worst honest one — the signature lesson 5 taught them
   to read, now drawn as bubbles around individual mislabelled points.
7. **The three methods, side by side**, with the noise ceiling at about 0.96 because 4%
   of labels are flipped: baseline 0.613, logistic regression 0.613, linear SVM 0.613,
   **Naive Bayes 0.933, k-NN 0.944, RBF SVM 0.947.** Three reach the ceiling by three
   different routes; two do not, and both are linear.

**The two-minute version.** *The SVM changes the criterion: among the many lines that
separate the data, take the one with the widest slab around it — and on the example
only three points of eighty decide it. On the pumps the linear SVM still scores the
base rate, because the problem was never which line to choose, and 79% of the points
end up as support vectors, which is what a boundary forced through a crowd looks like.
Then the kernel trick: add distance-from-centre as a third coordinate and a plane
separates the classes, and the kernel computes those inner products without ever
building the coordinates — 25 both ways on a two-point example. Training finds one
weight per pump, 922 of them zero; a prediction is the 278 support vectors voting by
similarity, +1.883 for the 48 Hz pump. RBF scores 0.947 against a noise ceiling of 0.96. The last
table is the lesson: three families reach the ceiling by three different mechanisms,
the two linear models sit exactly on the baseline, and the gap is 0.334.*
