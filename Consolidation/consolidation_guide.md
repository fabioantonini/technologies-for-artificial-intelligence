---
title: "Consolidation: the scikit-learn API, and every lesson's atoms"
subtitle: "Technologies for Artificial Intelligence — self-study, and two hours in class"
author: "Fabio Antonini — Università degli Studi dell'Aquila"
date: "Not examinable"
---

## What this is

Lessons 1 to 5 ran every scikit-learn call on hundreds or thousands of rows: 569
tumours, 2,000 customers, 600 houses, 8,000 disk drives. At that size the result of a call is visible, and what the
call *did* is not, and the lesson notebooks often take in one line a step that someone
new to machine learning needs to see on its own. The atoms take those steps one at
a time, on two to ten rows, small enough that every number can be checked with a
pencil.

This folder holds the notebook that runs across the lessons, `00_the_scikit_learn_api.ipynb`.
Each lesson's own atoms live inside that lesson, in `Lessons/NN_*/Notebooks/atoms.ipynb`,
and each lesson's notebook guide lists them, with the notebook they can replace in class.
In this guide `02·9` means lesson 2's atoms notebook, atom 9, and `00·3` atom 3 of the API
notebook.

Each section of a notebook is an **atom**: one idea, one call. Many atoms recompute
the worked examples of the lesson 2, 3 and 4 commentaries
(`Docs/*_commentary.md`) with the library, so that the arithmetic done by hand and
the call that does it sit side by side; the numbers are the same.

An atom is here for one of two reasons: it shows a step that the lesson notebooks
take without stopping on it (what `fit` stores, why `transform` and not
`fit_transform` on the test set, what `predict` does with a probability), or it
computes a worked example of a lesson's commentary or slides with the library call
that does it. An idea the lesson notebook already shows in full is not repeated here.

It is **not examinable**, and it adds no new material. It does not replace the
lesson notebooks either: every atom names the slide and the notebook section where
its idea is treated at full scale.

Reading time: about 40 minutes for the API notebook and for most lessons' atoms, and 15
for lesson 1's.

## How to use an atom

Every atom has the same shape:

1. **What it shows**, in a sentence or two, and *From:* the lesson slide (by title)
   and notebook section it comes from.
2. **The data**, tiny, printed.
3. **Your prediction, before you run the next cell**: one concrete question.
4. **The code**: a few lines, with the call being taught in plain sight.
5. **What to look at**: what the output means, in the units of the problem.
6. **Try changing**: one small change to make and run.

The third step is the one that matters. Write the answer down, even as a single
number, *before* running. A prediction that turns out wrong marks the exact place
where your picture of the library differs from what it does, and that is worth more
than any number of predictions that came out right.

## The files

| Notebook | What it covers | Atoms |
|-------------------|----------------------|--|
| `00_the_scikit_learn_api.ipynb` | `fit`, `transform`, `fit_transform`, `predict`, `predict_proba` and the threshold, `score`, the shape of `X`, `random_state`, `Pipeline`, `decision_function`, the thresholds behind a curve, the defaults `get_params()` shows, `kneighbors`, the fitted attributes of `GaussianNB` and `SVC`, `cross_validate` with training scores | 15 |
| lesson 1, `Notebooks/atoms.ipynb` | splitting and `stratify`, the baseline, the confusion matrix, precision and recall | 4 |
| lesson 2, `Notebooks/atoms.ipynb` | looking, missing values, outliers, scaling, encoding, `ColumnTransformer`, the area under the curve, two leaks | 15 |
| lesson 3, `Notebooks/atoms.ipynb` | cost, the normal equation, gradient descent, Ridge and Lasso, what a coefficient means | 15 |
| lesson 4, `Notebooks/atoms.ipynb` | odds and log-odds, log loss, one gradient step, the confusion matrix in order, the two averages, AUC and average precision, a threshold from costs, class weights, softmax, the whole chain on one fitted model | 12 |
| lesson 5, `Notebooks/atoms.ipynb` | cross-validation round by round, the five round models, the spread and the report line, stratified, repeated and grouped folds, training against validation error, `GridSearchCV`, nested cross-validation one round at a time, a fresh test set, the threshold sweep, seeds | 16 |
| lesson 6, `Notebooks/atoms.ipynb` | one pump's five nearest and the vote, scaling first, training accuracy inside the folds, distances that stop varying, choosing columns, Naive Bayes's ten numbers and three ratios, the two log totals, uncorrelated but not independent, the interaction and its product, confidence, the margin in one dimension, slack and the weights, the kernel trick and the RBF similarity, one pump's vote by hand, γ and C | 15 |
| lesson 7, `Notebooks/atoms.ipynb` | Gini by hand, the gain by counting, thresholds and midpoints, a tree on ten loans and one path, depth against a budget of leaves, the noise ceiling, five folds by hand, sampling with and without replacement, bagging and out-of-bag by hand, one column per split, importance and permutation by hand, boosting by hand | 14 |
| lesson 8, `Notebooks/atoms.ipynb` | standardising first, k-means by hand, starts and the number of groups, the silhouette, the adjusted Rand index (ARI) by pairs, Ward's merges, DBSCAN by counting, the widest shadow three ways, reconstruction error and 37 of 40 | 16 |
| lesson 10, `Notebooks/atoms.ipynb` | Keras layers with weights set by hand, no training: one window by hand, Keras as cross-correlation, the output size, a contrast kernel and its threshold, shifting and pooling, the receptive field, parameter counts, the grader's error rate, shuffled pixels, augmentation, what transfer copies | 15 |

They run in the course container like every other notebook, each in a few seconds,
with no network. Start with `00`: the lessons' atoms assume it. Lesson 9's atoms are still to come.

## The atoms

Italic titles are worked examples of the lesson's commentary: same data, same
numbers.

### 00 — the scikit-learn API

| # | Atom | Lesson source |
|--|--------------------------|------------------------|
| 1 | Estimators learn in `fit`, and keep what they learned | L1 *Step 5: pipeline, not two steps*; L2 *What to do about a gap* |
| 2 | `transform` applies what `fit` learned; `fit_transform` does both | L2 *StandardScaler vs MinMaxScaler* |
| 3 | `predict` gives a label, `predict_proba` the probabilities behind it, and 0.5 sits between them | L1 *Behind every label, a probability*; *The trade-off, drawn* |
| 4 | `score` is a metric with a default | L1 *Accuracy: the share of right answers*; L3 *A cost needs something to be compared against* |
| 5 | Why arrays, and why `X` has two dimensions | L1 notebook 2; L3 *$\theta$ and X: all parameters, all houses* |
| 6 | `random_state`: the same randomness, every time | L1 *One split is one measurement* |
| 7 | `Pipeline`: one `fit`, and every step fitted on the right rows | L1 *Step 5*; L2 *ColumnTransformer + Pipeline* |
| 8 | The contract, in one table | lessons 1 to 3 |
| 9 | `decision_function`: the score before the sigmoid | L4 *Log-odds of p = P(y = 1 \| x): a line*; *Three classes: softmax (optional)* |
| 10 | `roc_curve` and `precision_recall_curve` return the thresholds too | L4 *The precision-recall curve*; *The ROC curve* |
| 11 | `get_params()`: the defaults you did not choose | L4 *And it is convex, which squared error is not*; L4 notebook 1, section 6 |
| 12 | k-NN: `fit` stores, `kneighbors` finds, `predict` votes | L6 *k-nearest neighbours: the whole algorithm*; *What k-NN costs* |
| 13 | Naive Bayes: `theta_`, `var_`, `class_prior_`, and the log scores | L6 *Training computes ten numbers*; *From the two totals to probabilities* |
| 14 | SVM: the support vectors are attributes | L6 *The SVM's solution: a weighted sum of pumps*; *Training: one weight per pump* |
| 15 | `cross_validate`: the training score beside the validation score | L6 *Training against cross-validated accuracy, by k* |

## Two hours in class

A selection of thirteen atoms, run together from the projector. For each one: read the
question aloud, give the room a minute to commit to an answer (a show of hands works
for most), run the cell, then ask who was surprised and why. The surprise is the
point; the explanation follows from it.

| Minutes | Notebook · atom | What it settles |
|-----|--------|----------------------------|
| 5 | — | how an atom works: predict, then run |
| 8 | 00 · 1 | what `fit` learns and where it keeps it |
| 10 | 00 · 2 | `transform` against `fit_transform` on the test set |
| 10 | 00 · 3 | `predict` is a probability compared with 0.5, and moving the 0.5 |
| 7 | 00 · 7 | a pipeline fits each step on the training rows |
| 8 | 01 · 1 | what `stratify` keeps, on eight patients |
| 12 | 01 · 3 and 4 | the confusion matrix, then precision and recall from it |
| 10 | — | break |
| 10 | 02 · 9 | three encodings of ten customers |
| 7 | 02 · 12 | the area under the curve as a count of pairs |
| 12 | 02 · 14 and 15 | the target-encoding leak, and the fix |
| 8 | 03 · 4 | the normal equation on three houses |
| 10 | 03 · 8 | gradient descent, one iteration at a time |
| 3 | — | close: which prediction surprised you most? |
| **120** | | |

Everything not in the selection is left for self-study; each lesson's notebook guide
is where to find it.

## What students will get wrong, and why it is reasonable

- **"`fit_transform` on the test set is the same thing, just shorter."** The name
  invites it: if `fit_transform` is `fit` plus `transform`, using it everywhere looks
  like tidiness. On the training set it is; on the test set it learns from the rows
  being measured. Atom 00·2 shows a test customer at 2.24 standard deviations above
  average becoming exactly average.
- **"`predict` is the model's answer."** It is the model's probability compared
  with 0.5, a threshold nobody chose for the problem at hand, and it is natural to
  take a default for a decision. Atom 00·3 shows a malignant tumour at probability
  0.21 that the default never flags, and what a threshold of 0.2 costs to catch it.
- **"A higher R² means a better model."** Only on the same test set. Atom 03·3 shows
  the model's error falling by more than two thirds while R² falls from 0.971 to
  0.625, because the second test set has less spread to explain.
- **"`TargetEncoder` is target encoding, so it leaks."** It is the one class in
  scikit-learn where `fit_transform` is not `fit` followed by `transform`: it
  cross-fits, and atom 02·15 shows the two giving different columns. The leak comes
  back only if `transform` is called on the training rows by hand.
- **"`tn, fp, fn, tp` or `tp, fp, fn, tn`, it is the same four numbers."** The four
  names look interchangeable, and the matrix is small. But `.ravel()` reads the matrix
  row by row, truth in rows, so the order is fixed: TN, FP, FN, TP. Atom 04·5 shows that
  unpacking in the wrong order raises no error and swaps every metric.
- **"`LogisticRegression()` fits the plain log loss."** Nothing in the call says
  otherwise, and lesson 4 derived the plain log loss, so the reading is natural. But
  the default adds an L2 penalty with `C=1.0`. Atom 00·11 shows it shrinking a
  coefficient from 0.971 to 0.685 on eight rows, a gap that almost vanishes on
  6,000, and `get_params()` showing every default before it matters.
- **"A high AUC means the probabilities can be trusted."** The AUC is presented as the
  best single summary of a classifier, so it is natural to read it as a guarantee.
  Atom 04·7 divides every score by ten: the AUC stays 0.792 while every decision at 0.5
  changes, because the AUC sees only the order.
- **"`cross_val_score` returns the error."** For a regression it returns the error with
  a minus sign: its scorers are all "higher is better", so `neg_mean_squared_error`
  makes a larger error a smaller number. The name says so, and it is still easy to read
  past. Atom 05·1 shows five MSEs from 274.6 to 1,569.0 kWh² coming back as −274.6 to
  −1,569.0.
- **"`best_score_` is the AUC of the model `GridSearchCV` hands back."** The refit is the
  last thing the search does, so it is natural to think the score belongs to it. But the
  score is the winner's mean over the folds, computed before the refit, and the refitted
  model is never scored. Atom 05·9 shows `best_score_` equal to the highest of the 25
  means, 0.7999, on data with no signal at all.
