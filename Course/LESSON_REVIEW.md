# Reviewing a finished lesson

`tools/verify_lesson.py` decides whether a lesson is *built*. This document is about
whether it is *right*, which is a different question and is not automatable in full.

Every check below earned its place by catching something real in lesson 1 during the
review of 28 August 2026. The examples are kept deliberately: a check without a
failure it once caught tends to be skipped.

Work in this order. The mechanical checks are cheap and narrow the field; the
judgement ones need the built PDF open in front of you.

---

## 1. Mechanical

### 1.1 No figure sits directly under a heading

Every figure must follow the paragraph that introduces it, so a reader meets the
argument before the picture. A section that opens with a stack of images is a section
whose figures were never placed.

> **Caught in lesson 1:** thirteen of twenty-one figures. Section 7 opened with five
> pictures in a row, then the prose explaining them.

```bash
python tools/review_lesson.py NN        # reports every figure under a bare heading
```

### 1.2 Every number in prose appears in a committed notebook output

The handout and slides quote numbers the notebooks produce. Nothing but this check
ties them together — `verify_lesson.py` compares `worked_examples.py` against the
handout, and both can agree while the notebook beside them says something else.

> **Caught in lesson 1's neighbours:** lesson 2's handout said AUC 0.752 where its
> notebook had long printed 0.751, and lesson 3's headline 247,514 had become
> 3,097,038,010 under the shipped image. Both lessons passed verification throughout.

### 1.3 Notation follows the table in CLAUDE.md

`m` is the number of examples, `n` the number of features, `α` the learning rate, `λ`
the penalty. Slips are invisible one document at a time.

> **Caught in lesson 1:** slide notes said the standard error "falls as one over root
> n" where they meant `m`; the quiz wrote the empirical risk as `(1/n) Σ`.

### 1.4 No em dashes in anything projected

Slide bodies, slide titles, metadata — and **inside figures**, where the text is drawn
into the PNG and no search of the markdown will find it. Handouts keep theirs: an em
dash in a typeset PDF is ordinary publishing.

> **Caught across the course:** 839 in the decks, then a further 41 drawn inside
> figures, which the first pass could not reach.

### 1.5 The lesson plan's slide ranges match the segments

`verify_lesson.py` checks the cited slides *exist*, not that they are the right ones.

> **Caught in lesson 1:** the plan cited 9–22, 23–30, 31–39, 40–51 for segments that
> actually began at 21, 31, 37 and 50.

### 1.6 Every notebook is cited in the handout body, with its numbers

Not only in the lesson plan table. A notebook nobody quotes is a notebook nobody runs.

> **Caught in lesson 1:** notebook 02 appeared in the plan table, one homework line and
> three sets of speaker notes — and nowhere else. Handout section 5 had no numbers at
> all, the only section in the lesson arguing entirely in the abstract.

---

## 2. Judgement — the built PDF open, not the markdown

### 2.1 Open every figure and check the caption describes *that* image

The most productive check in the whole list, and the one no tool can do.

> **Caught in lesson 1:** three captions described figures that had never been made.
> `error_costs.png` was promised as "two problems side by side" and shows one.
> `overfitting.png` was captioned as two error curves and shows three polynomial fits.
> A slide's notes said "follow the two curves" of a histogram.

### 2.2 Speaker notes point at the slide they are on

A note describing a figure that is on the *next* slide is the standard failure, because
notes get appended while the author is thinking about what comes after.

> **Caught in lesson 1:** five of them.

```bash
python tools/review_lesson.py NN        # flags deictic notes with no figure beside them
```

### 2.3 Slide titles are true, and carry the argument

> **Caught in lesson 1:** "The same idea, on real data" over a figure fitted to 22
> synthetic points. Retitled to what its own notes tell the lecturer to point at.

### 2.4 Every term is defined before it is used

Students cannot look it up mid-lecture. Expand acronyms, and define words that look
ordinary but are not.

> **Caught in lesson 1:** `R²` used four times and never defined — nor anywhere else in
> the ten lessons. `n_init=10` with no explanation. "cultivar" used seven times.

### 2.5 No claim promises more than the cell delivers

> **Caught in lesson 1:** the self-supervised section said the learned representation
> "transfers to tasks you do care about" and demonstrated no transfer at all. The
> shortcut-feature section said the column "is empty" at deployment, implying an error
> that in practice never comes.

### 2.6 Criteria are stated at the moment they are applied

A test the reader can only run after the fact is not a criterion.

> **Caught in lesson 1:** "errors are catastrophic and unexplainable" as a reason *not
> to build* a model described a model already deployed and already failing.

### 2.7 Look at the rendered deck

```bash
soffice --headless --convert-to pdf deck.pptx
pdftoppm -png -r 55 deck.pdf slide
```

Overlapping annotations, a legend across a curve, a cropped axis label, a title
wrapping to two lines: none of these fail a check, and all are obvious in the render.

---

## 3. When a number moves

If re-running a notebook changes a number the material quotes, decide whether the
number is *stable* before reprinting it. A figure that depends on an ill-conditioned
solve will move again at the next image rebuild.

> **Lesson 3:** the largest coefficient of the degree-12 fit is deterministic inside
> the image and moved by a factor of 12,500 when the linear-algebra backend changed —
> the design matrix is full rank with condition number 4.2e9. The lesson's carry-home
> number was moved to one that had not budged.
