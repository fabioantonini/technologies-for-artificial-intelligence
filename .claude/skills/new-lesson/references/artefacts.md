# What each artefact contains

Read the section for the artefact you are writing. `CLAUDE.md` carries the
conventions that apply everywhere — audience, notation table, acronym rule, git
rules — and is not repeated here.

Naming, for a lesson `NN` on `TopicName` / `topic_snake_case`:

| Artefact | Path |
|---|---|
| Dataset module | `Notebooks/{something}_data.py` |
| Notebooks | `Notebooks/{01..}_{descriptive_snake_case}.ipynb` |
| Handout | `Docs/{topic_snake_case}.md` |
| Recomputation | `Docs/worked_examples.py` |
| Slides | `Slides/{topic_snake_case}_slides.md` |
| Quiz | `Quizzes/{TopicName}-Quiz.ipynb` |
| Exercise | `Exercises/{NN}_{topic_snake_case}.md` |
| Solution | `Exercises/{NN}_solution.ipynb` — **never committed** |
| Supplementary | `Resources/{descriptive_snake_case}.md` |

---

## Contents

1. [Dataset module](#1-dataset-module) — phase A
2. [Notebooks](#2-notebooks) — phase A
3. [Handout](#3-handout) — phase B
4. [worked_examples.py](#4-worked_examplespy) — phase B
5. [Slides](#5-slides) — phase C
6. [Quiz](#6-quiz) — phase C
7. [Exercise](#7-exercise) — phase C
8. [Supplementary reading](#8-supplementary-reading) — phase C
9. [Instructor solution](#9-instructor-solution) — phase C

---

## 1. Dataset module

A module beside the notebooks holding the loaders and, at the top, the `TRUE_*`
constants that record how the labels were generated.

```python
TRUE_BAND = tuple(range(18, 42))     # channels carrying any signal
TRUE_DEAD_CHANNEL = 26               # one photodiode contributing nothing
TRUE_LAB_ERROR_RATE = 0.03           # the ceiling no model can pass
```

Publishing the truth does three things. It lets the exercise ask a student to
check their conclusions against the generating rule, which is the most
instructive part of any exercise here. It makes calibration honest — you can
confirm the data does what you intend before a model sees it. And it puts a
**ceiling** on the lesson, so every score is read against something rather than
against 1.000.

Give each loader a docstring describing the physics or the process honestly and
at length: for the exercise, that docstring is the brief.

No network calls, no API keys, no paid services. Synthetic data is preferred over
downloads because it can be designed to make one point cleanly.

---

## 2. Notebooks

Two to four, numbered from `01`.

- **Cell 1** (markdown): `# {Title}`, then what this notebook shows and which
  handout section it implements.
- **Cell 2** (markdown): `## 1. Setup`
- **Cell 3** (code): a commented `pip install` line with versions **pinned to
  what the container actually has**, then imports, then the seed.

  Check them rather than recalling them — lesson 6's first draft pinned two
  versions that do not exist:

  ```bash
  docker exec tai_course python -c "import numpy, pandas, sklearn, matplotlib, scipy; print(numpy.__version__, pandas.__version__, sklearn.__version__, matplotlib.__version__, scipy.__version__)"
  ```

- Then alternate: markdown with the idea (2–4 sentences and the key formula),
  code implementing it, markdown on what to observe.
- Final cell: `## Summary`.

Rules:

- Runs top to bottom in a clean container.
- Real implementations on real or synthetic data — **never mocked results**.
- At least one notebook implements the core method **from scratch with NumPy**
  before showing the library version. This audience learns by building.
- Every figure the slides use is generated here, into `Figures/`.
- Outputs are committed, so a student sees what a notebook does before running
  it.
- Write them with `scripts/make_notebook.py`, which builds a notebook from a
  list of `(kind, source)` pairs. Keep the generator beside the notebook while
  drafting, named with a leading underscore so `lesson_state.py` does not read
  it as the dataset module, and delete it when the notebook is final.

---

## 3. Handout

The reference text, and **where the mathematics lives**.

Header, then the lesson plan, which must sum to **180 minutes**:

```markdown
| Time | Minutes | Segment | Material |
|---|---|---|---|
| 0:00–0:10 | 10 | Exercise N returned; the problem | Slides 2–6 |
| ... |
| | **180** | **Total** | **NN slides, 3 notebooks** |
```

Write the plan first and size the content from it: roughly 25–30 slides per
lecture hour, 20–30 minutes per notebook worked through live, one break. The
slide numbers must match the built deck — pandoc emits a title slide from the
front matter, so **page = heading index + 1**.

Then:

- Numbered `##` sections, `###` subsections, flowing prose that explains the
  *why*. No bullet-only sections.
- **Complete derivations** in LaTeX, written out. No "it can be shown that" —
  showing it is the point of this document.
- **Intuition before symbols.** Each derivation opens with plain language: what
  we are trying to do and why this approach is natural. Then the mathematics.
  Then what the result means in the units of the problem.
- **Every section ends in a concrete example** worked with real numbers from the
  lesson's dataset, ideally checkable against a notebook output.
- **Every figure the notebooks generated appears here**, with an italic caption,
  in the section that discusses it. Not "every figure the slides use" — the
  slides come later, so writing to that rule leaves orphans for Phase C to find.
  The slides-⊆-handout direction is enforced once both exist.
- State assumptions, and say where each result stops holding.
- **Name the predictable mistake** — and say why the wrong answer is reasonable.
  The second half is what makes it useful.
- **One number worth remembering**, repeated at the close.
- Two or three `> **Try this:**` boxes between sections.
- Close with a **Further reading** table of 4–6 entries.

At least 8 sections; 12 for a heavier topic.

---

## 4. worked_examples.py

Recomputes every number the handout works out by hand and asserts it against
what the handout prints.

The rule that gives it its value: **start from the raw inputs, never from the
handout's own intermediate values.** A script that reuses the handout's
intermediates confirms only that the handout is self-consistent, which a wrong
handout also is.

Strengthen it where you can:

- reach the answer by a **second route** (a closed form checked against an
  iterative one, or `S_xy / S_xx` against the normal equation);
- re-derive simulated quantities with a **different seed and sample size** than
  the notebook used, so that agreement is evidence rather than a copy.

And keep the tolerances honest in both directions. A tolerance states how
precisely a number is known, so if the handout's wording is tighter than the
check standing behind it, the wording is wrong — lesson 7 claimed two scores
matched "to four decimal places" while its own check confirmed them only to
0.015, and the check was the honest one. A claim that two quantities *agree*
needs the opposite of a tolerance as well: a check that they **differ** at some
other seed. One that could only ever pass proves nothing.

Print one line per number checked and a count at the end. `verify_lesson.py` runs
it and reports the count.

### Checking a derivation, not just the number it produces

This script recomputes **numbers**, and that is a real limit rather than a
shortcoming to work around. A derivation can be wrong while every number it
produces is right: the code implements the correct thing, the prose writes it
down with a transpose out of place or the sum over the batch missing, and the
check dutifully agrees — with the code. The handout is wrong and the gate is
green.

Everything the course has caught so far has been a wrong number. A lesson that
*derives* rather than measures — backpropagation, convolution arithmetic — puts
the weight on the half nothing is watching. So where the handout derives, find
a check on the derivation:

- **Against a numerical gradient.** For anything reached by differentiation,
  compare the analytic result against finite differences at random points:
  `(f(w + h) − f(w − h)) / 2h` with `h ≈ 1e-5` should agree to roughly `1e-7`
  relative in double precision. This is the standard defence, it is four lines,
  and it catches precisely the misplaced-transpose class of error.
- **Against a second route.** The same quantity by a different derivation —
  eigendecomposition against the SVD, a closed form against an iterative
  solver, a hand-differentiated gradient against an autodiff one.
- **Against the shapes it claims.** Writing the gradient as `Xᵀ(ŷ − y)/m`
  claims a shape as well as a formula. Assert it: most transposition errors are
  shape errors, and a shape assertion is one line.
- **At more than one point.** An identity that holds at the lesson's worked
  example and nowhere else is not an identity. Evaluate both sides on random
  inputs, the same way a claim of agreement gets re-run at another seed.

The test of whether this has been done properly: if someone edited a transpose
out of place in the handout's algebra — leaving the code untouched — would
anything go red?

---

## 5. Slides

Markdown for **pandoc**, not Marp. Read `Course/slides_syntax_example.md` first.

Front matter exactly:

```markdown
---
title: "Lesson {NN} — {Human-readable title}"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini — Università degli Studi dell'Aquila"
date: "{date from Course/SCHEDULE.md}"
---
```

- `#` starts a slide. Slide 1 is the agenda.
- **Maximum 5–7 bullets.** Slides are shown, not read.
- **No full derivations.** State the result, give the intuition, point to the
  handout section. That separation is why both documents exist.
- **A display equation gets its own slide.** Sharing one with bullets sends
  pandoc to the two-column caption layout, where the title shrinks to 15pt and
  the equation renders *smaller than the text explaining it*. The build fails a
  deck when it finds one.
- Same for a figure: a slide carrying a figure carries little else.
- **Speaker notes on every content slide**, in a `::: notes` div: what the
  lecturer says, one question to put to the room, and the handout section with
  the derivation. Sixty characters is the floor; a real note is a paragraph or
  three.
- Ordinary LaTeX for maths — the build converts inline to Unicode and display to
  images. Display maths goes **last** on its slide.
- Roughly 45–60 slides for three hours.

---

## 6. Quiz

Markdown cells only, no code cells, no outputs.

```markdown
**1. Question text ending with a question mark?**

<details>
<summary>
    <font size='3', color='darkgreen'><b>Answer</b></font>
</summary>
    <p>
    <ul>
        <li>First point — <b>bold</b> for key terms, <code>code</code> for formulas</li>
    </ul>
    </p>
</details>
```

- Numbered sequentially across the whole notebook, one question per cell.
- Section headers in their own `##` cells.
- At least 15 questions; 25 for a heavier topic.
- Mix definition, explanation, comparison, "why does X happen", and applied
  judgement.
- At least three requiring **reasoning about a derivation** from the handout,
  not recall.

---

## 7. Exercise

Assessed weekly work, so it must be unambiguous and self-contained. Set on the
lesson date, due the next lesson's date.

- A dataset that ships with the exercise and needs no network.
- Parts with explicit marks summing to 100.
- **Ask for a prediction before any fitting**, and mark the reasoning rather
  than the prediction. A confident wrong prediction that engages with the right
  properties is worth more than a hedged list, and saying so in the brief is
  what makes students actually commit.
- A final part that opens the generator's `TRUE_*` constants and asks the
  student to check themselves against the rule.
- State explicitly what loses marks, and what does not.
- Solvable in 2–3 hours by a student who attended.

---

## 8. Supplementary reading

One document per lesson in `Resources/`, opening by declaring it is **not
examinable** and giving a reading time.

The brief is *orthogonal* rather than *deeper*: not more theory, but the
consequences, applications and constraints of the lesson's material. Written so
far: the history of AI (L1), where data comes from and what it costs — GDPR,
annotation labour (L2), where linear models still win and regression to the mean
outside ML (L3), classifying people — COMPAS and incompatible fairness
definitions (L4), the replication crisis (L5), finding neighbours at scale —
approximate nearest neighbour search (L6).

Registered for the remaining lessons: L7 the right to an explanation and GDPR
Article 22; L8 segmentation and the ethics of inferred categories; L9 the energy
cost of training; L10 facial recognition and its regulation.

Close with a "Where to look next" list of primary sources.

---

## 9. Instructor solution

Executed, gitignored, never committed. Beyond the answers it carries **marking
notes**: what earns credit, what the common wrong answer looks like, and which
question carries the point of the lesson.

Worth including: a note on what deserves credit that a marking scheme would
miss. In lesson 6, a student who reports the top three models are not
distinguishable at n=200 and declines to name a winner has understood the
methodology lesson as well as this one, and that is worth more than a confident
ranking that happens to be right.
