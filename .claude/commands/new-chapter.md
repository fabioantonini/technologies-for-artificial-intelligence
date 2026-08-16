Generate all five artefacts for one lesson of Technologies for Artificial Intelligence.

## Arguments

`$ARGUMENTS` has the form:

```
<TopicName> <NN> "<brief description>" [--refs "url1, url2"] [--depth standard|deep]
```

Examples:

```
Regression 03 "Linear regression, cost function, gradient descent, Ridge and Lasso"

TreesAndEnsembles 07 "Decision trees, bagging, random forests, gradient boosting, XGBoost" --depth deep

ExperimentalMethodology 05 "Train/validation/test, cross-validation, bias-variance, leakage" --refs "https://scikit-learn.org/stable/modules/cross_validation.html"
```

Parse into:

- `TopicName` — PascalCase, used for the quiz filename (e.g. `TreesAndEnsembles`)
- `topic_snake_case` — lowercase with underscores (e.g. `trees_and_ensembles`)
- `NN` — two-digit lesson number; the target folder is `Lessons/{NN}_*/` which
  **already exists** — do not create a new one, and do not rename it
- `description`, `refs` (default none), `depth` (default `standard`)

---

## Before writing anything

Read `CLAUDE.md`. The audience section is not boilerplate: these are first-year CS MSc
students, strong programmers, comfortable with linear algebra and probability, with no
datasets of their own. Every artefact below depends on that.

Three hard constraints, no exceptions:

- **No LLM content**, and nothing requiring an API key or a network call at runtime.
- **No Coursera / DeepLearning.AI derived material** — not the labs, not the figures,
  not the slide notes.
- **English throughout**, including file names.

If `--refs` was given, fetch each URL with `WebFetch` **first** and treat it as the
source of truth for current API signatures and version numbers.

---

## Artefact 1 — Handout (`Docs/{topic_snake_case}.md`)

The reference text. This is where the **mathematics lives**.

Header, followed immediately by the lesson plan:

```markdown
# {Human-readable title}

> **Lesson {NN} — Technologies for Artificial Intelligence**
> Estimated reading time: XX minutes

---

## Lesson plan

| Time | Segment | Material |
|---|---|---|
| 0:00-0:20 | ... | ... |
| ... | ... | ... |
| | **Total** | **180 minutes** |
```

**The plan must sum to 180 minutes.** Write it before the body and use it to size the
content: roughly 25-30 slides per lecture hour, 20-30 minutes per notebook worked
through live, one break. A lesson that would run out after 100 minutes is under-built.

Requirements:

- Numbered `##` sections, `###` subsections.
- Flowing prose that explains the *why*. No bullet-point-only sections.
- **Complete derivations**, written out step by step, in LaTeX. Inline `$...$`,
  display `$$...$$`. Do not skip algebra with "it can be shown that" — showing it is
  the point of this document.
- State assumptions explicitly, and say where each result stops holding.
- Tables for comparisons; ASCII diagrams for structures and flows.
- `standard`: at least 8 sections. `deep`: at least 12, each key concept getting a
  derivation subsection.
- Two or three challenge boxes between sections:

```markdown
> **Try this:** {A concrete experiment: change a parameter, swap a component,
> observe the difference. One or two sentences.}
```

- Close with a **Further reading** table of 4-6 entries (title, type, why it is worth
  reading). Prefer primary sources and official documentation.

---

## Artefact 2 — Slides (`Slides/{topic_snake_case}_slides.md`)

Markdown for **pandoc**, not Marp. Read `Course/slides_syntax_example.md` first — it
is the reference for every construct that survives conversion.

Front matter, exactly:

```markdown
---
title: "Lesson {NN} — {Human-readable title}"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini — Università degli Studi dell'Aquila"
date: "{lesson date from Course/SCHEDULE.md}"
---
```

Then:

- `#` starts a new slide. Slide 1 is the agenda.
- **Maximum 5-7 bullets per slide.** Slides are shown, not read.
- **No full derivations.** State the result, give the intuition, point to the handout
  section. This separation is the reason both documents exist.
- Figures come from the lesson notebooks, referenced as `![Caption](figure_name.png)`
  and living in the lesson's `Figures/`.
- Use tables for comparisons and fenced code blocks for short snippets.
- **Speaker notes on every content slide**, in a `notes` div. They never render on the
  slide; they appear in presenter view. Write what the lecturer will say, one question
  to put to the room, and the handout section carrying the derivation:

```markdown
::: notes
Motivate with the housing example before the formula. Ask: why square the
errors rather than take absolute values? Derivation in handout section 3.2.
:::
```

- Write ordinary LaTeX for maths. The build converts inline math to Unicode and display
  math to images automatically — you never write those substitutions yourself. Two
  rules, both enforced by build warnings:
  - **display maths goes last on its slide**, otherwise the text after it is orphaned
    onto a new untitled slide;
  - **inline maths must be simple enough for Unicode** (`x^2`, `\tfrac{1}{2}`, greek,
    set symbols). Anything heavier belongs in display position or in the handout.
- Two columns where a contrast helps:

```markdown
::: columns
:::: column
**Left**
::::
:::: column
**Right**
::::
:::
```

- A 3-hour lesson is roughly 45-60 slides.

Build and check it compiles:

```bash
python tools/build.py {NN}
```

---

## Artefact 3 — Notebooks (`Notebooks/{NN}_{topic_snake_case}.ipynb`)

Two to four notebooks per lesson, numbered from `01`.

- **Cell 1** (markdown): `# {Title}` plus a paragraph on what the notebook shows and
  which handout section it implements.
- **Cell 2** (markdown): `## 1. Setup`
- **Cell 3** (code): commented `pip install` with pinned versions, then imports, then
  `np.random.seed(...)` — reproducibility is taught by example here.
- Then alternate: markdown cell with the idea (2-4 sentences plus the key formula),
  code cell implementing it, optional markdown cell on what to observe.
- Final cell: `## Summary` recapping what was demonstrated.

Rules:

- Runs **top to bottom in a clean container**, no API key, no manual downloads beyond
  `sklearn`/`openml` fetches with caching.
- Real implementations on real or synthetic data — **never mocked results**.
- At least one notebook per lesson implements the core method **from scratch with
  NumPy** before showing the library version. This audience learns by building.
- **Every figure used in the slides is generated here** and saved into the lesson's
  `Figures/`:

```python
fig.savefig("../Figures/{descriptive_name}.png", dpi=150, bbox_inches="tight")
```

---

## Artefact 4 — Quiz (`Quizzes/{TopicName}-Quiz.ipynb`)

Markdown cells only, no code cells. Exact format per `CLAUDE.md`:

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

- Numbered sequentially across the whole notebook; section header cells separate from
  question cells.
- `standard`: at least 15 questions. `deep`: at least 25.
- Mix definition, explanation, comparison, "why does X happen", and applied judgement.
- Include at least three questions that require **reasoning about a derivation** from
  the handout, not recall.

Notebook JSON: `nbformat` 4, `nbformat_minor` 5, kernelspec Python 3, sequential cell
ids.

---

## Artefact 5 — Exercise (`Exercises/{NN}_{topic_snake_case}.md`)

Assessed weekly work, so it must be unambiguous and self-contained.

```markdown
# Exercise {NN} — {Title}

**Set:** {lesson date} · **Due:** {following lesson date}

## Goal
{One paragraph: what the student will be able to do afterwards.}

## Dataset
{Named dataset with the exact loading snippet. Never "your own data".}

## Tasks
1. ...
2. ...

## What to hand in
A notebook named `{surname}_{NN}.ipynb` containing {explicit deliverables}.

## Assessment criteria
| Criterion | Weight |
|---|---|
| Methodological correctness | 40% |
| Implementation | 30% |
| Interpretation and communication | 30% |
```

Tasks must build towards the final project. Difficulty: solvable in 2-3 hours by a
student who attended the lesson.

---

## Finally

Run the build, then report:

```
Lesson {NN} — {TopicName}

  Docs/{topic_snake_case}.md            {n} sections, {n} derivations
  Slides/{topic_snake_case}_slides.md   {n} slides -> .pptx built
  Notebooks/                            {n} notebooks, {n} figures generated
  Quizzes/{TopicName}-Quiz.ipynb        {n} questions
  Exercises/{NN}_{topic_snake_case}.md  due {date}

Options: --refs={refs or "none"} --depth={depth}
```

Flag anything you could not complete rather than reporting it as done.
