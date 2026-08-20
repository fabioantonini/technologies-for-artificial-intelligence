# Phase C subagent briefs

Four artefacts, genuinely independent of one another, all depending only on the
finished handout. Launch all four in one turn.

Substitute `{NN}`, `{TopicName}`, `{topic_snake_case}`, `{lesson date}`,
`{next lesson date}` and `{previous lesson folder}` before sending.

**Every brief opens with the block below.** A subagent starts cold: it has none
of the conversation, so anything it is not told, it does not know.

---

## Shared preamble

```
You are writing one artefact for lesson {NN} of "Technologies for Artificial
Intelligence", a 30-hour course for first-year Computer Science MSc students at
the University of L'Aquila. Repository root: /home/fabio/technologies-for-artificial-intelligence

Read first, in this order:

1. CLAUDE.md — the audience, the hard constraints, the conventions. All of it.
2. Lessons/{NN}_*/Docs/{topic_snake_case}.md — the finished handout. It is the
   source of truth for every number, every definition and every claim. Do not
   introduce a number that is not in it or in a notebook output.
3. Lessons/{NN}_*/Notebooks/ — the executed notebooks, for the outputs.
4. {previous lesson folder}/<your artefact> — the previous lesson's counterpart,
   as an example of register. Match its voice, not its content.

Hard constraints, no exceptions:
- No LLM content of any kind — no transformers, no retrieval-augmented
  generation, no agents, no API keys. This is a degree-programme rule.
- Nothing that needs a network call or a paid service at runtime.
- No Coursera or DeepLearning.AI derived material.
- English throughout, including file names.
- Expand every acronym on first use, in this artefact — a student reading only
  this one has not read the others.

When you finish, report: what you wrote, where, and the specific numbers or
counts that let someone check it without opening the file. If you could not do
part of it, say which part. Do not report as done anything you did not verify.
```

---

## 1. Slides

```
Write Lessons/{NN}_*/Slides/{topic_snake_case}_slides.md.

Read Course/slides_syntax_example.md before starting — it is the reference for
what survives pandoc conversion.

The handout's lesson plan table fixes the structure: it names segments, minutes
and slide ranges. Your deck must match it, so read the plan first and build to
it. Roughly 45-60 slides.

Front matter exactly:

---
title: "Lesson {NN} — {Human-readable title}"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini — Università degli Studi dell'Aquila"
date: "{lesson date}"
---

Then:
- `#` starts a slide; slide 1 is the agenda.
- Maximum 5-7 bullets. Slides are shown, not read.
- No derivations. State the result, give the intuition, cite the handout
  section. That separation is why both documents exist.
- A display equation gets its OWN slide, and so does a figure. Putting either
  alongside bullets sends pandoc to a two-column caption layout where the title
  shrinks to 15pt and the equation renders smaller than the text explaining it.
  The build fails a deck when it finds one.
- Display maths goes last on its slide; inline maths must be simple enough to
  become Unicode.
- Speaker notes on EVERY content slide, in a `::: notes` div: what the lecturer
  says, one question to put to the room, and the handout section carrying the
  derivation. A paragraph or three, not a sentence.
- Every figure comes from Lessons/{NN}_*/Figures/, referenced as
  `![](name.png)`, and must already appear in the handout.

Check it builds before reporting: python tools/build.py {NN}

Report: the slide count, and confirmation that it matches the handout's plan
table.
```

---

## 2. Quiz

```
Write Lessons/{NN}_*/Quizzes/{TopicName}-Quiz.ipynb.

Markdown cells only. No code cells, no outputs. Exact format:

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

- Numbered sequentially across the whole notebook, ONE question per cell.
- Section headers in their own `##` cells, separate from question cells.
- At least 15 questions, 25 for a heavy topic.
- Mix definition, explanation, comparison, "why does X happen", and applied
  judgement.
- At least three require reasoning about a derivation in the handout, not
  recall.
- Answers of two or more bullets, except where the question explicitly asks for
  one sentence.
- Every question must actually ask something: end with a question mark, or open
  with an imperative (Define, State, Explain, Compute, Compare).

Notebook JSON: nbformat 4, nbformat_minor 5, kernelspec Python 3, cell ids
present.

Verify before reporting: python tools/verify_lesson.py {NN}
It checks numbering, cell structure, the <details> pairing and answer length.

Report: the question count and the section headings.
```

---

## 3. Exercise

```
Write the assessed exercise for lesson {NN}:

  Lessons/{NN}_*/Exercises/{NN}_{topic_snake_case}.md   — the brief
  Lessons/{NN}_*/Exercises/{something}_data.py          — its dataset

Set {lesson date}, due {next lesson date}, 23:59. Solvable in 2-3 hours by a
student who attended. Parts with explicit marks summing to 100.

The dataset ships with the exercise and needs no network. Give it TRUE_*
constants recording how the labels were generated, and a long honest module
docstring describing the process — for the student, that docstring is half the
brief.

Design it so the lesson's idea has to be USED, not recalled. The strongest shape
found so far: two datasets from the same setting where the obvious answer
inverts between them, so that neither ordering can be guessed from the names of
the methods but both can be predicted from twenty minutes spent looking at the
data.

Structure that works:
- Part 1: predict the outcome BEFORE fitting anything, with reasons. Mark the
  reasoning, not the prediction — and say so in the brief, or students hedge.
- Part 2: measure it properly, cross-validated, with error bars.
- Part 3: explain any surprise, supporting each claim with a MEASUREMENT rather
  than an assertion. This part carries the most marks.
- Part 4: open the generator's TRUE_* constants and mark yourself against the
  rule that made the data.

State explicitly what loses marks, and what does not.

Before reporting, MEASURE YOUR OWN DATASET and put the numbers in your report:
row counts, positive rates, majority baselines, and the cross-validated score of
each method you expect students to try. If the effect you designed for is not
actually there, fix the data — do not describe an effect the data does not have.

Report: those numbers, per dataset.
```

---

## 4. Supplementary reading

```
Write Lessons/{NN}_*/Resources/{descriptive_snake_case}.md.

Not examinable, and it opens by saying so, with a reading time. See
Lessons/01_introduction_and_workflow/Resources/history_of_ai.md for the shape.

The brief is ORTHOGONAL, not deeper: not more theory, but where this material
meets the world — applications, consequences, legal and physical constraints,
the history of how it came to be done this way. A student should finish it
knowing something the exam will not ask about and an employer will.

Topic for this lesson: {topic}

It must connect to the lesson explicitly — open from something the handout
established, and say which section. Close with "What to take from it" (three or
four claims worth carrying) and "Where to look next" (primary sources: papers,
standards, official documentation, one good book chapter).

Aim for 150-250 lines. Prose, not bullets.

Report: the section headings and the reading time you declared.
```
