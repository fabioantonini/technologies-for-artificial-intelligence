---
name: new-lesson
description: Build one three-hour lesson of "Technologies for Artificial Intelligence" — notebooks, handout, slides, quiz, exercise, supplementary reading — in the phase order that keeps every number in the prose traceable to code that actually ran. Use this whenever the work touches anything under `Lessons/NN_*/` in this repository: starting a new lesson, resuming one left half-built, regenerating an existing one, or adding a single artefact to one that already exists. Use it even when the request sounds like a small edit ("add a quiz to lesson 8", "the slides for lesson 3 need a figure"), because the phase order and the verification gate still decide whether the change is safe to ship.
---

# Building a lesson

This skill assumes you remember nothing. That is the normal case: a lesson takes
long enough that a session is often summarised partway through, and anything
important that lives only in the conversation is lost when that happens.

So nothing here depends on conversation history. **The filesystem is the state.**
Every phase ends by writing files, and the files tell you where you are.

---

## Start here

Run this first. It reads the lesson folder and tells you which phase you are in
and what is missing:

```bash
python .claude/skills/new-lesson/scripts/lesson_state.py 07
```

Then read, in this order:

1. **`CLAUDE.md`**, all of it. It carries the audience, the hard constraints, the
   naming, the notation table and the slide conventions. It is not boilerplate —
   the audience section in particular decides how everything else is written.
2. **`Course/SCHEDULE.md`** — this lesson's date and title, and the *next*
   lesson's date, which is the exercise deadline.
3. **The previous lesson's handout and slides.** Read the handout's first ~150
   lines and skim the slides. This is for register: how much is explained, how
   the speaker notes are voiced, how numbers are cited.

**Read one previous lesson, not six.** Reading the whole course to "get the
context" is how a session runs out of room before writing anything. One lesson
establishes the register; the conventions are in `CLAUDE.md`.

Two more, when you reach the phase that needs them:

- `Course/slides_syntax_example.md` — before writing slides. It is the reference
  for what survives pandoc.
- `references/artefacts.md` in this skill — the specification of each artefact:
  what a handout section contains, the quiz format, what an exercise must state.

---

## The phase order, and why it is this one

**A — data and notebooks. B — handout. C — the other artefacts, in parallel.
D — an executable review.**

The tempting order is handout first, everything else derived from it. It is
wrong, and the repository has the evidence. A handout written before the code
runs has to **invent its numbers**, and then either the notebooks contradict it
or — worse — the notebooks get adjusted until they agree. Prose written ahead of
execution was wrong four times across lessons 3 to 5: "about 2,600" when the fit
gave 2,785; overfitting that was invisible at 84 training points; Lasso penalties
out by a factor of a thousand; two learning curves that looked identical.

Note what the ordering actually buys. It does **not** stop you writing a wrong
number — lesson 6 still had three wrong claims in a notebook when they were
typed. It shortens the distance between writing one and running the cell that
contradicts it, from a whole document to a single cell. That is why execution has
to follow prose immediately and not at the end of the phase.

---

## Phase A — data and notebooks

Ends when: every notebook is executed with its outputs saved, every figure is in
`Figures/`, and you have a written list of the numbers the lesson will cite.

1. **Design the dataset with published truth.** A `*_data.py` module beside the
   notebooks, with `TRUE_*` constants recording how labels were generated. It
   makes the exercise's Part 4 possible, and it makes calibration honest — you
   can check the data does what you think before any model sees it.

   Nothing may need a network call, an API key, or a paid service at runtime.

2. **Write and execute the notebooks**, in the running container:

   ```bash
   docker exec tai_course bash -lc 'cd /home/jovyan/work/Lessons/NN_*/Notebooks && jupyter nbconvert --to notebook --execute --inplace 01_*.ipynb'
   ```

   Outputs are committed. A student opening a notebook should see what it does
   before running anything.

3. **Recalibrate until the numbers tell the story.** If the lesson's point is
   that method X fails here, and X scores 0.986 against 0.999, the demonstration
   has failed and the dataset is wrong, not the prose. Lesson 6 discarded its
   first Naive Bayes dataset for exactly this and replaced it with one where the
   score fell to 0.404 — below chance, which is the version worth showing.

4. **Then re-read every notebook's prose against its own output.** Dump the
   executed outputs and check each number the markdown cites. This is not
   optional tidying: three claims in lesson 6's first notebook survived writing
   and review and were contradicted by the cell directly above them.

   ```bash
   python .claude/skills/new-lesson/scripts/notebook_outputs.py Lessons/NN_*/Notebooks
   ```

---

## Phase B — the handout

Ends when: the handout builds, carries every figure Phase A generated, and
`Docs/worked_examples.py` passes.

- Written **against the executed numbers**, quoting them as they came out.
- **Every figure Phase A generated appears here**, with an italic caption. Not
  "every figure the slides will use" — the slides do not exist yet, so that is
  unknowable, and lesson 7 found out the hard way: two figures surfaced as
  orphans only when the Phase C slides agent reached for them. The handout is
  what a student reads after the lesson, when the projected figures are gone.
  `verify_lesson.py` enforces the slides-⊆-handout direction once both exist.
- **`Docs/worked_examples.py` is written now, not afterwards.** It recomputes
  every number the handout works out by hand, **from the raw inputs — never from
  the handout's own intermediate values** — and asserts against what the handout
  prints. Where it can, it should reach the answer by a second route, or with a
  different seed and sample size, so that agreement is evidence rather than a
  copy.

  This is the single most valuable artefact in the lesson. The one arithmetic
  error that reached a built PDF survived being written, reviewed, rewritten by a
  separate agent, and audited across all 113 equations — because every number
  downstream of it had been correctly derived from the wrong one. Only
  recomputation from raw inputs caught it.

  See `Lessons/06_knn_naive_bayes_svm/Docs/worked_examples.py` for the shape.

---

## Phase C — slides, quiz, exercise, supplementary reading

These four are genuinely independent of one another and all depend only on the
finished handout, so this is the one phase where parallel subagents pay for
themselves. Every subagent restarts cold and re-derives context, which is the
expensive path — do not use them in phases A, B or D, where the work is
sequential anyway.

The briefs are written out in `references/subagent-briefs.md`. Give each agent
the finished handout, the notebook outputs, `CLAUDE.md`, and the previous
lesson's counterpart artefact as an example of register.

**Do not poll them.** You are notified when each one finishes. Lesson 7's run
spent several turns asking whether they were done and spawned a stray no-op
agent trying to cancel a scheduled wake-up. Launch, then do something else or
wait.

**Check any data a subagent generates yourself.** In lesson 6 the exercise agent
produced a two-station dataset and was cut off before reporting what it did;
measuring it directly confirmed the ranking really did invert, which the brief
had asked for but nothing had verified. A subagent's report is a claim, not a
result.

Ends when: all four artefacts exist and `build.py NN` reports `0 failed`.

---

## Phase D — the review that executes, then the one that looks

A review that reads finds plausible work plausible. Do these in order; each
catches a different class of defect.

**1. Mechanical.**

```bash
python tools/verify_lesson.py NN --run
```

Nine checks: notebooks run clean, versions match the container, figures exist and
are reproducible, slides have speaker notes, the lesson plan sums to 180 and
cites slides that exist, the quiz numbering is consecutive, acronyms are
expanded, cross-references resolve. It also runs `worked_examples.py`.

**2. Arithmetic.** Covered by the above — but read its count. "20 hand-worked
numbers recomputed" against a handout with forty is a `worked_examples.py` that
has not kept up.

**3. Numbers against prose.** Dump every executed output and read it beside the
prose that cites it, in the notebooks *and* the handout. **No tool does this
step**, which is why both defects that escaped every other check so far were of
this shape.

```bash
python .claude/skills/new-lesson/scripts/notebook_outputs.py Lessons/NN_*/Notebooks
```

Two failures worth hunting by name:

- **A claim stronger than its evidence.** Lesson 6's solution asserted a ratio
  "is worth a great deal" directly beneath a printed correlation of 0.079. The
  number was right; the claim needed a measurement that could see a
  non-monotonic relationship, and a correlation cannot.

- **A coincidence presented as a law.** Lesson 7 reported an out-of-bag score
  and a cross-validated score agreeing to four decimal places and wrote
  "*because* they are estimating the same quantity". They are — but four-decimal
  agreement happens on two seeds in twelve, and the honest gap is ±0.003.
  **Any sentence claiming two quantities agree must be re-run at another seed
  before it is written.** If the agreement is a property it survives; if it does
  not, what you had was one draw.

  This second one repays understanding rather than just avoiding, because
  everything downstream behaved correctly. The claim propagated into the slides,
  a notebook and a quiz answer — faithfully, in each case — and
  `worked_examples.py` confirmed it at `tolerance=0.015`, which is to say it
  proved the *weaker, true* statement while the prose asserted the tighter,
  false one. **A tolerance is a statement about how precisely a number is
  known.** When the wording is tighter than the check standing behind it, the
  wording is what is wrong.

**4. Look at the built PDF.** Not "check the deck" — open specific slides.

Every slide carrying a figure:

```bash
grep -n 'png' Lessons/NN_*/Slides/*_slides.md | grep -v eq_
```

**And the fullest slides of pure text**, which is the half this checklist used
to omit — and so two overflowing text slides shipped in lesson 6, one of them
its closing "What to take away", from a deck I had reviewed by eye. The build
now flags a text slide that runs past the bottom, but its estimate is coarse and
only catches the worst; sort by bullet count and look at the top few regardless.

The build's two layout checks — a figure squeezed by its text, a text slide off
the bottom — are narrow. They cannot see an annotation lying across a curve, a
legend over data, an axis label cropped away by `bbox_inches="tight"`, a panel
too small for the point it carries, or a figure that does not say what its
caption promises. All five were in lesson 6's first build, and every one of them
needed a person to look.

Also confirm the build is reproducible — two consecutive builds byte-identical:

```bash
python tools/build.py NN && md5sum Lessons/NN_*/Slides/*.pdf > /tmp/a
python tools/build.py NN && md5sum Lessons/NN_*/Slides/*.pdf > /tmp/b && diff /tmp/a /tmp/b
```

---

## What is never delegated

- **The notebooks.** The numbers are *discovered* by running code, not
  transcribed. Delegating them means writing the handout on numbers you did not
  watch appear.
- **The handout.** Same reason, one step later.
- **Phase D.** A subagent asked to review will report that the work looks
  correct, because it does.

---

## The exercise solution

Write a worked solution for the instructor, execute it, and **do not commit it**
— `.gitignore` covers `*_solution.ipynb`. It should carry marking notes saying
what to give credit for, not just the answers.

It is held to the same standard as everything else while being written, but
`verify_lesson.py` deliberately skips it: student-facing rules applied to the
marking key turn a green/red gate into one with a standing exception.

---

## Finished means all of these

- `verify_lesson.py NN --run` green
- `build.py NN` reports `0 failed`
- two consecutive builds byte-identical
- every figure slide opened and looked at
- `Resources/` holds one supplementary document, opening by saying it is not
  examinable
- the solution notebook exists on disk and is absent from `git status`
- **`git status` shows nothing modified outside this lesson's folder.** Phase A
  rewrites notebooks through the container, and it is easy to touch another
  lesson's kernel metadata in passing. Lesson 7 modified two of lesson 1's
  notebooks and reported "everything is untracked" without looking.

Report anything you could not finish rather than reporting it as done. A lesson
that is nine-tenths built and described as complete costs more than one honestly
described as nine-tenths built.
