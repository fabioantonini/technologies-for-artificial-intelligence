# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## What this repository is

Teaching material for **Technologies for Artificial Intelligence**, a 30-hour course
(10 lessons x 3 hours) for **first-year Computer Science MSc students** at Università
degli Studi dell'Aquila. For most of them this is the **first AI/ML course**.

Instructor: Fabio Antonini — fabio.antonini.1969@gmail.com

There is no application here, no test suite and no linter. The deliverable is
markdown that is compiled into slides and handouts, plus Jupyter notebooks, shipped
to students as a public Docker image plus this repository.

### Who the audience is, and why it matters

This drives nearly every content decision, so it is worth stating plainly.

CS MSc students are **strong programmers** who have already met linear algebra and
probability. They are **not** domain scientists with their own datasets. Concretely:

- **Do not** dilute the mathematics. They can handle derivations and will find a
  purely hand-wavy treatment thin. The handouts carry the full derivations.
- **Do not** write exercises of the form "apply this to your own research data".
  They have none. Every exercise ships with a dataset.
- **Do** invest in experimental methodology. They will produce models that *look*
  like they work very quickly; without honest validation they will produce invalid
  results with great confidence. This is why Lesson 5 exists and sits mid-course.
- **Do not** spend lecture time on Python or Jupyter basics beyond a brief tour.

### Hard constraints

- **No LLM content.** Excluded by the degree programme. No transformers, no RAG, no
  agents, no API keys. Nothing in this course may require a paid service or a network
  call at runtime.
- **No Coursera / DeepLearning.AI derived material.** This repository exists partly to
  break that dependency. Do not copy in labs, figures, or slide notes from those
  courses — not even temporarily. Anything committed stays in git history.
- **Everything in English.** See below.

---

## Language

**Every course artefact is in English**: handouts, slides, notebooks, quizzes,
exercises, assessment material, README, file and folder names, commit messages and
issues.

The single exception is `Course/it/`, which holds the Italian administrative documents
the university requires (official programme, course record). Nothing else in the
repository is in Italian.

---

## The three artefacts

Every lesson produces three things with **non-overlapping** roles. Keeping them
distinct is what lets the course be rigorous without slowing the lecture down.

| Artefact | Role | Source | Built output |
|---|---|---|---|
| **Handout** (`Docs/`) | The reference text: extended narrative and **complete mathematical derivations**. Studied after the lecture. | Markdown + LaTeX | `.pdf` |
| **Slides** (`Slides/`) | The in-class skeleton: statements, intuitions, figures. **Little mathematics.** | Markdown | `.pptx` + `.pdf` |
| **Notebook** (`Notebooks/`) | The implementation: code that makes operational what the handout proves. | Jupyter | — |

Plus `Quizzes/` (self-check), `Exercises/` (assessed homework) and, where useful,
`Resources/` (optional supplementary reading).

### Every lesson fills three hours, and every lesson sets homework

Two non-negotiables.

**Three hours of material.** Each lesson runs 3 hours, and the material must actually
fill it. Every handout opens with a **lesson plan table** budgeting the time across
segments — lecture, notebook work, break, discussion — summing to 180 minutes. Write
the plan first and use it to size the content; a lesson that runs out after 100
minutes has been under-built, and one that needs 240 has to be cut.

Reckon on roughly **25-30 slides per hour** of lecture, and **20-30 minutes per
notebook** worked through live. Three notebooks of substance is a normal lesson.

**Homework, every time.** Every lesson ends by setting an exercise from
`Exercises/`, due the following Friday. It is never optional and never skipped: the
exercises are assessed, and they are what carries students towards the final project.
Announce it on the last slide and state the deadline.

**Never put a full derivation on a slide.** That is the whole reason the handout
exists. In class you show the result and why it matters; the proof is read afterwards.

### Deadlines differ by artefact

Slides and notebooks must exist **on the Friday of the lesson**. The handout is
post-lecture study material and may land **by the following Monday**. Use that slack —
it is what makes ten mathematically serious handouts feasible part-time.

---

## Repository layout

```
Course/            SYLLABUS.md, SCHEDULE.md, PREREQUISITES.md, template.pptx, Setup/
Lessons/NN_topic/  Docs/ Slides/ Notebooks/ Quizzes/ Exercises/ Figures/ [Resources/]
Assessment/        Project/ Exam/ Exercises/
tools/             build.py, make_template.py, postprocess_pptx.py, render_math.py, release.py
```

`Resources/` is the one optional folder: supplementary reading that enriches a lesson
without being examinable, such as the history of AI in lesson 1. Everything else is
present in every lesson.

The layout is **uniform with no exceptions**. The previous course repository drifted
into five naming schemes and two parallel slide formats; that is the specific failure
this rule prevents.

### Naming

| Artefact | Pattern | Example |
|---|---|---|
| Lesson folder | `{NN}_{topic_snake_case}` | `03_regression` |
| Handout | `{topic_snake_case}.md` | `regression.md` |
| Slides | `{topic_snake_case}_slides.md` | `regression_slides.md` |
| Notebook | `{NN}_{topic_snake_case}.ipynb` | `01_gradient_descent_from_scratch.ipynb` |
| Quiz | `{TopicName}-Quiz.ipynb` | `Regression-Quiz.ipynb` |
| Exercise | `{NN}_{topic_snake_case}.md` | `01_fit_and_diagnose.md` |

---

## Where the work happens

**The working copy lives in WSL**, at `/home/fabio/technologies-for-artificial-intelligence`,
and every build runs there. Not a preference — three concrete reasons:

- **It is the only environment with a complete toolchain.** pandoc, LibreOffice and a
  working TeX Live. The MiKTeX install on the Windows side refuses to run, so handout
  PDFs cannot be produced there at all.
- **It is outside OneDrive.** A git repository inside a synced folder is a known way to
  corrupt an index, and it has already produced file locks that blocked a `git reset`.
- **Native ext4**, so git and notebook I/O are fast and there is no CRLF churn.

Edit from Windows if you prefer: VS Code opens the tree directly over Remote-WSL, and
Explorer reaches it at `\\wsl.localhost\Ubuntu-24.04\home\fabio\...`. There is no
second checkout to keep in step.

Python lives in `.venv/` there (Ubuntu 24.04 refuses system-wide pip installs), so
prefix commands with `.venv/bin/python`. Docker Compose is run from that directory too,
which is what makes the container serve this copy.

**Regenerate figures in one environment only — the container.** matplotlib versions
differ between host, WSL and image, so the same plot produces different bytes in each
and every notebook run shows up as a diff.

---

## Commands

### Build slides and handouts

```bash
python tools/build.py          # everything
python tools/build.py 03       # one lesson
python tools/build.py --no-pdf # slides only, skip PDFs
```

Slides are written in markdown and converted with **pandoc** against
`Course/template.pptx`, which carries the university identity. The resulting `.pptx`
has **native editable text**, real tables and **OMML equations**, so it can be tweaked
in PowerPoint before a lecture.

Handout PDFs need a working LaTeX engine. If none is available the script says so and
still builds the slides; the markdown sources render with full math on GitHub anyway.

Rebuild the template only when the visual identity changes:

```bash
python tools/make_template.py
```

### Name the predictable mistakes

Every lesson states, in the handout or the notes, **what students will get wrong
and why the wrong answer is reasonable**. That second half is what makes it
useful: "most will say mean imputation is free, and the reasoning is sound as far
as it goes — the mean is unbiased" tells whoever is teaching where to pause, and
tells the student their instinct was not stupid.

A list of correct answers is worth less than one well-understood misconception.

### One number per lesson worth remembering

Pick a single figure that carries the lesson's argument, quote it on a slide, and
repeat it at the close. Derivations fade; a number that surprised someone does
not.

The ones so far:

| Lesson | The number |
|---|---|
| 1 | **77%** accuracy on coin-flip labels, from leakage alone |
| 2 | **98 of 128** imputed rows borrowed a value from the test set |
| 3 | **247,514** — the largest coefficient of an overfitted fit, against 411 |

### Expand every acronym on first use

Write the words, then the abbreviation in brackets, every time a term first
appears in a document: mean squared error (MSE), missing completely at random
(MCAR), interquartile range (IQR), area under the curve (AUC).

This audience is new to the field, and an unexplained acronym is a silent stop:
the reader either guesses or loses the sentence. Repeat the expansion in each
artefact — a student reading only the slides has not seen the handout.

### Shared notation

The same symbol means the same thing in every lesson. This table is the
authority; check it before introducing a letter.

| Symbol | Meaning | Where it appears |
|---|---|---|
| $x$, $X$ | one input, the design matrix | throughout |
| $y$, $\hat{y}$ | true target, prediction | throughout |
| $w$, $b$ | coefficients, intercept | 3, 4, 9 |
| $\theta$ | all parameters together, intercept included | 3 |
| $m$ | number of examples | throughout |
| $n$ | number of features | throughout |
| $\alpha$ | **learning rate**, and nothing else | 3, 9 |
| $\lambda$ | **regularisation strength** (scikit-learn spells it `alpha`) | 3, 6, 9 |
| $J$ | the cost being minimised | throughout |
| $L$ | the loss on a single example | 1, 3, 4 |
| $\mathcal{D}$ | the unknown data distribution | 1, 5 |
| $R$, $\hat{R}$ | expected risk, empirical risk | 1, 5 |

Lesson 3 originally used $\alpha$ for both the learning rate and the penalty, a
few pages apart. That is the failure this table exists to prevent.

### Intuition comes before the symbols

**No derivation starts cold.** Before the notation appears, say in plain language
what is about to happen and why anyone would expect it — a sentence or two that a
student could repeat to a friend without writing anything down.

The order is always: *what we are trying to do* → *why this approach is natural*
→ *the mathematics* → *what the result means in the units of the problem*.

Some that work in this course:

- **The normal equation** is the shadow of the target on the space the features
  can reach. The residual is what is left over, and it is perpendicular, because
  anything else would mean part of the error still lay in a direction the model
  could have used.
- **The gradient** is a weighted vote: each example pulls a coefficient in
  proportion to how wrong it was *and* how large that feature was for it. Big
  houses shout louder about the price of floor space.
- **Ridge versus Lasso** is a disc versus a diamond. Corners sit on the axes,
  and a corner is where a coefficient is exactly zero, so the shape with corners
  is the one that produces zeros.
- **The condition number** is how stretched the valley is. One step size has to
  serve every direction, so the steep direction sets the limit and the shallow
  one crawls.

A student who has the picture can rebuild the algebra. A student who has only the
algebra has nothing to fall back on when it does not apply.

### Every general statement earns a concrete example

A derivation or a rule that stays abstract is one students can follow and not
use. **Every handout section states its result and then works it through with
actual numbers**, and the slides carry the numbers rather than only the claim.

Concretely:

- After a derivation, substitute real values from the lesson's dataset and
  compute the answer. "Mean imputation attenuates correlation by the square
  root of one minus the missing fraction" becomes "age is 8% missing, so
  0.96 - negligible; at 40% it is 0.77, which erases a quarter of the
  relationship".
- Prefer an example the students can check against the notebook output, so the
  handout and the code corroborate each other.
- On slides, quote the number. A slide that says "scaling matters" is weaker
  than one that says "tenure runs 0-70, charges 15-150, and the condition
  number is what sets the iteration count".
- Where a method has a failure mode, show it happening on data rather than
  describing it. Lesson 1's 77% accuracy on coin-flip labels does more than a
  paragraph about leakage would.

### Verify before it reaches a student

**A lesson is not finished when it builds. It is finished when
`tools/verify_lesson.py` passes on it.**

```bash
python tools/verify_lesson.py 06          # the mechanical checks, seconds
python tools/verify_lesson.py 06 --run    # also executes every notebook
python tools/verify_lesson.py             # every lesson
```

Run it as the last step of every lesson, before the commit. It exits non-zero
when something is wrong, so it can gate one.

The checks are in two groups, and the distinction matters more than the list.

**Mechanical, and automatic.** Notebooks that carry no stored tracebacks and
have been re-run since their last edit; pinned versions that match what the
image actually ships; every referenced figure present, every generated figure
reproducible from a notebook or `make_figures.py`, no orphaned equation images;
inline maths that survives the Unicode conversion; no empty or untitled slides;
speaker notes on every content slide; a lesson plan summing to 180 minutes and
citing slides that exist; acronyms expanded on first use in each artefact;
cross-references that resolve.

**Arithmetic, and yours to write.** Every lesson carries
`Docs/worked_examples.py`, which recomputes each number the handout works out by
hand and asserts it against what the handout prints. The verifier discovers and
runs it; a lesson without one is reported as incomplete.

**The rule that makes it worth anything:** each check must reach the handout's
figure by a route that does not reuse the handout's intermediate values. Start
again from the raw inputs. Where a second method exists — the normal equation
against $S_{xy}/S_{xx}$, a formula against a simulation — use both.

That rule is not fastidiousness. Section 3.2 of lesson 3 once printed 165,200
where the three houses give 165,600, and **every figure after it was derived
correctly from the wrong one**: the determinant, the inverse, both components of
$\theta$. The example was internally consistent, read as careful work, survived
a review and a deck rewrite, and was repeated on a slide and twice in the
speaker notes. Nothing but recomputing from the three houses would have caught
it.

Two habits follow, for content the checker cannot reach:

- **A figure is checked by looking at it.** Open the built PDF. Overlapping
  annotations, a legend across a curve, an axis label cut off — none of these
  fail a test.
- **A number in prose is checked against the cell that produced it.** If the
  handout says a coefficient came out at 1.66, that digit must appear in a
  committed notebook output, not in your memory of the run.

### Look at what you built

**Render the deck and look at it before calling a lesson finished.** Every layout
fault in lesson 1 - figures printed over text, a title shrunk to 15pt, a table split
across pages, captions growing upwards into the boxes above them - was invisible in
the markdown and obvious in the render. None would have been caught by reading the
source.

```bash
soffice --headless --convert-to pdf deck.pptx
pdftoppm -png -r 50 deck.pdf slide
```

That gives one PNG per slide. Check at minimum: the title slide, every slide carrying
a figure, and any slide with a table or a code block.

The same applies to handouts: render the first pages of the PDF rather than trusting
that the markdown implies the layout.

### Run the course environment

```bash
docker compose up --build
```

Then open `http://127.0.0.1:8888/lab?token=aicourse`.

The repository is bind-mounted at `/home/jovyan/work`, shadowing the snapshot baked
into the image. The `root` user plus `CHOWN_*` and the `fix-permissions` startup
command exist to fix "cannot save notebook" errors caused by host/container UID
mismatch. **Do not simplify that away.**

### Publish the student image

```bash
python tools/release.py patch --dry-run
python tools/release.py patch
```

`VERSION` is the **single source of truth**; the script bumps it and rewrites every
version reference in the documentation. Never edit those references by hand — silent
drift between the README and the student quickstart is precisely what went wrong in
the previous repository.

The image is republished only when **dependencies** change, not per lesson. New
lessons reach students through `git pull`.

---

## Conventions for content

### Notebooks

- Must run **top to bottom in a clean container**, with no API key and no manual
  setup. Datasets are either committed (small) or fetched via `sklearn`/`openml` with
  caching.
- Every figure used in the slides is **generated by a code cell** and exported to the
  lesson's `Figures/`. No screenshots, no images of unknown origin. This keeps figures
  reproducible and slides consistent with the labs.
- Setup cell opens with a commented `pip install` line carrying pinned versions.
- **Commit the notebooks with their outputs.** Two reasons: the handout and the
  slides quote numbers that come from these cells, so the committed output is what
  makes those claims checkable; and a student browsing the repository on GitHub sees
  the results without running anything. The cost is diff noise, which is why figures
  are regenerated in the container only — see "Where the work happens".

### Slides markdown

`Course/slides_syntax_example.md` is the reference for every construct that survives
the pandoc conversion: title metadata, bullets, figures, tables, code blocks, inline
and display math, two-column layouts via `::: columns`, and speaker notes. Start from
it when in doubt, and rebuild it if you change the template.

**One block element per slide.** pandoc ends a slide at any block: a figure, a
display equation, a fenced `::: columns` layout. Put two on one slide and the
second lands on a new, untitled one. So a figure cannot share a slide with an
equation or a two-column block - pick the one that does more work in the room
and move the other to the handout or to a slide of its own. `tools/build.py`
warns when prose follows a block, but it cannot see the other combinations, so
check the slide count after a rebuild.

**Speaker notes are expected on every content slide.** They go in a `notes` div and
never appear on the slide itself — they surface in PowerPoint's presenter view and on
printed notes pages:

```markdown
# Gradient descent

- Update rule and intuition

::: notes
What you plan to say, the question to put to the room, and a pointer to the
handout section carrying the derivation.
:::
```

They are what lets the slide stay uncluttered while the lecture still has its thread:
the slide shows the result, the notes carry the delivery, the handout holds the proof.

### Mathematics in slides

**Do not expect LaTeX to survive as equations.** pandoc converts `$...$` into OMML that
no renderer handles: LibreOffice drops the whole slide body and PowerPoint reports
damaged content and strips it. This was verified against pandoc's own default
template, so it is not something the course template causes.

`tools/render_math.py` therefore takes the math out before pandoc runs:

- **inline** math becomes Unicode (`$x^2$` → `x²`, `$\tfrac{1}{2}$` → `½`,
  `$\frac{1}{2m}$` → `1/(2m)`) and stays editable text
- **display** math is rendered to a transparent PNG in the lesson's `Figures/`

You still write ordinary LaTeX in the source — the conversion happens on a temporary
copy, and handouts keep real LaTeX because a TeX engine renders them properly.

Two rules follow from how pandoc builds slides, and the build warns about both:

- **Display math must be the last thing on its slide.** A block image ends the slide,
  so a sentence written after an equation lands on a new, untitled one.
- **Inline math must be Unicode-representable.** There is no image fallback inline:
  an image inside a paragraph is dropped during conversion, leaving a hole in the
  sentence. If the build says a formula has no Unicode form, simplify it, promote it
  to display position, or leave it to the handout — where it belongs anyway.

### Quizzes

Markdown cells only, no code. Each question uses this exact collapsible format:

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

Numbered sequentially across the notebook; each section gets its own `##` header cell.

---

## Git conventions

- Imperative, present tense: `Add lesson 3 handout`, `Fix resource path on Windows`.
- `core.ignorecase = true` on Windows ignores case-only renames; force them with an
  explicit `git mv`.
- Generated `.pptx` and `.pdf` **are committed** — this repository is the distribution
  channel and students must not need pandoc.
- Keep it lean: the repository is copied into the Docker image.

---

## Skills

### `/new-chapter`

Generates all five artefacts for a lesson in the conventions above.

```
/new-chapter <TopicName> <NN> "<brief description>" [--refs "url1, url2"] [--depth standard|deep]
```

See `.claude/commands/new-chapter.md`. It is the main accelerator for building this
course part-time — prefer it over creating files by hand.
