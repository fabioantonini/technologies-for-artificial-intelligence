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
- Clear outputs before committing.

### Slides markdown

`Course/slides_syntax_example.md` is the reference for every construct that survives
the pandoc conversion: title metadata, bullets, figures, tables, code blocks, inline
and display math, two-column layouts via `::: columns`, and speaker notes. Start from
it when in doubt, and rebuild it if you change the template.

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
