"""Check a finished lesson before it reaches a student.

    python tools/verify_lesson.py 05          # the mechanical checks
    python tools/verify_lesson.py 05 --run    # also execute every notebook
    python tools/verify_lesson.py             # every lesson that exists

The rule this exists to enforce: **a student should never be the one who finds
the error.**

What it can and cannot do is worth being precise about, because the failure that
prompted it would have slipped through a naive version. Lesson 3 once claimed
that X-transpose y was 165,200 when the data gives 165,600. Every number after
it — the determinant, the inverse, the two components of theta — was derived
*correctly from the wrong figure*, so the example was perfectly self-consistent
and read as careful work. No consistency check catches that. Only recomputing
from the raw data does.

So there are two kinds of check here, and only one of them is automatic:

**Mechanical** — the checks in this file. Notebooks that run, figures that
exist, formulas that survive conversion, slide counts that match the lesson
plan, acronyms expanded, cross-references that resolve. These run on every
lesson, every time, and cost nothing.

**Arithmetic** — a per-lesson ``Docs/worked_examples.py``. Every number a
handout works out by hand must be recomputed there from the raw inputs, by a
route that does not reuse the handout's intermediate values, and asserted
against what the handout prints. This file discovers and runs those; it cannot
write them, and a lesson that works numbers by hand without one is reported as
incomplete.

Exit code is 0 when everything passes, 1 otherwise, so it can gate a commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTAINER = "tai_course"

DOLLAR = chr(36)
DISPLAY_MATH = re.compile(re.escape(DOLLAR * 2) + "(.+?)" + re.escape(DOLLAR * 2), re.S)
INLINE_MATH = re.compile(
    "(?<!" + re.escape(DOLLAR) + ")" + re.escape(DOLLAR)
    + "([^" + DOLLAR + "\n]+?)" + re.escape(DOLLAR)
    + "(?!" + re.escape(DOLLAR) + ")")
FIGURE_REF = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

#: Acronyms the course expands on first use in every artefact. Kept here rather
#: than in a doc so the check and the rule cannot drift apart.
ACRONYMS = {
    "MSE": ("mean squared error",),
    "RMSE": ("root mean squared error",),
    "MCAR": ("missing completely at random",),
    "MAR": ("missing at random",),
    "MNAR": ("missing not at random", "not missing at random"),
    "IQR": ("interquartile range",),
    "OLS": ("ordinary least squares",),
    "AUC": ("area under",),
    "ROC": ("receiver operating",),
    "PCA": ("principal component",),
    "SVM": ("support vector",),
    "EDA": ("exploratory data analysis",),
    "API": ("application programming interface",),
    "MLE": ("maximum likelihood",),
    "SMART": ("self monitoring",),
    "TPR": ("true positive rate",),
    "FPR": ("false positive rate",),
}


class Report:
    """Collects problems so one run reports all of them, not just the first."""

    def __init__(self) -> None:
        self.problems: list[str] = []
        self.notes: list[str] = []

    def fail(self, check: str, detail: str) -> None:
        self.problems.append(f"{check}: {detail}")

    def note(self, detail: str) -> None:
        self.notes.append(detail)

    @property
    def ok(self) -> bool:
        return not self.problems


def markdown_of(path: Path) -> str:
    """The prose of a file: markdown cells only, for a notebook."""
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text(encoding="utf8"))
        return "\n".join("".join(c.get("source", []))
                         for c in nb.get("cells", [])
                         if c.get("cell_type") == "markdown")
    return path.read_text(encoding="utf8")


def shipped(paths) -> list[Path]:
    """Those of `paths` that a student will actually receive.

    Instructor-only content is marked by a git ignore rule: the worked
    exercise solutions, generated for whoever marks the work and never
    committed. They are held to the same standards while being written, but
    not by this gate, which exists to check what reaches a student. A red
    result on a file no student will open is a red result people learn to
    scroll past.

    Git is asked about the paths in hand, on every call, rather than once into
    a cache at start-up. A `--run` pass spends twenty minutes executing
    notebooks, which is long enough for an instructor file to be written while
    it is still going; a set built before that file existed reports it as
    shippable. That is how lesson 9's solution notebook came to be checked,
    and failed, for an acronym no student will ever read.
    """
    paths = list(paths)
    if not paths:
        return []
    try:
        # check-ignore exits 1 when nothing matches, which is not an error,
        # so this deliberately does not pass check=True.
        result = subprocess.run(
            ["git", "check-ignore", "--stdin"], cwd=ROOT,
            capture_output=True, text=True,
            input="\n".join(str(p) for p in paths))
    except (FileNotFoundError, OSError):
        return paths                      # not a checkout: check everything
    skip = {Path(line).resolve() for line in result.stdout.splitlines() if line}
    return [p for p in paths if p.resolve() not in skip]


def artefacts(lesson: Path) -> list[Path]:
    return shipped(p for p in sorted(lesson.rglob("*"))
                   if p.suffix in (".md", ".ipynb")
                   and "checkpoint" not in str(p)
                   and "__pycache__" not in str(p))


# --------------------------------------------------------------- notebooks

def check_notebooks(lesson: Path, report: Report, run: bool) -> None:
    notebooks = sorted((lesson / "Notebooks").glob("*.ipynb"))
    notebooks += sorted((lesson / "Quizzes").glob("*.ipynb"))
    notebooks += shipped(sorted((lesson / "Exercises").glob("*.ipynb")))
    if not notebooks:
        report.fail("notebooks", "none found")
        return

    for path in notebooks:
        name = path.name
        try:
            nb = json.loads(path.read_text(encoding="utf8"))
        except json.JSONDecodeError as error:
            report.fail("notebooks", f"{name} is not valid JSON: {error}")
            continue

        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        is_quiz = "Quizzes" in path.parts

        if is_quiz:
            if code_cells:
                report.fail("quiz", f"{name} has {len(code_cells)} code cells; "
                                    "quizzes are markdown only")
            continue

        if not code_cells:
            report.fail("notebooks", f"{name} has no code cells")
            continue

        # A stored traceback means the committed notebook is broken.
        for index, cell in enumerate(code_cells, 1):
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    report.fail("notebooks",
                                f"{name} cell {index} stored an error: "
                                f"{output.get('ename')}")

        # Outputs are committed on purpose (see CLAUDE.md); missing ones mean
        # the notebook was edited and never re-run.
        without = [i for i, c in enumerate(code_cells, 1) if not c.get("outputs")]
        if len(without) > 1:
            report.fail("notebooks",
                        f"{name}: {len(without)} code cells have no output "
                        f"(cells {without[:6]}) — re-run before committing")

        # The pinned versions must be the ones the image actually ships.
        source = "".join("".join(c["source"]) for c in code_cells)
        if "pip install" not in source:
            report.fail("notebooks", f"{name} has no pinned pip install line")

    if run:
        for path in notebooks:
            if "Quizzes" in path.parts:
                continue
            relative = path.relative_to(ROOT)
            # Execute into a scratch directory, never over the committed file:
            # --inplace would rewrite execution counts and kernel metadata, so
            # checking a lesson would leave every notebook modified.
            result = subprocess.run(
                ["docker", "exec", "-w", f"/home/jovyan/work/{relative.parent}",
                 CONTAINER, "jupyter", "nbconvert", "--to", "notebook",
                 "--execute", "--output-dir", "/tmp/verify",
                 "--ExecutePreprocessor.timeout=1800", path.name],
                capture_output=True, text=True)
            if result.returncode != 0:
                tail = result.stderr.strip().splitlines()[-3:]
                report.fail("notebooks --run",
                            f"{path.name} failed: {' | '.join(tail)}")
            else:
                report.note(f"runs clean: {path.name}")


def check_pins(lesson: Path, report: Report) -> None:
    """Pinned versions must match what the container actually has."""
    result = subprocess.run(
        ["docker", "exec", CONTAINER, "python", "-c",
         "import numpy, pandas, sklearn, matplotlib; "
         "print(numpy.__version__, pandas.__version__, "
         "sklearn.__version__, matplotlib.__version__)"],
        capture_output=True, text=True)
    if result.returncode != 0:
        report.note("container not running; skipped the pinned-version check")
        return

    numpy_v, pandas_v, sklearn_v, matplotlib_v = result.stdout.split()
    expected = {"numpy": numpy_v, "pandas": pandas_v,
                "scikit-learn": sklearn_v, "matplotlib": matplotlib_v}

    for path in artefacts(lesson):
        if path.suffix != ".ipynb":
            continue
        text = path.read_text(encoding="utf8")
        for package, version in expected.items():
            for found in re.findall(package + r"==([0-9][0-9.]*)", text):
                if found != version:
                    report.fail("pinned versions",
                                f"{path.name} pins {package}=={found}, "
                                f"the image has {version}")


# ----------------------------------------------------------------- figures

def check_figures(lesson: Path, report: Report) -> None:
    figures_dir = lesson / "Figures"
    if not figures_dir.exists():
        return

    referenced: set[str] = set()
    for path in artefacts(lesson):
        for name in FIGURE_REF.findall(markdown_of(path)):
            referenced.add(Path(name).name)
            if not (figures_dir / Path(name).name).exists():
                report.fail("figures",
                            f"{path.name} references {name}, which is missing")

    on_disk = {p.name for p in figures_dir.glob("*.png")}

    # Equation images are generated; an unused one is stale output.
    deck = next(iter((lesson / "Slides").glob("*.pptx")), None)
    if deck is not None:
        embedded = set()
        with zipfile.ZipFile(deck) as archive:
            for entry in archive.namelist():
                if entry.startswith("ppt/media/"):
                    embedded.add(hashlib.sha256(archive.read(entry)).hexdigest())
        for path in sorted(figures_dir.glob("eq_*.png")):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest not in embedded:
                report.fail("figures", f"{path.name} is an orphaned equation image")

    # Data figures come from notebooks, conceptual ones from make_figures.py.
    produced = ""
    for path in sorted((lesson / "Notebooks").glob("*.ipynb")):
        produced += path.read_text(encoding="utf8")
    maker = figures_dir / "make_figures.py"
    if maker.exists():
        produced += maker.read_text(encoding="utf8")

    for name in sorted(on_disk):
        if name.startswith("eq_"):
            continue
        if name not in produced:
            report.fail("figures",
                        f"{name} is not produced by any notebook or "
                        "make_figures.py — its origin is unreproducible")
        elif name not in referenced:
            report.note(f"{name} is generated but never shown")


    # The handout is what gets read after the lecture, when the projected
    # figures are gone. Every figure the slides show, it shows too.
    slide_figures: set[str] = set()
    for path in sorted((lesson / "Slides").glob("*_slides.md")):
        slide_figures |= {Path(n).name for n in FIGURE_REF.findall(
            path.read_text(encoding="utf8"))}

    # Reading material means the handout or the supplementary Resources:
    # the history figures belong with the history essay, not crammed into
    # the one section of the handout that summarises it.
    handout_figures: set[str] = set()
    reading = sorted((lesson / "Docs").glob("*.md")) + sorted(
        (lesson / "Resources").glob("*.md"))
    for path in reading:
        handout_figures |= {Path(n).name for n in FIGURE_REF.findall(
            path.read_text(encoding="utf8"))}

    # Equation images are excluded: the handout renders real LaTeX.
    missing = {n for n in slide_figures - handout_figures
               if not n.startswith("eq_")}
    for name in sorted(missing):
        report.fail("handout figures",
                    f"{name} is on a slide but not in the handout")


    # Referencing a figure is not the same as shipping one. pandoc drops an
    # image it cannot find without a word of complaint, so look in the PDF.
    for handout in sorted((lesson / "Docs").glob("*.md")):
        referenced = {Path(n).name for n in FIGURE_REF.findall(
            handout.read_text(encoding="utf8"))}
        pdf = handout.with_suffix(".pdf")
        if not referenced or not pdf.exists():
            continue
        embedded = len(re.findall(rb"/Subtype\s*/Image", pdf.read_bytes()))
        if embedded < len(referenced):
            report.fail("handout figures",
                        f"{handout.name} references {len(referenced)} figures "
                        f"but {pdf.name} embeds {embedded} images — pandoc "
                        "could not find them")


# ------------------------------------------------------------------ slides

def check_slides(lesson: Path, report: Report) -> None:
    sources = sorted((lesson / "Slides").glob("*_slides.md"))
    if not sources:
        report.fail("slides", "no slides source found")
        return
    source = sources[0]
    text = source.read_text(encoding="utf8")

    # Inline maths must survive the Unicode conversion; there is no image
    # fallback inline, so anything else silently mangles on the slide.
    sys.path.insert(0, str(ROOT / "tools"))
    import render_math

    for match in INLINE_MATH.finditer(text):
        formula = match.group(1)
        rendered = render_math.to_unicode(formula)
        if rendered is None or chr(92) in str(rendered):
            report.fail("slides maths",
                        f"inline formula has no Unicode form: {DOLLAR}{formula}{DOLLAR}")

    deck = source.with_suffix(".pptx")
    if not deck.exists():
        report.fail("slides", f"{deck.name} has not been built")
        return

    from pptx import Presentation
    slides = list(Presentation(deck).slides)

    for index, slide in enumerate(slides, 1):
        has_text = any(sh.has_text_frame and sh.text_frame.text.strip()
                       for sh in slide.shapes)
        has_picture = any(sh.shape_type == 13 for sh in slide.shapes)
        has_table = any(sh.has_table for sh in slide.shapes)
        if not (has_text or has_picture or has_table):
            report.fail("slides", f"slide {index} is empty")
        if not slide.shapes.title or not slide.shapes.title.text.strip():
            report.fail("slides", f"slide {index} has no title")

        if index == 1:
            continue
        notes = ""
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
        title = slide.shapes.title.text.strip() if slide.shapes.title else ""
        if len(notes) < 60 and title.lower() not in ("break",):
            report.fail("slides",
                        f"slide {index} ({title[:34]}) has thin speaker notes")

    check_lesson_plan(lesson, len(slides), report)


def check_lesson_plan(lesson: Path, slide_count: int, report: Report) -> None:
    handouts = sorted((lesson / "Docs").glob("*.md"))
    if not handouts:
        report.fail("handout", "no handout found")
        return
    plan = handouts[0].read_text(encoding="utf8")

    minutes = 0
    for h1, m1, h2, m2 in re.findall(r"\|\s*(\d):(\d\d)[-–](\d):(\d\d)\s*\|", plan):
        minutes += (int(h2) * 60 + int(m2)) - (int(h1) * 60 + int(m1))
    if minutes != 180:
        report.fail("lesson plan",
                    f"segments sum to {minutes} minutes, not 180")

    cited = []
    for start, end in re.findall(r"Slides? (\d+)(?:[-–](\d+))?", plan):
        cited.append(int(start))
        if end:
            cited.append(int(end))
    if cited and max(cited) > slide_count:
        report.fail("lesson plan",
                    f"cites slide {max(cited)} but the deck has {slide_count}")


# --------------------------------------------------------------- acronyms

def normalise(text: str) -> str:
    """Lower-case, hyphens and underscores to spaces, whitespace collapsed."""
    cleaned = text.replace("*", " ").replace("-", " ").replace("_", " ")
    return " ".join(cleaned.split()).lower()


def strip_maths(text: str) -> str:
    """Remove formulas and code spans before looking for acronyms.

    An acronym inside a formula is notation, not prose: TPR appearing in a
    display equation does not introduce the term to anybody. The sentence
    around it does, and that sentence is what this check is about.
    """
    text = DISPLAY_MATH.sub(" ", text)
    text = INLINE_MATH.sub(" ", text)
    return re.sub(r"`[^`]*`", " ", text)


def check_acronyms(lesson: Path, report: Report) -> None:
    for path in artefacts(lesson):
        body = strip_maths(markdown_of(path))
        for acronym, expansions in ACRONYMS.items():
            match = re.search(rf"\b{acronym}\b", body)
            if not match:
                continue
            line_end = body.find("\n", match.end())
            prefix = normalise(body[:line_end if line_end != -1 else len(body)])
            if not any(normalise(e) in prefix for e in expansions):
                report.fail("acronyms",
                            f"{path.name} uses {acronym} before expanding it")


# -------------------------------------------------------------------- quiz

#: Question stems that are legitimately imperative rather than interrogative.
#: Half of lesson 5's are ("Describe k-fold cross-validation..."), so the check
#: has to accept them or it becomes noise nobody reads.
IMPERATIVES = ("define", "state", "describe", "list", "explain", "compute",
               "summarise", "summarize", "draw", "give", "write", "identify",
               "distinguish", "derive", "compare", "name", "read", "take",
               "prove", "argue", "diagnose", "prescribe", "justify", "sketch",
               "show", "contrast", "resolve", "criticise", "rank", "report",
               "propose", "outline", "discuss", "interpret", "decide", "choose")

def asks_something(stem: str) -> bool:
    """Does this stem actually pose a question or give an instruction?

    Deliberately generous. Most stems set a scene first and ask afterwards, so
    the ask is looked for anywhere rather than at either end. What this rejects
    is a stem that describes a situation and never gets round to the question.
    """
    if "?" in stem:
        return True
    lowered = stem.lower()
    return any(re.search(rf"\b{verb}\b", lowered) for verb in IMPERATIVES)


def question_stem(source: str) -> str:
    """The first paragraph of a question cell, with markup removed.

    Not "everything up to the next **": a stem containing bold would be cut off
    at the first one, which is how this check first reported four false
    positives in lesson 1.
    """
    paragraph = source.strip().split("\n\n")[0]
    paragraph = re.sub(r"^\*\*\d+\.\s*", "", paragraph.strip())

    # Order matters. The whole stem is bold, so its closing asterisks sit after
    # the difficulty marker; strip the markup first or the anchored pattern
    # below never reaches the end of the string.
    paragraph = re.sub(r"<[^>]+>", " ", paragraph)
    paragraph = paragraph.replace("*", " ").replace("`", " ")
    paragraph = re.sub(r"\(\s*(reasoning|recall|calculation)\s*\)\s*$", "",
                       " ".join(paragraph.split()), flags=re.I)
    return " ".join(paragraph.split())


def check_quiz(lesson: Path, report: Report) -> None:
    quizzes = sorted((lesson / "Quizzes").glob("*.ipynb"))
    if not quizzes:
        report.fail("quiz", "no quiz found")
        return

    for path in quizzes:
        name = path.name
        try:
            nb = json.loads(path.read_text(encoding="utf8"))
        except json.JSONDecodeError:
            continue        # check_notebooks already reported it

        numbers: list[int] = []
        sections = 0

        for cell in nb["cells"]:
            source = "".join(cell.get("source", []))
            stripped = source.lstrip()

            if stripped.startswith("## "):
                sections += 1

            starts = re.findall(r"^\*\*(\d+)\.", source, re.M)
            if not starts:
                continue
            if len(starts) > 1:
                report.fail("quiz",
                            f"{name}: one cell holds questions "
                            f"{', '.join(starts)} — each gets its own cell")
            numbers.append(int(starts[0]))

            # The collapsible answer, in the house format.
            if source.count("<details>") != source.count("</details>"):
                report.fail("quiz",
                            f"{name} question {starts[0]}: <details> is not closed")
            if source.count("<details>") != 1:
                report.fail("quiz",
                            f"{name} question {starts[0]}: expected exactly one "
                            f"answer block, found {source.count('<details>')}")
            if "<summary>" not in source or "color='darkgreen'" not in source:
                report.fail("quiz",
                            f"{name} question {starts[0]}: the answer summary is "
                            "not in the course format")

            # An answer that says nothing is worse than no answer, because it
            # looks answered. Length rather than bullet count: a question that
            # asks for the rule "in one sentence" deserves one bullet.
            points = re.findall(r"<li>(.*?)</li>", source, re.S)
            answered = sum(len(re.sub(r"<[^>]+>", "", p).strip()) for p in points)
            if not points:
                report.fail("quiz",
                            f"{name} question {starts[0]}: the answer has no points")
            elif answered < 80:
                report.fail("quiz",
                            f"{name} question {starts[0]}: the answer is a stub "
                            f"({answered} characters)")

            stem = question_stem(source)
            if stem and not asks_something(stem):
                report.fail("quiz",
                            f"{name} question {starts[0]}: the stem states a "
                            f"situation but never asks anything — {stem[:52]}")

        if numbers != list(range(1, len(numbers) + 1)):
            report.fail("quiz",
                        f"{name}: questions are numbered {numbers[:8]}... — "
                        "they must run consecutively from 1")
        if sections == 0:
            report.fail("quiz", f"{name}: no '## ' section headers")
        if len(numbers) < 15:
            report.fail("quiz",
                        f"{name}: only {len(numbers)} questions; a lesson's quiz "
                        "carries at least 15")


# ------------------------------------------------------- worked arithmetic

def check_worked_examples(lesson: Path, report: Report) -> None:
    """Run the lesson's own recomputation of every hand-worked number."""
    checker = lesson / "Docs" / "worked_examples.py"
    handouts = sorted((lesson / "Docs").glob("*.md"))
    handout = handouts[0].read_text(encoding="utf8") if handouts else ""

    # Does the handout work any arithmetic by hand at all?
    works_numbers = bool(
        re.search(r"worked example|by hand|A worked step", handout, re.I))

    if not checker.exists():
        if works_numbers:
            report.fail("worked examples",
                        "the handout works numbers by hand but there is no "
                        "Docs/worked_examples.py recomputing them")
        return

    result = subprocess.run([sys.executable, str(checker)],
                            capture_output=True, text=True, cwd=str(ROOT))
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()[-4:]
        report.fail("worked examples", " | ".join(tail))
    else:
        for line in result.stdout.strip().splitlines():
            report.note(line)


# ------------------------------------------------------- cross-references

def check_cross_references(lesson: Path, report: Report) -> None:
    notebooks = {p.name[:2] for p in (lesson / "Notebooks").glob("*.ipynb")}
    for path in artefacts(lesson):
        body = markdown_of(path)
        for number in re.findall(r"[Nn]otebook (\d)\b", body):
            if f"0{number}" not in notebooks:
                report.fail("cross-references",
                            f"{path.name} mentions notebook {number}, "
                            "which does not exist")


# -------------------------------------------------------------------- main

def verify(lesson: Path, run: bool) -> Report:
    report = Report()
    check_notebooks(lesson, report, run)
    check_pins(lesson, report)
    check_figures(lesson, report)
    check_slides(lesson, report)
    check_quiz(lesson, report)
    check_acronyms(lesson, report)
    check_worked_examples(lesson, report)
    check_cross_references(lesson, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lesson", nargs="?", help="lesson number, e.g. 05")
    parser.add_argument("--run", action="store_true",
                        help="also execute every notebook in the container")
    args = parser.parse_args()

    lessons = sorted(ROOT.glob("Lessons/[0-9][0-9]_*"))
    if args.lesson:
        lessons = [p for p in lessons if p.name.startswith(args.lesson)]
        if not lessons:
            print(f"no lesson matching {args.lesson}")
            return 1
        # A lesson with no handout is dropped from the sweep below, which is
        # right when verifying everything: one nobody has started yet is not a
        # defect. Asked for by name it is the opposite - lesson 10, stopped
        # halfway through phase A, reported "0 verificate, nessun problema" and
        # exited 0, which reads as a pass on work that does not exist.
        unbuilt = [p for p in lessons if not any(p.glob("Docs/*.md"))]
        if unbuilt:
            for path in unbuilt:
                print(f"{path.name}  —  NON COSTRUITA: nessuna dispensa in "
                      f"Docs/, quindi non c'è niente da verificare.")
            print("\n    .  python .claude/skills/new-lesson/scripts/"
                  f"lesson_state.py {args.lesson}  dice in che fase è.")
            return 1
    else:
        lessons = [p for p in lessons if any(p.glob("Docs/*.md"))]

    failed = 0
    for lesson in lessons:
        report = verify(lesson, args.run)
        status = "OK" if report.ok else f"{len(report.problems)} PROBLEMI"
        print(f"\n{lesson.name}  —  {status}")
        for problem in report.problems:
            print(f"    x  {problem}")
        for note in report.notes:
            print(f"    .  {note}")
        if not report.ok:
            failed += 1

    print()
    if failed:
        print(f"{failed} lezione/i con problemi")
        return 1
    print(f"{len(lessons)} lezione/i verificate, nessun problema")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
