#!/usr/bin/env python3
"""Say which phase a lesson is in, by looking at what is on disk.

    python .claude/skills/new-lesson/scripts/lesson_state.py 07

A lesson takes long enough that the session building it is often summarised
partway through. Anything about "where we got to" that lives only in the
conversation is lost when that happens, so nothing is kept there: each phase ends
by writing files, and this reads the files back.

Run it when starting, when resuming, and whenever unsure. It is cheap and it
never guesses - every line it prints is something it found or did not find.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def find_lesson(number: str) -> Path:
    matches = sorted(ROOT.glob(f"Lessons/{number}_*"))
    if not matches:
        raise SystemExit(f"no lesson folder matching Lessons/{number}_*")
    return matches[0]


def executed(path: Path) -> bool:
    """True if the notebook carries saved outputs for at least one code cell."""
    try:
        nb = json.loads(path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError):
        return False
    return any(cell.get("outputs") for cell in nb.get("cells", [])
               if cell.get("cell_type") == "code")


def figures_referenced(path: Path) -> set[str]:
    """Basenames of the figures a document shows.

    Basenames, not the written paths: lesson 8's handout referenced
    `../Figures/x.png` where every other lesson writes `x.png`, and comparing
    the raw strings made this report nine orphaned figures that were not
    orphaned at all. `verify_lesson.py` normalises, so the two tools disagreed
    silently until someone read both.
    """
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf8")
    return {Path(m).name
            for m in re.findall(r"!\[[^\]]*\]\(([^)]+\.png)\)", text)
            if not Path(m).name.startswith("eq_")}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2

    lesson = find_lesson(argv[0].zfill(2))
    print(f"{lesson.name}\n")

    # ---------------------------------------------------------------- phase A
    data_modules = [p for p in lesson.glob("Notebooks/*.py")
                    if not p.name.startswith("_")]
    notebooks = sorted(lesson.glob("Notebooks/*.ipynb"))
    unexecuted = [p for p in notebooks if not executed(p)]
    figures = sorted(p for p in lesson.glob("Figures/*.png")
                     if not p.name.startswith("eq_"))

    a_missing = []
    if not data_modules:
        a_missing.append("no dataset module in Notebooks/ (the *_data.py with "
                         "TRUE_* constants)")
    if not notebooks:
        a_missing.append("no notebooks")
    if unexecuted:
        a_missing.append("not executed: "
                         + ", ".join(p.name for p in unexecuted))
    if notebooks and not figures:
        a_missing.append("no figures in Figures/")

    # ---------------------------------------------------------------- phase B
    handouts = sorted(lesson.glob("Docs/*.md"))
    worked = lesson / "Docs" / "worked_examples.py"
    slides_src = sorted(lesson.glob("Slides/*_slides.md"))

    b_missing = []
    if not handouts:
        b_missing.append("no handout in Docs/")
    if not worked.exists():
        b_missing.append("no Docs/worked_examples.py")
    if handouts and slides_src:
        in_slides = figures_referenced(slides_src[0])
        in_handout = set()
        for handout in handouts:
            in_handout |= figures_referenced(handout)
        orphaned = in_slides - in_handout
        if orphaned:
            b_missing.append("figures in the slides but not the handout: "
                             + ", ".join(sorted(orphaned)))

    # ---------------------------------------------------------------- phase C
    quizzes = sorted(lesson.glob("Quizzes/*.ipynb"))
    exercises = [p for p in lesson.glob("Exercises/*.md")]
    resources = sorted(lesson.glob("Resources/*.md"))

    c_missing = []
    if not slides_src:
        c_missing.append("no slides")
    if not quizzes:
        c_missing.append("no quiz")
    if not exercises:
        c_missing.append("no exercise brief")
    if not resources:
        c_missing.append("no Resources/ document")

    # ---------------------------------------------------------------- report
    phases = [("A  data and notebooks", a_missing),
              ("B  handout", b_missing),
              ("C  slides, quiz, exercise, resources", c_missing)]

    current = None
    for name, missing in phases:
        mark = "." if not missing else "x"
        print(f"  {mark}  phase {name}")
        for item in missing:
            print(f"        - {item}")
        if missing and current is None:
            current = name
            done_before = phases.index((name, missing))

    solution = sorted(lesson.glob("Exercises/*_solution.ipynb"))
    print(f"  {'.' if solution else 'x'}  instructor solution "
          f"({'present, uncommitted' if solution else 'not written'})")

    print()
    if current is None:
        print("  -> phase D. Nothing is missing, so the work left is review:")
        print("       1. python tools/verify_lesson.py "
              f"{lesson.name[:2]} --run")
        print("       2. read every executed output against the prose citing it")
        print("       3. open the built PDF and look at each figure slide")
        print("       4. confirm two consecutive builds are byte-identical")
    else:
        letter = current.split()[0]
        if done_before:
            names = [p[0].split()[0] for p in phases[:done_before]]
            earlier = (f"Phase {names[0]} is" if len(names) == 1
                       else f"Phases {' and '.join(names)} are")
            print(f"  -> resume at phase {letter}. "
                  f"{earlier} complete; do not redo that work.")
            if letter != "A":
                # What this cannot see, and must not imply it can.
                print("       Except phase A step 4 - reading each notebook's"
                      " prose against its")
                print("       own output - which leaves no trace on disk, so"
                      " nothing here knows")
                print("       whether it happened. It is cheap and idempotent."
                      " Redo it:")
                print("         python .claude/skills/new-lesson/scripts"
                      "/notebook_outputs.py \\")
                print(f"             Lessons/{lesson.name}/Notebooks")
        else:
            print(f"  -> start at phase {letter}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
