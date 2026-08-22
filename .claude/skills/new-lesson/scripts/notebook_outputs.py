#!/usr/bin/env python3
"""Print what a notebook's cells actually printed, beside the prose that cites it.

    python .claude/skills/new-lesson/scripts/notebook_outputs.py Lessons/06_*/Notebooks
    python .claude/skills/new-lesson/scripts/notebook_outputs.py path/to/one.ipynb

Phase D step 3. Notebook prose is written before the cell below it runs, so a
number in the text is a prediction until someone checks it against the output. In
lesson 6 three such predictions were wrong and were contradicted by the cell
directly above them; a fourth, in the instructor's solution, asserted a ratio
"is worth a great deal" beneath a printed correlation of 0.079.

Reading a notebook in the editor does not catch these, because the eye slides
over an output it has already decided it knows. Laid out as text, side by side,
they are hard to miss.

Numbers appearing in the markdown are marked so they can be found quickly; the
marking is a reading aid, not a check - a number can be present and still wrong.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w])")

#: Ordered-list markers and heading numbers, which are formatting rather than
#: claims. Left in, they bury the numbers actually worth checking.
LIST_MARKER = re.compile(r"^\s{0,3}\d+[.)]\s|^#+\s*\d+[.)]?\s", re.MULTILINE)

#: A print() call, and the {...} placeholders inside an f-string. Stripping the
#: placeholders leaves the part of the message that was typed rather than
#: computed - and a decimal surviving that is a number being stated.
PRINT_CALL = re.compile(r"print\s*\((.*?)\)\s*$", re.MULTILINE | re.DOTALL)
PLACEHOLDER = re.compile(r"\{[^{}]*\}")
DECIMAL = re.compile(r"(?<![\w.])\d+\.\d+")


def asserted_numbers(source: str) -> list[str]:
    """Decimals typed into a print(), which the output will then appear to confirm.

    `print(f"mean {x:.4f}")` computes its number and is fine. `print("mean
    0.0208")` states one, and step 4 cannot tell the difference downstream -
    prose and output agree because the same literal produced both.
    """
    found = []
    for call in PRINT_CALL.findall(source):
        typed = PLACEHOLDER.sub(" ", call)
        found += DECIMAL.findall(typed)
    return found


def outputs_of(cell: dict) -> str:
    parts = []
    for out in cell.get("outputs", []):
        if "text" in out:
            parts.append("".join(out["text"]))
        data = out.get("data", {})
        if "text/plain" in data:
            parts.append("".join(data["text/plain"]))
        if "image/png" in data and "text/plain" not in data:
            parts.append("<figure>")
    return "".join(parts).rstrip()


def report(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf8"))
    print("=" * 72)
    print(path.name)
    print("=" * 72)

    pending: list[str] = []
    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown":
            pending.append(source)
            continue

        text = outputs_of(cell)
        if not text:
            continue

        prose = "\n\n".join(pending[-2:]).strip()
        pending = []

        if prose:
            claimed = sorted(set(NUMBER.findall(LIST_MARKER.sub(" ", prose))))
            print("\n--- prose above ---")
            print(prose[:900] + ("..." if len(prose) > 900 else ""))
            if claimed:
                print(f"    numbers it states: {', '.join(claimed)}")
        print("\n--- what ran below it printed ---")
        print(text[:1500] + ("..." if len(text) > 1500 else ""))
        # Only the ones the prose above also states. A constant echoed into a
        # message nobody cites is harmless; the failure is prose and output
        # agreeing because one literal produced both, and that needs the number
        # to appear in both places.
        asserted = {n for n in asserted_numbers(source) if n in prose}
        if asserted:
            print(f"    !! the prose above states these, and the print below "
                  f"was TYPED with them rather than computing them: "
                  f"{', '.join(sorted(asserted))}")
            print("       agreement here proves nothing - one literal made both")
        print()


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2

    paths: list[Path] = []
    for name in argv:
        target = Path(name)
        if target.is_dir():
            paths += sorted(p for p in target.glob("*.ipynb")
                            if "checkpoint" not in str(p))
        elif target.exists():
            paths.append(target)
        else:  # let the shell's glob fail loudly rather than silently skipping
            print(f"missing: {target}", file=sys.stderr)
            return 1

    if not paths:
        print("no notebooks found", file=sys.stderr)
        return 1

    for path in paths:
        report(path)

    print("Read each block against the one above it. A number in the prose that "
          "does not appear\nin the output is either derived, stale, or invented "
          "- and the third is common enough\nto be worth checking every time.")
    print("\nAnything marked !! was typed into the print rather than computed "
          "by it. Prose and\noutput agree there no matter what, because the "
          "same literal produced both.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
