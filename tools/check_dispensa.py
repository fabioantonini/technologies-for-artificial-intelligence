"""Check an Italian study translation against the handout it came from.

    python tools/check_dispensa.py 02     # one lesson
    python tools/check_dispensa.py        # every translation that exists

A translated handout that alters a digit is worse than no translation at all:
`Docs/worked_examples.py` gates the English text only, so nothing else would
ever catch it. Three comparisons, all of them mechanical:

- **Figures.** The same images, in the same order. The translation embeds the
  lesson's own `Figures/`, so a missing one is a section that lost its picture.
- **Numbers.** Every figure quoted in the English must appear in the Italian,
  and the Italian must invent none. Italian decimal commas are normalised
  before comparing, since 0,986 and 0.986 are the same number.
- **Stray Cyrillic.** A Latin `a` and a Cyrillic `а` are indistinguishable on
  screen and different to every tool. They arrive by accident and survive
  proofreading, so they are worth a machine's attention rather than an
  author's.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FIGURE = re.compile(r'!\[\]\(([^)]+)\)')
#: A number with any mixture of grouping and decimal separators, or a bare
#: integer of two digits or more. One digit is not worth comparing: "3 volte"
#: against "3 times" would fill the report with noise.
NUMBER = re.compile(r'\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|\d+[.,]\d+|\b\d{2,}\b')


def numbers(text: str) -> set:
    """Every quoted figure, reduced to one spelling.

    The two languages group and punctuate differently - 3,329.7 against
    3.329,7 against a bare 3329.7 - and all three are the same measurement, so
    the comparison has to see past the formatting or it reports nothing but
    formatting. The rule: the last separator followed by something other than
    exactly three digits is the decimal point; every other separator is
    grouping and is dropped. A lone separator with exactly three digits after
    it is grouping, which is the one genuinely ambiguous case and the reading
    that is right far more often.
    """
    found = set()
    for raw in NUMBER.findall(text):
        body = raw.replace(" ", "")
        parts = re.split(r'[.,]', body)
        if len(parts) == 1:
            found.add(body)
            continue
        if len(parts[-1]) == 3 and all(len(p) == 3 for p in parts[1:]):
            found.add("".join(parts))          # grouping all the way down
        else:
            found.add("".join(parts[:-1]) + "." + parts[-1])
    return found


def cyrillic(text: str) -> list:
    """Latin-looking letters that are not Latin, with the line they sit on."""
    out = []
    for number, line in enumerate(text.splitlines(), start=1):
        for character in line:
            if "CYRILLIC" in unicodedata.name(character, ""):
                out.append((number, character, line.strip()[:70]))
    return out


def check(lesson: Path) -> list:
    italian = lesson / "Review" / "dispensa_it.md"
    handouts = [p for p in sorted((lesson / "Docs").glob("*.md"))]
    if not italian.exists() or not handouts:
        return []
    english = handouts[0]

    en, it = english.read_text(encoding="utf-8"), italian.read_text(encoding="utf-8")
    problems = []

    fe, fi = FIGURE.findall(en), FIGURE.findall(it)
    if fe != fi:
        missing = [f for f in fe if f not in fi]
        extra = [f for f in fi if f not in fe]
        detail = []
        if missing:
            detail.append(f"missing {', '.join(missing)}")
        if extra:
            detail.append(f"not in the handout: {', '.join(extra)}")
        if not detail:
            detail.append("same figures, different order")
        problems.append(f"figures ({len(fe)} against {len(fi)}): " + "; ".join(detail))

    ne, ni = numbers(en), numbers(it)
    lost = sorted(ne - ni)
    if lost:
        problems.append(f"{len(lost)} number(s) in the handout and not in the "
                        f"translation: {', '.join(lost[:12])}"
                        + (" ..." if len(lost) > 12 else ""))
    invented = sorted(ni - ne)
    if invented:
        problems.append(f"{len(invented)} number(s) in the translation and not "
                        f"in the handout: {', '.join(invented[:12])}"
                        + (" ..." if len(invented) > 12 else ""))

    for line, character, context in cyrillic(it):
        problems.append(f"line {line}: Cyrillic {character!r} "
                        f"(U+{ord(character):04X}) in {context!r}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("lesson", nargs="?", help="lesson number, e.g. 02")
    args = parser.parse_args()

    pattern = f"Lessons/{int(args.lesson):02d}_*" if args.lesson else "Lessons/*"
    lessons = [p for p in sorted(ROOT.glob(pattern))
               if (p / "Review" / "dispensa_it.md").exists()]
    if not lessons:
        print("no translation to check")
        return 1

    total = 0
    for lesson in lessons:
        problems = check(lesson)
        total += len(problems)
        if problems:
            print(f"{lesson.name}  —  {len(problems)} to fix")
            for problem in problems:
                print(f"    x  {problem}")
        else:
            print(f"{lesson.name}  —  figures, numbers and characters all agree")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
