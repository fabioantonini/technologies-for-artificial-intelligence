"""Typeset the Italian study translations of the handouts.

    python tools/build_dispensa.py 01     # one lesson
    python tools/build_dispensa.py        # every one that exists

Source is ``Lessons/NN_*/Review/dispensa_it.md``, output the PDF beside it.

**These are not course artefacts.** The English handout in ``Docs/`` is the
official text and the one students receive; this is the instructor's own
revision copy, which is why it lives in ``Review/`` — already excluded from the
repository, already where his Italian working material goes. The course's
English-only rule is untouched, and nothing here is ever shipped.

The pandoc invocation deliberately matches ``build.py``'s handout path, so the
two read as the same document in two languages: same margins, same body size,
same table of contents. ``Figures/`` is on the resource path because the
translation embeds the lesson's own figures rather than copies of them.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINES = ("xelatex", "lualatex", "pdflatex", "tectonic")


def engine() -> str | None:
    return next((e for e in ENGINES if shutil.which(e)), None)


def build(source: Path, tex: str) -> bool:
    lesson = source.parent.parent
    pdf = source.with_suffix(".pdf")
    print(f"  {source.relative_to(ROOT)} -> {pdf.name}")
    result = subprocess.run(
        ["pandoc", str(source), "-o", str(pdf),
         f"--pdf-engine={tex}", "--toc",
         "-V", "geometry:margin=2.5cm",
         "-V", "fontsize=11pt",
         "-V", "colorlinks=true",
         "-V", "lang=it",
         f"--resource-path={source.parent}:{lesson / 'Figures'}"],
        capture_output=True, text=True)
    if result.returncode:
        tail = (result.stderr or result.stdout).strip().splitlines()[-3:]
        print("    FAILED: " + " ".join(tail))
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("lesson", nargs="?", help="lesson number, e.g. 01")
    args = parser.parse_args()

    if not shutil.which("pandoc"):
        print("pandoc not found on PATH", file=sys.stderr)
        return 1
    tex = engine()
    if tex is None:
        print(f"no LaTeX engine found; tried {', '.join(ENGINES)}",
              file=sys.stderr)
        return 1

    pattern = f"Lessons/{int(args.lesson):02d}_*" if args.lesson else "Lessons/*"
    sources = sorted(p / "Review" / "dispensa_it.md" for p in ROOT.glob(pattern))
    sources = [p for p in sources if p.exists()]
    if not sources:
        where = f"lesson {args.lesson}" if args.lesson else "any lesson"
        print(f"no Review/dispensa_it.md for {where}")
        return 1

    built = sum(build(source, tex) for source in sources)
    print(f"\n{built} of {len(sources)} built")
    return 0 if built == len(sources) else 1


if __name__ == "__main__":
    sys.exit(main())
