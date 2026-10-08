"""Shared machinery for the worked examples in a lesson's Review/lN-topics.md.

Each lesson keeps its examples in ``Lessons/NN_*/Review/lN_examples.py``: one
function per slide, decorated with ``@example(slide)``, returning the markdown of
a small worked example whose every number it has computed itself. This module
does the rest:

    from topics_examples import example, fmt, run
    ...
    if __name__ == "__main__":
        run(__file__)

``run`` prints the examples; with ``--write`` it splices each one into the
topics file under its slide, between markers, so a rerun replaces rather than
duplicates, and rebuilds the PDF. Where the lesson has a slide-by-slide
commentary written from the instructor's study session (``lN-commentary.md``),
that document replaces the topics and receives the examples instead.

Review/ is outside the repository - the examples are teaching notes for the
instructor, built from invented values small enough for the board - so only
this machinery is versioned, not the examples themselves.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

EXAMPLES: dict[int, str] = {}
MARK_OPEN, MARK_CLOSE = "<!-- example:begin -->", "<!-- example:end -->"


def example(slide: int):
    """Register the decorated function's markdown as the example for ``slide``."""
    def register(fn):
        if slide in EXAMPLES:
            raise SystemExit(f"two examples for slide {slide}")
        EXAMPLES[slide] = fn().strip()
        return fn
    return register


def fmt(value, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def splice(topics: Path) -> None:
    """Put every example under its slide's entry, replacing earlier versions."""
    text = topics.read_text(encoding="utf-8")
    text = re.sub(rf"\n*{re.escape(MARK_OPEN)}.*?{re.escape(MARK_CLOSE)}\n*", "\n\n",
                  text, flags=re.S)
    for slide, body in sorted(EXAMPLES.items()):
        heads = list(re.finditer(r"^## Slide (\d+) — .*$", text, flags=re.M))
        found = [i for i, m in enumerate(heads) if int(m.group(1)) == slide]
        if not found:
            raise SystemExit(f"{topics.name} has no entry for slide {slide}")
        idx = found[0]
        end = heads[idx + 1].start() if idx + 1 < len(heads) else len(text)
        chunk = text[heads[idx].start():end]
        # Stop before a rule or a part heading that precedes the next slide.
        tail = re.search(r"\n(---\n|# Part |# The thread)", chunk)
        cut = heads[idx].start() + (tail.start() if tail else len(chunk.rstrip("\n")))
        # Keep an example on one page: if what is left of the page cannot hold
        # it, start it on the next. A table split across pages leaves a bare
        # header behind.
        lines = len(body.splitlines()) + 6
        block = (f"\n\n{MARK_OPEN}\n\n\\Needspace{{{lines}\\baselineskip}}\n\n"
                 f"{body}\n\n{MARK_CLOSE}\n")
        text = text[:cut].rstrip("\n") + block + "\n" + text[cut:].lstrip("\n")
    topics.write_text(text, encoding="utf-8")


NOTE_OPEN, NOTE_CLOSE = "<!-- examples-note:begin -->", "<!-- examples-note:end -->"
NUMBERS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
           8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
           13: "thirteen", 14: "fourteen", 15: "fifteen"}


def note(topics: Path, script: str) -> None:
    """Say at the top of the document where the examples are and what they are."""
    text = topics.read_text(encoding="utf-8")
    text = re.sub(rf"\n*{re.escape(NOTE_OPEN)}.*?{re.escape(NOTE_CLOSE)}\n*", "\n\n",
                  text, flags=re.S)
    slides = sorted(EXAMPLES)
    count = NUMBERS.get(len(slides), str(len(slides)))
    where = ("small enough to work through by hand" if topics.parent.name == "Docs"
             else "small enough to do at the board")
    body = (f"> **Worked examples** sit under {count} slides ({', '.join(map(str, slides))}):\n"
            f"> a handful of invented values each, {where}. Their\n"
            f"> numbers are not the lesson's data; every one is computed by `{script}`,\n"
            f"> beside this file, which also writes them here.")
    # After the opening blockquote, the first run of lines starting with ">".
    match = re.search(r"(?m)^>.*(?:\n>.*)*", text)
    if not match:
        raise SystemExit(f"{topics.name} has no opening blockquote to follow")
    end = match.end()
    text = (text[:end] + f"\n\n{NOTE_OPEN}\n\n{body}\n\n{NOTE_CLOSE}\n"
            + "\n" + text[end:].lstrip("\n"))
    topics.write_text(text, encoding="utf-8")


def build(topics: Path) -> bool:
    result = subprocess.run(
        ["pandoc", str(topics), "-o", str(topics.with_suffix(".pdf")),
         "--pdf-engine=xelatex", "--toc", "-V", "geometry:margin=2.5cm",
         "-V", "fontsize=11pt", "-V", "colorlinks=true",
         "-V", "header-includes=\\usepackage{needspace}",
         f"--resource-path={topics.parent}:{topics.parent.parent / 'Figures'}"],
        capture_output=True, text=True)
    missing = [l for l in result.stderr.splitlines() if "Missing character" in l]
    print(f"{topics.with_suffix('.pdf').name}: rc={result.returncode}, "
          f"missing glyphs={len(missing)}")
    for line in missing[:5]:
        print("  ", line)
    if result.returncode:
        print(result.stderr[-1500:])
    return result.returncode == 0 and not missing


def target_of(script: Path) -> Path:
    """The document a lesson's examples belong in.

    ``Review/lN_examples.py`` writes into the instructor's ``lN-commentary.md``
    if there is one, else ``lN-topics.md``. ``Docs/commentary_examples.py`` is
    the same machinery promoted into the repository, beside the students'
    ``{topic}_commentary.md``.
    """
    if script.stem == "commentary_examples":
        found = sorted(script.parent.glob("*_commentary.md"))
        if len(found) != 1:
            raise SystemExit(f"{script.parent}: expected one *_commentary.md, found {len(found)}")
        return found[0]
    number = re.match(r"l(\d+)_examples", script.stem).group(1)
    # A lesson whose study session has been written up keeps one document, the
    # commentary, which absorbs the topics; the examples follow it there.
    commentary = script.with_name(f"l{number}-commentary.md")
    return commentary if commentary.exists() else script.with_name(f"l{number}-topics.md")


def run(script: str) -> None:
    """Print the examples; with --write, splice them in (and rebuild a Review
    PDF); with --check, exit non-zero if the document's examples are stale."""
    script = Path(script).resolve()
    target = target_of(script)
    if "--check" in sys.argv:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            # Same folder name as the original: note() words itself by it.
            copy = Path(tmp) / target.parent.name / target.name
            copy.parent.mkdir()
            copy.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
            splice(copy)
            note(copy, script.name)
            if copy.read_text(encoding="utf-8") != target.read_text(encoding="utf-8"):
                raise SystemExit(f"{target.name}: worked examples differ from what "
                                 f"{script.name} computes - run it with --write")
        print(f"{target.name}: {len(EXAMPLES)} worked examples match {script.name}")
        return
    for slide, body in sorted(EXAMPLES.items()):
        print(f"\n===== slide {slide}\n{body}")
    if "--write" in sys.argv:
        splice(target)
        note(target, script.name)
        print(f"\n{len(EXAMPLES)} examples spliced into {target.name}")
        if target.parent.name == "Review":
            build(target)
