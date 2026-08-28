"""The mechanical half of Course/LESSON_REVIEW.md.

    python tools/review_lesson.py 02
    python tools/review_lesson.py            # every lesson

`verify_lesson.py` decides whether a lesson is built. This decides whether some
things about it are consistent, and it is deliberately a separate tool: everything
here is advisory, several checks are heuristic, and a warning may be the right
answer rather than a fault to fix. It never exits non-zero.

The judgement half of a review - does the caption describe the image, does the
slide title tell the truth - is in the document, not here.
"""

import re
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Notes that point at something the audience is supposed to be looking at.
DEICTIC = (
    "point at", "this figure", "the figure", "this picture", "the picture",
    "left panel", "right panel", "on the left", "on the right", "the dashed",
    "the shaded", "look at the", "the diagram", "these cells", "trace the",
    "top row", "bottom row", "the curve", "the two panels", "each panel",
)

# Phrasings that deliberately describe a figure before showing it. Not faults.
FORWARD = (
    "before the figure", "before showing", "before you advance", "before the picture",
    "next slide", "before any picture", "before the next", "before we show",
)


def slides_of(path):
    """Split a deck into slides, each with its title, image and speaker notes."""
    out, cur = [], None
    for line in path.read_text(encoding="utf8").split("\n"):
        if line.startswith("# "):
            if cur:
                out.append(cur)
            cur = {"title": line[2:].strip(), "img": None, "body": []}
        elif cur is not None:
            m = re.match(r"^!\[.*?\]\(([^)]+)\)", line)
            if m and not m.group(1).startswith("eq_"):
                cur["img"] = m.group(1)
            cur["body"].append(line)
    if cur:
        out.append(cur)
    for s in out:
        joined = "\n".join(s["body"])
        n = re.search(r"::: notes(.*?):::", joined, re.S)
        s["notes"] = n.group(1).strip() if n else ""
        s["prose"] = re.sub(r"::: notes.*?:::", "", joined, flags=re.S).strip()
    return out


def notebook_outputs(lesson):
    """Every line any notebook in this lesson printed, as one blob."""
    blob = []
    for nb_path in sorted((lesson / "Notebooks").glob("*.ipynb")):
        if "checkpoint" in str(nb_path):
            continue
        nb = json.loads(nb_path.read_text(encoding="utf8"))
        for cell in nb.get("cells", []):
            for out in cell.get("outputs", []):
                if out.get("output_type") == "stream":
                    blob.append("".join(out.get("text", [])))
                elif out.get("output_type") == "execute_result":
                    blob.append("".join(out.get("data", {}).get("text/plain", [])))
    return "\n".join(blob)


def figures_under_headings(lesson, say):
    """1.1 - a figure should follow the paragraph that introduces it."""
    for doc in sorted((lesson / "Docs").glob("*.md")):
        lines = doc.read_text(encoding="utf8").split("\n")
        for i, line in enumerate(lines):
            if not re.match(r"^!\[\]\(", line):
                continue
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            if j >= 0 and re.match(r"^#{2,3} ", lines[j].strip()):
                name = re.match(r"^!\[\]\(([^)]+)\)", line).group(1)
                say(f"figure under a bare heading: {name} "
                    f"({doc.name}:{i + 1}, under {lines[j].strip()[:44]!r})")


def notes_without_their_figure(lesson, say):
    """2.2 - a note describing the next slide's figure."""
    for deck in sorted((lesson / "Slides").glob("*_slides.md")):
        slides = slides_of(deck)
        for i, s in enumerate(slides):
            nxt = slides[i + 1] if i + 1 < len(slides) else None
            low = s["notes"].lower()
            if s["img"] or not nxt or not nxt["img"]:
                continue
            if any(f in low for f in FORWARD):
                continue                      # deliberate build-up, not a fault
            if "|" in s["prose"] or "$$" in s["prose"]:
                continue                      # a table or equation to point at
            hit = [d for d in DEICTIC if d in low]
            if hit:
                say(f"slide {i + 2} {s['title']!r}: notes say {hit[0]!r} but the "
                    f"figure is on the next slide ({nxt['img']})")


def prose_numbers_not_in_notebooks(lesson, say):
    """1.2 - the check that would have caught lesson 2's drift.

    Heuristic, and it is meant to be: it looks only at numbers precise enough to
    have come from a run - three or more decimals, or a thousands separator.
    """
    printed = notebook_outputs(lesson)
    if not printed:
        return
    # A number the lesson works out by hand is checked by worked_examples.py, which
    # is the arithmetic gate. Only numbers that claim to come from a run belong here.
    worked = lesson / "Docs" / "worked_examples.py"
    printed += "\n" + (worked.read_text(encoding="utf8") if worked.exists() else "")
    pattern = re.compile(r"\b\d+\.\d{3,}\b|\b\d{1,3}(?:,\d{3})+\b")
    for src in sorted((lesson / "Docs").glob("*.md")) + \
               sorted((lesson / "Slides").glob("*_slides.md")):
        seen = set()
        for raw in pattern.findall(src.read_text(encoding="utf8")):
            if raw in seen:
                continue
            seen.add(raw)
            if raw in printed or raw.replace(",", "") in printed:
                continue
            say(f"{src.name}: {raw} is quoted but no notebook printed it")


def em_dashes_where_they_show(lesson, say):
    """1.4 - projected text, including text drawn inside figures."""
    for deck in sorted((lesson / "Slides").glob("*_slides.md")):
        for s in slides_of(deck):
            if "—" in s["title"] or "—" in s["prose"]:
                say(f"em dash on the slide {s['title'][:44]!r}")
    figures = lesson / "Figures" / "make_figures.py"
    sources = [figures] if figures.exists() else []
    for src in sources:
        n = src.read_text(encoding="utf8").count("—")
        if n:
            say(f"{n} em dash(es) drawn inside figures ({src.name})")
    for nb_path in sorted((lesson / "Notebooks").glob("*.ipynb")):
        if "checkpoint" in str(nb_path):
            continue
        nb = json.loads(nb_path.read_text(encoding="utf8"))
        n = 0
        for cell in nb.get("cells", []):
            if cell["cell_type"] != "code":
                continue
            for line in cell["source"]:
                if "—" in line and re.search(
                        r"set_title|set_xlabel|set_ylabel|suptitle|annotate|label=|\.text\(",
                        line):
                    n += 1
        if n:
            say(f"{n} em dash(es) drawn inside figures ({nb_path.name})")


def review(lesson):
    problems = []
    figures_under_headings(lesson, problems.append)
    notes_without_their_figure(lesson, problems.append)
    prose_numbers_not_in_notebooks(lesson, problems.append)
    em_dashes_where_they_show(lesson, problems.append)

    print(f"\n{lesson.name}  —  {len(problems) or 'nothing'} to look at")
    for p in problems:
        print(f"    .  {p}")
    return len(problems)


def main():
    wanted = sys.argv[1] if len(sys.argv) > 1 else None
    lessons = sorted(d for d in (ROOT / "Lessons").iterdir() if d.is_dir())
    if wanted:
        lessons = [d for d in lessons if d.name.startswith(wanted)]
        if not lessons:
            sys.exit(f"no lesson matching {wanted!r}")
    total = sum(review(d) for d in lessons)
    print(f"\n{total} thing(s) worth a look across {len(lessons)} lesson(s).")
    print("Advisory only. The judgement half is in Course/LESSON_REVIEW.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
