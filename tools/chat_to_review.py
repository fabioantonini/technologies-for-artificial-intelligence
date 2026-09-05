"""Turn a shared ChatGPT study session into a lesson's Review/ document.

    python tools/chat_to_review.py 03 https://chatgpt.com/share/<id>
    python tools/chat_to_review.py 03 --html saved.html      # offline
    python tools/chat_to_review.py 03 <url> --no-pdf

The workflow this serves: study a lesson against its own slides in ChatGPT,
share the session, and get back a typeset document to consult while teaching.
Output lands in ``Lessons/NN_*/Review/``, which .gitignore excludes, so the
material stays out of the repository while this script stays in it.

Why a script rather than a paste. The share page renders client-side, so
fetching it yields a JavaScript shell; the conversation is really there, but as
a React Router turbo-stream: a flat pool of ~14,000 values in which objects
address their keys and values by index. Reassembling it is most of what follows
and is not something to rediscover once per lesson.

Three things worth knowing about the result.

**Most of what you pasted is already in the repository.** A study session is
mostly slides and speaker notes fed back in, question by question. Reprinting
them would duplicate the deck, so a pasted message is matched against the
lesson's own slides and collapses to a heading plus that slide's figure, taken
from ``Figures/`` at full resolution rather than from the screenshot you
uploaded. Only messages that are *not* in the deck are treated as questions and
quoted in full.

**Attached images are not recoverable and mostly do not matter.** They sit
behind an authenticated endpoint. In practice they are screenshots of the very
slides just matched, so the originals are better. An image the assistant
generated is a real loss; the script says so at the end, and the honest repair
is to recompute the figure rather than to reproduce it.

**Chat formatting is not document formatting.** A transcript puts almost every
formula on a display line of its own, which costs roughly a third of the page
count. One-line equations are folded back into the prose; nothing longer is
touched, and no character of any equation is dropped.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADER = Path(__file__).resolve().parent / "review_header.tex"

USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

#: Longest single-line equation folded back into the prose. Chosen so that a
#: folded formula still fits comfortably inside a line of body text.
INLINE_LIMIT = 55


# --------------------------------------------------------------- the transcript

def fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", "replace")


def decode_stream(html: str) -> list:
    """Recover the turbo-stream pool the share page hands to React Router.

    The payload arrives as one or more `enqueue("...")` calls holding an escaped
    JSON array, followed by further lines the client resolves lazily. Only the
    first line is the pool, and it is the whole conversation.
    """
    chunks = re.findall(r'streamController\.enqueue\("((?:[^"\\]|\\.)*)"\)', html)
    if not chunks:
        raise SystemExit(
            "no conversation payload in the page.\n"
            "    The link may be private, expired, or served behind a challenge.\n"
            "    Open it in a browser, save the page, and pass --html.")
    raw = "".join(json.loads('"' + chunk + '"') for chunk in chunks)
    return json.loads(raw.split("\n")[0])


def resolver(pool: list):
    """Resolve pool[i], following the index references objects and lists use.

    Keys are stored as "_<index>" and values as bare indices, so a node is only
    meaningful relative to the pool. The visited set guards the self-references
    the format permits.
    """

    def resolve(index, seen=frozenset()):
        if not isinstance(index, int) or not 0 <= index < len(pool):
            return None
        if index in seen:
            return None
        value = pool[index]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        deeper = seen | {index}
        if isinstance(value, dict):
            return {(resolve(int(k[1:]), deeper) if k.startswith("_") else k):
                    resolve(v, deeper) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(item, deeper) for item in value]
        return value

    return resolve


def conversation(pool: list) -> list:
    """The node list, found by locating the key that names it."""
    resolve = resolver(pool)
    try:
        key = pool.index("linear_conversation")
    except ValueError:
        raise SystemExit("the payload carries no conversation")
    for value in pool:
        if isinstance(value, dict) and f"_{key}" in value:
            nodes = resolve(value[f"_{key}"])
            if isinstance(nodes, list):
                return nodes
    raise SystemExit("the conversation key resolves to nothing")


def messages(nodes: list) -> tuple[list[dict], int]:
    """Visible user and assistant turns, in order, plus a generated-image count.

    Everything else a session accumulates - tool calls, reasoning recaps, the
    system preamble, hidden context - is scaffolding rather than content.
    """
    kept, generated = [], 0
    for node in nodes:
        message = (node or {}).get("message") if isinstance(node, dict) else None
        if not isinstance(message, dict):
            continue
        role = ((message.get("author") or {}).get("role"))
        content = message.get("content") or {}
        parts = content.get("parts") or []
        text = "\n".join(p for p in parts if isinstance(p, str))
        images = [p for p in parts if isinstance(p, dict)]
        if role not in ("user", "assistant"):
            generated += len(images)
            continue
        if content.get("content_type") not in ("text", "multimodal_text"):
            continue
        if (message.get("metadata") or {}).get("is_visually_hidden_from_conversation"):
            continue
        if not text.strip() and not images:
            continue
        kept.append({"role": role, "text": text, "images": len(images)})
    return kept, generated


# ---------------------------------------------------------------- the lesson

def lesson_dir(number: str) -> Path:
    matches = sorted(ROOT.glob(f"Lessons/{int(number):02d}_*"))
    if not matches:
        raise SystemExit(f"no lesson {number} under Lessons/")
    return matches[0]


def deck_path(lesson: Path) -> Path:
    decks = sorted(lesson.glob("Slides/*_slides.md"))
    if not decks:
        raise SystemExit(f"{lesson.name} has no deck to match against")
    return decks[0]


def flatten(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).lower()
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]+', ' ', text)).strip()


def parse_deck(path: Path) -> tuple[list[dict], str]:
    """Every slide with its title, notes and figures, plus the deck's own title."""
    source = path.read_text(encoding="utf-8")
    title = re.search(r'^title:\s*"(.*?)"', source, re.M)
    slides = []
    for chunk in re.split(r'\n(?=# )', source.split("---", 2)[-1]):
        if not chunk.startswith("# "):
            continue
        notes = re.search(r'::: notes\n(.*?)\n:::', chunk, re.S)
        slides.append({
            "n": len(slides) + 1,
            "title": chunk.split("\n", 1)[0][2:].strip(),
            "figures": re.findall(r'!\[\]\(([^)]+)\)', chunk),
            "probe": flatten((notes.group(1) if notes else "") + " "
                             + chunk.split("\n", 1)[0]),
        })
    return slides, (title.group(1) if title else path.stem)


def match_slide(text: str, slides: list[dict]) -> dict | None:
    """Which slide was pasted, if any.

    Scored on shared fragments rather than equality: what reaches the chat has
    been through a copy and a re-wrap, so it is never byte-identical to the
    source. A third of the probes agreeing is decisive in practice - unrelated
    prose about the same subject shares none.
    """
    flat = flatten(text)
    if len(flat) < 40:
        return None
    probes = [flat[k:k + 60] for k in range(0, max(1, len(flat) - 60), 90)][:10]
    best, score = None, 0
    for slide in slides:
        hits = sum(1 for probe in probes if probe and probe in slide["probe"])
        if hits > score:
            best, score = slide, hits
    return best if score >= max(1, len(probes) // 3) else None


# ------------------------------------------------------------------ formatting

def fix_math(text: str) -> str:
    """Chat maths into pandoc markdown.

    Only a *single* backslash opens maths: "\\\\[1mm]" is a line break inside a
    display equation, and rewriting its bracket splits the equation in half.
    Headings are demoted below the ones this script emits, and capped at level
    five so they stay legible.
    """
    text = re.sub(r'(?<!\\)\\\(', "$", text)
    text = re.sub(r'(?<!\\)\\\)', "$", text)
    text = re.sub(r'(?<!\\)\\\[', "\n$$\n", text)
    text = re.sub(r'(?<!\\)\\\]', "\n$$\n", text)
    text = re.sub(r'^[ \t]*(#{1,6})[ \t]+',
                  lambda m: "\n" + "#" * min(5, len(m.group(1)) + 2) + " ",
                  text, flags=re.M)
    return re.sub(r'\n{3,}', "\n\n", text).replace("​", "")


def inline_short_equations(text: str) -> str:
    """Fold one-line display equations back into the prose.

    A chat window displays "y = X\\\\beta" on three lines of its own; a document
    should not. Anything aligned, boxed, multi-line or long keeps its display,
    and every character of every equation survives either way.
    """
    lines = text.split("\n")
    out, i = [], 0
    while i < len(lines):
        if lines[i].strip() == "$$":
            j = i + 1
            while j < len(lines) and lines[j].strip() != "$$":
                j += 1
            body = "\n".join(lines[i + 1:j]).strip()
            foldable = (j < len(lines) and body and "\n" not in body
                        and len(body) <= INLINE_LIMIT
                        and not any(t in body for t in ("\\begin", "\\\\", "\\boxed")))
            if foldable:
                tail, stop = body.rstrip(), ""
                if tail.endswith((".", ",", ";", ":")):
                    tail, stop = tail[:-1].rstrip(), tail[-1]
                math = f"${tail}${stop}"
                previous = len(out) - 1
                while previous >= 0 and not out[previous].strip():
                    previous -= 1
                if previous >= 0 and out[previous].rstrip().endswith(":"):
                    del out[previous + 1:]
                    out[previous] = out[previous].rstrip() + " " + math
                else:
                    if out and out[-1].strip():
                        out.append("")
                    out.append(math)
                i = j + 1
                continue
            # Not foldable: copy the block whole, delimiters included. Emitting
            # only the opening "$$" would leave the closing one to be read as
            # the next opening, pairing every later delimiter with the wrong
            # partner and swallowing prose into maths.
            if j < len(lines):
                out.extend(lines[i:j + 1])
                i = j + 1
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def load_extras(review: Path) -> list[dict]:
    """Per-lesson insertions, declared in Review/extras.json if there are any.

    Anything the session produced that this script cannot recover - a figure the
    assistant generated, a diagram worth redrawing - belongs to one lesson and
    not to the tool. Each entry names the question it follows by a fragment of
    its text rather than by position, so the file survives a re-fetch that
    shifts the numbering.

        [{"after_question": "figura del diamante",
          "figure": "lasso_no_selection.png",
          "caption": "*What to look at.*"}]
    """
    path = review / "extras.json"
    if not path.exists():
        return []
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"{path.relative_to(ROOT)} is not valid JSON: {error}")
    for entry in entries:
        missing = {"after_question", "figure"} - set(entry)
        if missing:
            raise SystemExit(f"{path.relative_to(ROOT)}: entry missing {missing}")
        if not (review / entry["figure"]).exists():
            raise SystemExit(f"{path.relative_to(ROOT)}: no such figure "
                             f"{entry['figure']} in {review.relative_to(ROOT)}")
    return entries


def compose(turns: list[dict], slides: list[dict], deck_title: str,
            extras: list[dict]) -> tuple[str, dict]:
    out, stats = [], {"pasted": 0, "questions": 0, "answers": 0,
                              "figures": 0, "extras": 0}
    for index, turn in enumerate(turns):
        if index == 0 and turn["role"] == "user" and len(turn["text"]) < 40:
            continue                       # the opening "let us call this ..."
        if turn["role"] == "assistant":
            stats["answers"] += 1
            out.append("\n" + fix_math(turn["text"]).strip() + "\n")
            continue
        slide = match_slide(turn["text"], slides)
        if slide:
            stats["pasted"] += 1
            out.append(f"\n\\newpage\n\n## Slide {slide['n']} — {slide['title']}\n")
            out.append("*Slide e note del deck, sottoposte all'analisi. "
                       "Il testo integrale è nel deck della lezione.*\n")
            for figure in slide["figures"]:
                stats["figures"] += 1
                out.append(f"![]({figure})\n")
        else:
            stats["questions"] += 1
            # Title the section with the question: a table of contents full of
            # identical "Domanda" entries cannot be navigated during a lecture,
            # which is the one thing this document is for.
            flat = " ".join(turn["text"].split())
            label = (flat[:70].rstrip() + "…") if len(flat) > 70 else flat
            out.append(f"\n\\newpage\n\n## Domanda — {label or 'sulla figura'}\n")
            body = turn["text"].strip() or "*(domanda posta su un'immagine allegata)*"
            out.append("> **" + body.replace("\n", "\n> ") + "**\n")
            if turn["images"] and not turn["text"].strip():
                out.append("\n*Riferita alla figura della slide precedente.*\n")
            for extra in extras:
                if extra["after_question"].lower() in turn["text"].lower():
                    stats["extras"] += 1
                    out.append(f"\n![]({extra['figure']})\n")
                    if extra.get("caption"):
                        out.append(f"\n{extra['caption']}\n")
    return inline_short_equations("".join(out)), stats


def preamble(deck_title: str) -> str:
    return f"""---
title: "{deck_title} — Approfondimenti"
subtitle: "Technologies for Artificial Intelligence"
author: "Fabio Antonini — Università degli Studi dell'Aquila"
date: "Materiale di appoggio alla lezione, non esaminabile"
lang: it
---

## Che cos'è questo documento

Non è un artefatto del corso. È la trascrizione ordinata di una sessione di
approfondimento condotta a partire dalle slide della lezione, pensata come
**materiale da consultare mentre si insegna**: esempi numerici da cui attingere,
derivazioni svolte per esteso e chiarimenti sui punti che in aula generano più
domande.

Non è esaminabile e non va distribuito agli studenti. Vive in `Review/`, che è
esclusa dal repository.

**Come è organizzato.** La sequenza è quella originale della sessione. Le sezioni
intitolate *Slide N* riprendono una slide sottoposta all'analisi: se ne riporta
solo la figura, perché testo e note stanno già nel deck. Le sezioni intitolate
*Domanda* sono le domande poste al di fuori del commento slide per slide, e sono
riportate testualmente.

---
"""


# ----------------------------------------------------------------------- build

def build_pdf(markdown: Path, lesson: Path) -> bool:
    pdf = markdown.with_suffix(".pdf")
    command = [
        "pandoc", str(markdown), "-o", str(pdf),
        "--pdf-engine=xelatex", "--toc",
        "-V", "geometry:margin=2.5cm", "-V", "fontsize=11pt",
        "-V", "colorlinks=true",
        f"--resource-path={markdown.parent}:{lesson / 'Figures'}",
    ]
    if HEADER.exists():
        command += ["-H", str(HEADER)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()
        print("    PDF FAILED: " + " ".join(tail[-3:]))
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("lesson", help="lesson number, e.g. 03")
    parser.add_argument("url", nargs="?", help="ChatGPT share link")
    parser.add_argument("--html", help="a saved copy of the share page instead")
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args()

    if not args.url and not args.html:
        parser.error("give a share link, or --html with a saved page")

    lesson = lesson_dir(args.lesson)
    deck = deck_path(lesson)
    slides, deck_title = parse_deck(deck)
    print(f"{lesson.name}: {len(slides)} slides in {deck.name}")

    if args.html:
        html = Path(args.html).read_text(encoding="utf-8", errors="replace")
    else:
        try:
            html = fetch(args.url)
        except urllib.error.URLError as error:
            raise SystemExit(f"could not fetch the share link: {error}")

    review = lesson / "Review"
    review.mkdir(exist_ok=True)

    turns, generated = messages(conversation(decode_stream(html)))
    extras = load_extras(review)
    body, stats = compose(turns, slides, deck_title, extras)

    markdown = review / "approfondimenti.md"
    markdown.write_text(preamble(deck_title) + body, encoding="utf-8")

    print(f"  {stats['pasted']} pasted slides matched, "
          f"{stats['questions']} questions, {stats['answers']} answers")
    print(f"  {stats['figures']} figures taken from Figures/ at full resolution")
    if extras:
        print(f"  {stats['extras']} of {len(extras)} declared extras placed")
        for extra in extras:
            if extra["after_question"].lower() not in "\n".join(
                    t["text"].lower() for t in turns):
                print(f"    UNPLACED: no question matches "
                      f"{extra['after_question']!r}")
    print(f"  {markdown.relative_to(ROOT)}")

    if not args.no_pdf and build_pdf(markdown, lesson):
        print(f"  {markdown.with_suffix('.pdf').relative_to(ROOT)}")

    if generated:
        print(f"\n  NOTE: {generated} image(s) were generated during the session "
              f"and cannot be\n  downloaded from a share link. Recompute them "
              f"rather than reproducing them;\n  Lessons/03_regression/Review/"
              f"make_review_figure.py is the worked precedent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
