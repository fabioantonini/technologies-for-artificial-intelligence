"""Regenerate every distributable artefact from its markdown source.

    python tools/build.py            # everything
    python tools/build.py 03         # only lesson 03
    python tools/build.py --no-pdf   # skip the PDF step
    python tools/build.py --course   # only the course-level documents

Per lesson:

    Slides/*.md     ->  .pptx, then .pdf converted from it by LibreOffice
    Docs/*.md       ->  .pdf   the handout
    Exercises/*.md  ->  .pdf   assessed work, so students want it printable
    Resources/*.md  ->  .pdf   supplementary reading

Course-level, built once rather than per lesson:

    Course/*.md     ->  .pdf   syllabus, schedule, prerequisites
    Course/Setup/*.md, Assessment/**/*.md

Everything except the slides needs a working LaTeX engine.

Every generated .pptx is passed through tools/postprocess_pptx.py, which
repairs two defects in pandoc's output that break rendering in viewers other
than PowerPoint. Never ship a deck that skipped that step.

The generated files are committed: students receive them with a plain
`git pull` and never need pandoc installed.

PDF generation needs a working LaTeX engine. If none is usable the script
reports it and still produces the .pptx files, which are what the lecture
actually needs.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import postprocess_pptx  # noqa: E402
import render_math  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "Course" / "template.pptx"
LESSONS = ROOT / "Lessons"

PDF_ENGINES = ("xelatex", "lualatex", "pdflatex", "tectonic")

# Per-lesson folders whose markdown becomes a PDF. Slides are handled
# separately because they go through the pptx chain first.
PROSE_FOLDERS = ("Docs", "Exercises", "Resources")

COURSE_DOCS = (
    "Course/*.md",
    "Course/Setup/*.md",
    "Assessment/*.md",
    "Assessment/*/*.md",
)

# Author-facing files that are not course material and should not be published
# as PDFs. The root README is a landing page of relative links, meaningless
# once printed; the syntax example exists to be read next to the template.
NOT_PUBLISHED = {"slides_syntax_example.md"}


# TeX writes the current time into every PDF as /CreationDate and /ModDate, so
# rebuilding an unchanged handout produces a different file. SOURCE_DATE_EPOCH
# is the cross-toolchain convention for pinning it; FORCE_SOURCE_DATE makes
# older pdftex honour it. Together with the timestamp normalisation in
# postprocess_pptx.py, this makes the whole build reproducible: the same source
# yields byte-identical output, so only real changes show up in git status.
BUILD_ENV = {
    **os.environ,
    "SOURCE_DATE_EPOCH": "0",
    "FORCE_SOURCE_DATE": "1",
}


def find_pdf_engine() -> str | None:
    """Return the first LaTeX engine that actually compiles a document.

    A binary on PATH is not proof it works - a broken MiKTeX install still
    answers `--version` but refuses real jobs - so we compile a probe.
    """
    probe = "\\documentclass{article}\\begin{document}$x^2$\\end{document}"
    for engine in PDF_ENGINES:
        if not shutil.which(engine):
            continue
        try:
            result = subprocess.run(
                ["pandoc", "-f", "latex", "-t", "pdf", f"--pdf-engine={engine}"],
                input=probe.encode(),
                capture_output=True,
                timeout=120,
            )
        except (subprocess.TimeoutExpired, OSError):
            continue
        if result.returncode == 0 and result.stdout[:4] == b"%PDF":
            return engine
    return None


#: Two 32-character hex strings derived from the clock and a random seed.
#: xdvipdfmx writes them tightly as `/ID[<..><..>]`; LibreOffice spaces them out
#: and puts a newline between the halves, so the pattern has to tolerate both.
#: Every substitution below preserves length, because the cross-reference table
#: holds byte offsets that would otherwise all shift.
PDF_ID = re.compile(
    rb"(/ID\s*\[\s*<)([0-9A-Fa-f]{32})(>\s*<)([0-9A-Fa-f]{32})(>\s*\])")

#: LibreOffice ignores SOURCE_DATE_EPOCH and stamps the wall clock, which is
#: enough on its own to make every rebuilt deck look modified.
PDF_DATE = re.compile(rb"(/(?:Creation|Mod)Date\s*\(D:)(\d{14})((?:[+-]\d\d'\d\d')?)")
FIXED_DATE = b"19800101000000"
FIXED_OFFSET = b"+00'00'"

#: LibreOffice's own trailer checksum. It is computed over the document before
#: the two substitutions above run, so it carries their randomness even once
#: they are pinned, and has to be flattened too.
PDF_CHECKSUM = re.compile(rb"(/DocChecksum\s*/)([0-9A-Fa-f]{32})")


def normalise_pdf(path: Path) -> None:
    """Pin the identifier and the dates so rebuilds are byte-identical.

    Without this every build rewrites every PDF, and since the generated PDFs
    are committed — this repository is the distribution channel — a rebuild
    that changed nothing would still produce a diff touching every deck.
    """
    if not path.exists():
        return
    data = path.read_bytes()

    patched = PDF_ID.sub(
        lambda m: m.group(1) + b"0" * 32 + m.group(3) + b"0" * 32 + m.group(5),
        data)
    patched = PDF_DATE.sub(
        lambda m: m.group(1) + FIXED_DATE + (FIXED_OFFSET if m.group(3) else b""),
        patched)
    patched = PDF_CHECKSUM.sub(lambda m: m.group(1) + b"0" * 32, patched)

    if patched != data:
        assert len(patched) == len(data), "la sostituzione ha cambiato lunghezza"
        path.write_bytes(patched)


def resource_path(source: Path) -> str:
    """Where pandoc looks for images: alongside the source, and in Figures/.

    The separator is os.pathsep - ';' on Windows, ':' elsewhere.
    """
    return os.pathsep.join(
        [str(source.parent), str(render_math.figures_dir(source))]
    )


def run(cmd: list[str]) -> bool:
    result = subprocess.run(cmd, capture_output=True, text=True, env=BUILD_ENV)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()
        print(f"    FAILED: {' '.join(tail[-2:]) if tail else 'unknown error'}")
        return False
    return True


def pptx_to_pdf(pptx: Path) -> bool:
    """Convert a built deck to PDF with LibreOffice.

    The PDF used to be produced by pandoc with -t beamer, which never sees
    Course/template.pptx: no university crest, LaTeX's default colours and
    fonts, a document that looks nothing like the deck it claims to represent.

    Converting the .pptx instead makes the PDF a faithful rendering of what
    will actually be projected, at the cost of needing LibreOffice. Where it is
    missing we skip the PDF rather than emit a misleading one.
    """
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice is None:
        return False
    result = subprocess.run(
        [soffice, "--headless", "--convert-to", "pdf", "--outdir",
         str(pptx.parent), str(pptx)],
        capture_output=True,
        env=BUILD_ENV,
    )
    return result.returncode == 0 and pptx.with_suffix(".pdf").exists()


def build_slides(source: Path, engine: str | None) -> tuple[int, int]:
    made = failed = 0
    pptx = source.with_suffix(".pptx")
    print(f"  {source.relative_to(ROOT)} -> {pptx.name}")

    # Take the LaTeX out before pandoc sees it: its OMML output is not
    # renderable. See tools/render_math.py. The author's source is untouched.
    staged = source.with_suffix(".staged.md")
    uni, img, warnings = render_math.process(source, staged)
    if uni or img:
        print(f"    math: {uni} to Unicode, {img} to image")
    for warning in warnings:
        print(f"    note: {warning}")

    try:
        ok = run(
            [
                "pandoc",
                str(staged),
                "-o",
                str(pptx),
                f"--reference-doc={TEMPLATE}",
                f"--resource-path={resource_path(source)}",
            ]
        )
    finally:
        staged.unlink(missing_ok=True)

    clean = True
    if ok:
        # pandoc's pptx output still needs repairing before it is safe to ship;
        # see tools/postprocess_pptx.py for what and why. Its result is part of
        # whether the deck built: when the repair does not run, the file on disk
        # is raw pandoc output with the title stacked on the body, and reporting
        # that as a success is how a broken deck reaches a lecture room.
        clean = postprocess_pptx.process(pptx)
    made, failed = (made + 1, failed) if ok and clean else (made, failed + 1)

    if ok:
        pdf = pptx.with_suffix(".pdf")
        print(f"  {pptx.relative_to(ROOT)} -> {pdf.name}")
        if pptx_to_pdf(pptx):
            normalise_pdf(pdf)
            made += 1
        else:
            print("    note: LibreOffice not found, slide PDF skipped")
    return made, failed


def pdf_name(source: Path) -> Path:
    """Where the PDF for this source goes.

    A file called README.md would produce README.pdf, which tells a student
    nothing once it is sitting in their downloads folder. Name it after the
    folder instead: Assessment/Project/README.md -> Assessment/Project/Project.pdf
    """
    if source.stem.lower() == "readme":
        return source.with_name(f"{source.parent.name}.pdf")
    return source.with_suffix(".pdf")


def build_handout(source: Path, engine: str) -> tuple[int, int]:
    pdf = pdf_name(source)
    print(f"  {source.relative_to(ROOT)} -> {pdf.name}")
    ok = run(
        [
            "pandoc",
            str(source),
            "-o",
            str(pdf),
            f"--pdf-engine={engine}",
            "--toc",
            "-V",
            "geometry:margin=2.5cm",
            "-V",
            "fontsize=11pt",
            "-V",
            "colorlinks=true",
            f"--resource-path={resource_path(source)}",
        ]
    )
    if ok:
        normalise_pdf(pdf)
    return (1, 0) if ok else (0, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lesson", nargs="?", help="lesson number, e.g. 03")
    parser.add_argument("--no-pdf", action="store_true", help="skip PDF output")
    parser.add_argument(
        "--course", action="store_true", help="only the course-level documents"
    )
    args = parser.parse_args()

    if not shutil.which("pandoc"):
        print("pandoc not found on PATH - install it first", file=sys.stderr)
        return 1
    if not TEMPLATE.exists():
        print(f"missing {TEMPLATE} - run tools/make_template.py", file=sys.stderr)
        return 1

    pattern = f"{args.lesson}_*" if args.lesson else "*"
    lessons = sorted(d for d in LESSONS.glob(pattern) if d.is_dir())
    if not lessons and not args.course:
        print(f"no lesson matches {pattern!r}", file=sys.stderr)
        return 1

    engine = None if args.no_pdf else find_pdf_engine()
    if args.no_pdf:
        print("PDF generation skipped (--no-pdf)\n")
    elif engine:
        print(f"PDF engine: {engine}\n")
    else:
        print(
            "No working LaTeX engine found - building .pptx only.\n"
            "  Every PDF is skipped: handouts, exercises, resources and course\n"
            "  documents. The markdown still renders with full math on GitHub,\n"
            "  so this is not blocking.\n"
        )

    made = failed = 0

    if not args.course:
        for lesson in lessons:
            sources = sorted(lesson.glob("Slides/*.md"))
            if engine:
                for folder in PROSE_FOLDERS:
                    sources += sorted(lesson.glob(f"{folder}/*.md"))
            if not sources:
                continue
            print(f"{lesson.name}")
            for source in sources:
                m, f = (
                    build_slides(source, engine)
                    if source.parent.name == "Slides"
                    else build_handout(source, engine)
                )
                made, failed = made + m, failed + f

    # Course-level documents are not per-lesson, so build them once - unless a
    # single lesson was requested, when rebuilding them would be surprising.
    if engine and (args.course or not args.lesson):
        course_sources = sorted(
            path
            for pattern in COURSE_DOCS
            for path in ROOT.glob(pattern)
            if path.name not in NOT_PUBLISHED
        )
        if course_sources:
            print("course documents")
            for source in course_sources:
                m, f = build_handout(source, engine)
                made, failed = made + m, failed + f

    print(f"\n{made} file(s) written, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
