"""Repair a pandoc-generated .pptx so it renders identically everywhere.

    python tools/postprocess_pptx.py deck.pptx [deck2.pptx ...]

Called automatically by tools/build.py after each conversion. Two fixes, in
order, because the second cannot run until the first has made the file valid.

1. **Undeclared namespace prefixes.**
   When pandoc puts an equation inside a table cell it emits `<a14:m>` without
   declaring the `a14` prefix on the slide root, producing XML that is simply
   not well formed. PowerPoint silently tolerates it; stricter readers - lxml,
   and therefore python-pptx - refuse the file outright, and some viewers drop
   the slide. Equations outside tables are emitted correctly, which is why the
   problem only shows up on some slides.

2. **Inherited geometry.**
   pandoc emits shapes with no <a:off>/<a:ext>, leaving position and size to be
   inherited from layout and master. That is valid OOXML and PowerPoint honours
   it, but several viewers fall back to (0, 0) and stack the title on top of the
   body. Course material gets opened in PowerPoint, Google Slides, LibreOffice,
   the OneDrive web preview and on phones, so we resolve the inheritance here
   and write the result into each slide.
"""

import re
import shutil
import sys
import zipfile
from pathlib import Path

from pptx import Presentation
from pptx.oxml.ns import qn

# Prefixes pandoc may use without declaring them, and their correct URIs.
NAMESPACES = {
    "a14": "http://schemas.microsoft.com/office/drawing/2010/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "p14": "http://schemas.microsoft.com/office/powerpoint/2010/main",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
}

SLIDE_XML = re.compile(r"ppt/slides/slide\d+\.xml$")


def _root_span(xml: str) -> tuple[int, int] | None:
    """Character span of the root element's opening tag.

    Skips the XML declaration and any processing instruction or comment, so we
    do not append attributes inside `<?xml ... ?>`.
    """
    for match in re.finditer(r"<(?![?!])[^>]*>", xml):
        return match.start(), match.end() - 1
    return None


def _declare_missing(xml: str) -> tuple[str, list[str]]:
    """Add xmlns declarations for any prefix used but not declared."""
    span = _root_span(xml)
    if span is None:
        return xml, []
    start, end = span
    root = xml[start:end]

    added = []
    for prefix, uri in NAMESPACES.items():
        if not re.search(rf"<{prefix}:|\s{prefix}:", xml):
            continue
        if f"xmlns:{prefix}=" in root:
            continue
        root += f' xmlns:{prefix}="{uri}"'
        added.append(prefix)

    if not added:
        return xml, []
    return xml[:start] + root + xml[end:], added


def repair_namespaces(path: Path) -> dict[str, list[str]]:
    """Rewrite the package, declaring namespaces pandoc left dangling."""
    repaired: dict[str, list[str]] = {}
    temp = path.with_suffix(".repair.pptx")

    with zipfile.ZipFile(path) as src, zipfile.ZipFile(
        temp, "w", zipfile.ZIP_DEFLATED
    ) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if SLIDE_XML.match(item.filename):
                fixed, added = _declare_missing(data.decode("utf8"))
                if added:
                    repaired[item.filename.split("/")[-1]] = added
                    data = fixed.encode("utf8")
            dst.writestr(item, data)

    if repaired:
        temp.replace(path)
    else:
        temp.unlink()
    return repaired


def _layout_counterpart(shape, layout):
    """The layout placeholder a slide placeholder inherits geometry from."""
    if not shape.is_placeholder:
        return None
    idx = shape.placeholder_format.idx
    for candidate in layout.placeholders:
        if candidate.placeholder_format.idx == idx:
            return candidate
    return None


def pin_geometry(path: Path) -> int:
    """Give every shape explicit position and size. Returns shapes changed."""
    prs = Presentation(str(path))
    pinned = 0

    for slide in prs.slides:
        layout = slide.slide_layout
        for shape in slide.shapes:
            spPr = shape._element.find(qn("p:spPr"))
            if spPr is None:
                continue
            xfrm = spPr.find(qn("a:xfrm"))
            if xfrm is not None and xfrm.find(qn("a:off")) is not None:
                continue  # already explicit

            counterpart = _layout_counterpart(shape, layout)
            resolved = {}
            for attr in ("left", "top", "width", "height"):
                value = getattr(shape, attr)
                if value is None and counterpart is not None:
                    value = getattr(counterpart, attr)
                resolved[attr] = value

            if any(v is None for v in resolved.values()):
                # Nothing to inherit from; leave it rather than invent a place.
                continue

            for attr, value in resolved.items():
                setattr(shape, attr, value)
            pinned += 1

    if pinned:
        prs.save(str(path))
    return pinned


def process(path: Path, verbose: bool = True) -> bool:
    backup = path.with_suffix(".orig.pptx")
    shutil.copy2(path, backup)
    try:
        repaired = repair_namespaces(path)
        pinned = pin_geometry(path)
    except Exception as exc:  # keep the original rather than a broken file
        shutil.copy2(backup, path)
        print(f"  {path.name}: postprocess FAILED ({exc}) - left as generated")
        return False
    finally:
        backup.unlink(missing_ok=True)

    if verbose:
        detail = []
        if repaired:
            slides = ", ".join(sorted(repaired))
            detail.append(f"namespaces declared on {slides}")
        if pinned:
            detail.append(f"{pinned} shape(s) pinned")
        print(f"  {path.name}: {'; '.join(detail) if detail else 'nothing to fix'}")
    return True


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    ok = True
    for name in argv:
        path = Path(name)
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 1
        ok &= process(path)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
