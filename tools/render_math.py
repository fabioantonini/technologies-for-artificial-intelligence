"""Replace LaTeX math in a slide source with something PowerPoint can render.

pandoc converts `$...$` into OMML wrapped in `<a14:m>`. That markup is not
renderable in practice: LibreOffice drops the entire slide body, and PowerPoint
reports damaged content and strips it during repair. Verified against pandoc's
own default template, so it is not a template or post-processing problem - the
OMML route is simply unusable for slides.

So we take the math out before pandoc ever sees it:

- **Inline math** becomes Unicode: `$x^2$` -> `x²`, `$\alpha$` -> `α`. Still real
  text, so it stays editable, searchable and correctly styled.
- **Display math** becomes a transparent PNG rendered with matplotlib's
  mathtext, dropped into the lesson's Figures/ folder.

Anything too complex to convert to Unicode inline is promoted to a display
image rather than silently mangled.

This only affects slides. Handouts keep real LaTeX: they are built by a TeX
engine, which renders math properly.

    python tools/render_math.py Lessons/03_regression/Slides/regression_slides.md
"""

import hashlib
import re
import subprocess
import sys
from pathlib import Path

# Every letter Unicode actually has a raised form for - q has none.
SUPERSCRIPT = str.maketrans("0123456789+-=()abcdefghijklmnoprstuvwxyz",
                            "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ")
# Unicode has far fewer lowered letters than raised ones; these are all
# of them. A subscript using any other letter has no honest rendering.
SUBSCRIPT = str.maketrans("0123456789+-=()aehijklmnoprstuvx",
                          "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")

SYMBOLS = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\epsilon": "ε", r"\varepsilon": "ε", r"\zeta": "ζ", r"\eta": "η",
    r"\theta": "θ", r"\lambda": "λ", r"\mu": "μ", r"\nu": "ν", r"\xi": "ξ",
    r"\pi": "π", r"\rho": "ρ", r"\sigma": "σ", r"\tau": "τ", r"\phi": "φ",
    r"\chi": "χ", r"\psi": "ψ", r"\omega": "ω",
    r"\Gamma": "Γ", r"\Delta": "Δ", r"\Theta": "Θ", r"\Lambda": "Λ",
    r"\Sigma": "Σ", r"\Phi": "Φ", r"\Psi": "Ψ", r"\Omega": "Ω",
    r"\times": "×", r"\cdot": "·", r"\pm": "±", r"\mp": "∓",
    r"\leq": "≤", r"\geq": "≥", r"\neq": "≠", r"\approx": "≈",
    r"\equiv": "≡", r"\propto": "∝", r"\sim": "∼",
    r"\in": "∈", r"\notin": "∉", r"\subset": "⊂", r"\subseteq": "⊆",
    r"\cup": "∪", r"\cap": "∩", r"\emptyset": "∅",
    r"\forall": "∀", r"\exists": "∃", r"\nabla": "∇", r"\partial": "∂",
    r"\infty": "∞", r"\sum": "∑", r"\prod": "∏", r"\int": "∫",
    r"\rightarrow": "→", r"\to": "→", r"\leftarrow": "←",
    r"\Rightarrow": "⇒", r"\Leftarrow": "⇐", r"\leftrightarrow": "↔",
    r"\mathbb{R}": "ℝ", r"\mathbb{N}": "ℕ", r"\mathbb{Z}": "ℤ",
    r"\mathbb{Q}": "ℚ", r"\mathbb{C}": "ℂ", r"\mathbb{E}": "𝔼",
    r"\top": "ᵀ", r"\ldots": "…", r"\dots": "…", r"\cdots": "⋯",
    r"\,": " ", r"\;": " ", r"\!": "", r"\ ": " ",
}

# Constructs with no honest inline Unicode rendering.
TOO_COMPLEX = re.compile(r"\\(begin|sum_|prod_|int_|binom|matrix|left|right)")

VULGAR_FRACTIONS = {
    ("1", "2"): "½", ("1", "3"): "⅓", ("2", "3"): "⅔",
    ("1", "4"): "¼", ("3", "4"): "¾", ("1", "5"): "⅕",
    ("2", "5"): "⅖", ("3", "5"): "⅗", ("4", "5"): "⅘",
    ("1", "6"): "⅙", ("5", "6"): "⅚", ("1", "8"): "⅛",
    ("3", "8"): "⅜", ("5", "8"): "⅝", ("7", "8"): "⅞",
}

#: \hat was being dropped, so the shapes table in lesson 9 listed ŷ as "y".
#: Unicode has a precomposed circumflex for some letters and a combining one
#: for the rest, which every font in the template renders.
PRECOMPOSED_HAT = {
    "a": "â", "c": "ĉ", "e": "ê", "g": "ĝ", "h": "ĥ", "i": "î", "j": "ĵ",
    "o": "ô", "s": "ŝ", "u": "û", "w": "ŵ", "y": "ŷ",
    "A": "Â", "C": "Ĉ", "E": "Ê", "G": "Ĝ", "H": "Ĥ", "I": "Î", "J": "Ĵ",
    "O": "Ô", "S": "Ŝ", "U": "Û", "W": "Ŵ", "Y": "Ŷ",
}
COMBINING_CIRCUMFLEX = "\u0302"
HAT = re.compile(r"\\hat\{?(\w)\}?")


def _apply_hats(text: str) -> str:
    return HAT.sub(
        lambda m: PRECOMPOSED_HAT.get(m.group(1), m.group(1) + COMBINING_CIRCUMFLEX),
        text,
    )


FRACTION = re.compile(r"\\[tdc]?frac\{([^{}]*)\}\{([^{}]*)\}")
ROOT = re.compile(r"\\sqrt\{([^{}]*)\}")


def _flatten_fractions(text: str) -> str:
    """Turn simple fractions into readable inline text.

    `\\frac{1}{2}` becomes ½ where a vulgar fraction exists, otherwise a slash
    form with brackets when the denominator is compound, so `\\frac{1}{2m}`
    reads as 1/(2m) rather than the ambiguous 1/2m.
    """

    def repl(match: re.Match) -> str:
        num, den = match.group(1).strip(), match.group(2).strip()
        if (num, den) in VULGAR_FRACTIONS:
            return VULGAR_FRACTIONS[(num, den)]
        den_text = den if len(den) == 1 else f"({den})"
        return f"{num}/{den_text}"

    previous = None
    while previous != text:  # resolve one nesting level at a time
        previous = text
        text = FRACTION.sub(repl, text)
    return text


#: Symbols that stand where a letter would. The space after them in the source
#: exists only to terminate the control word - TeX sets \Delta G closed up - so
#: it has to go, or slides read "Δ G" and "Xᵀ X". Relations and binary operators
#: are deliberately absent: their spacing is real, and α = 0.1 must keep it.
LETTERLIKE = "αβγδεζηθλμνξπρστφχψωΓΔΘΛΣΦΨΩᵀ∞∂∇ℝℕℤℚℂ𝔼"


def _tighten_spacing(text: str) -> str:
    """Close the gap a control word's terminating space left behind."""
    text = re.sub(rf"([{LETTERLIKE}]) (?=[A-Za-z0-9{LETTERLIKE}])", r"\1", text)
    # A sign in unary position binds to what follows: \pm 1 is ±1. After a
    # term it is a binary operator - a ± b - and keeps its spaces, so only an
    # opening bracket, a relation or the start of the formula counts.
    text = re.sub(r"(^|[(\[{=,] ?)([±∓]) (?=[\w(])", r"\1\2", text)
    return text


def to_unicode(latex: str) -> str | None:
    """Render simple inline math as Unicode, or None if it cannot be done."""
    if TOO_COMPLEX.search(latex):
        return None

    text = _apply_hats(latex)
    text = _flatten_fractions(text)
    text = ROOT.sub(lambda m: f"√{m.group(1)}", text)
    for command, symbol in sorted(SYMBOLS.items(), key=lambda kv: -len(kv[0])):
        text = text.replace(command, symbol)

    def script(match: re.Match, table: dict) -> str:
        body = match.group(1) or match.group(2) or ""
        converted = body.translate(table)
        # Refuse a PARTIAL conversion. Letting one through printed F_{t-1} as
        # "Ft₋₁" and r_{ij} as "rᵢj", which read as products, not subscripts.
        # Leaving the marker in place makes to_unicode give up, and the build
        # then says so instead of shipping something misleading.
        # Already-raised characters count as converted: \top has become ᵀ
        # by this point, and X^ᵀ must still close up to Xᵀ.
        marks = set(table.values()) | {ord("ᵀ")}
        if body and all(
            character in table or character in marks
            for character in map(ord, body)
        ):
            return converted
        # Anything less is refused, marker and all: letting a partial through
        # printed F_{t-1} as "Ft₋₁" and r_{ij} as "rᵢj", and letting one with
        # no rendering at all through printed m_L as "mL". Both read as
        # products. Keeping the marker makes to_unicode give up, and the build
        # then names the formula instead of shipping something misleading.
        return match.group(0)

    text = re.sub(r"\^\{([^{}]*)\}|\^(\w)", lambda m: script(m, SUPERSCRIPT), text)
    text = re.sub(r"_\{([^{}]*)\}|_(\w)", lambda m: script(m, SUBSCRIPT), text)

    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = _tighten_spacing(text)
    if "\\" in text or "^" in text or "_" in text:
        return None  # something survived that we cannot render honestly
    return text


# matplotlib's mathtext understands a large subset of LaTeX, but not all of it.
# These have exact equivalents it does know.
MATHTEXT_ALIASES = {
    r"\tfrac": r"\frac",
    r"\dfrac": r"\frac",
    r"\operatorname": r"\mathrm",
    r"\text": r"\mathrm",
    r"\mathbb{E}": r"\mathrm{E}",
    r"\bigl": r"\left",
    r"\bigr": r"\right",
    r"\Bigl": r"\left",
    r"\Bigr": r"\right",
    r"\!": "",
    r"\,": r"\ ",
    r"\;": r"\ ",
}


def _space_out_words(latex: str) -> str:
    r"""Keep the spaces inside \text{...} alive through mathtext.

    mathtext maps \text to \mathrm, which sets its argument as a single run
    and throws the spaces away: ``\text{only if }`` came out as ``onlyif``.
    Setting one \mathrm per word, joined by an explicit ``\ ``, preserves
    them - leading and trailing spaces included, which is how these are
    usually written.
    """
    pattern = re.compile(r"\\(?:text|operatorname)\{([^{}]*)\}")

    def rewrite(match):
        inner = match.group(1)
        if " " not in inner:
            return match.group(0)
        words = [w for w in inner.split(" ") if w]
        spaced = r"\ ".join(rf"\mathrm{{{w}}}" for w in words)
        if inner.startswith(" "):
            spaced = r"\ " + spaced
        if inner.endswith(" "):
            spaced = spaced + r"\ "
        return spaced

    return pattern.sub(rewrite, latex)


def _normalise(latex: str) -> str:
    latex = _space_out_words(latex)
    for command, replacement in sorted(
        MATHTEXT_ALIASES.items(), key=lambda kv: -len(kv[0])
    ):
        latex = latex.replace(command, replacement)
    return latex


#: The course image, and where it bind-mounts this repository.
CONTAINER = "tai_course"
WORKDIR = "/home/jovyan/work"
REPO = Path(__file__).resolve().parent.parent

#: Display maths is placed on the slide at its true size and never enlarged
#: (see `relayout_content_slides`), so this is the point size the audience
#: reads: rendering runs at 220 dpi onto a 10-inch slide, which makes a point
#: in the PNG a point on the slide. At 26 a short formula had to be stretched
#: to fill the width, which set lesson 2's z-score at 151pt and Tukey's fence,
#: on the next slide, at 42.
DISPLAY_PT, INLINE_PT = 56, 18

#: Equations drawn here rather than in the container during this run. Silence
#: is the failure mode that matters: a build with the container down would
#: otherwise produce host-rendered images, and the author would commit them
#: without ever being told the environment had changed under them.
_drawn_on_host: list = []

#: The drawing itself. Kept as source rather than a function because the
#: container executes it with `python -c`.
DRAW = """
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
latex, out, size = sys.argv[1], sys.argv[2], float(sys.argv[3])
fig = plt.figure(figsize=(0.01, 0.01))
fig.text(0, 0, latex, fontsize=size, color="#1A1A1A")
try:
    fig.savefig(out, dpi=220, transparent=True, bbox_inches="tight",
                pad_inches=0.06)
except ValueError:
    raise SystemExit(3)
"""


def _draw_in_container(latex: str, out: Path, size: float):
    """Draw inside the image, or None if there is no container to draw in.

    Equation images are figures, and this repository has one rule for figures:
    regenerate them in a single environment, because matplotlib versions differ
    and every run otherwise shows up as a diff. That rule was written for the
    notebooks and quietly excluded these - the host was drawing them with
    matplotlib 3.11 against the image's pinned 3.8, and the same z-score came
    out 701x239 here against 683x264 there.

    True if it drew, False if mathtext refused, None if there is no container
    and the caller should fall back to drawing here.
    """
    alive = subprocess.run(["docker", "exec", CONTAINER, "true"],
                           capture_output=True, text=True).returncode == 0
    if not alive:
        return None
    try:
        inside = out.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return None  # outside the bind mount; only the host can reach it
    drawn = subprocess.run(
        ["docker", "exec", "-w", WORKDIR, CONTAINER, "python", "-c", DRAW,
         latex, inside, str(size)],
        capture_output=True, text=True)
    return drawn.returncode == 0


def render_image(latex: str, figures: Path, display: bool = True) -> Path | None:
    """Render one equation to a transparent PNG, or None if mathtext refuses.

    A formula we cannot draw must never break the build: the caller falls back
    to leaving readable text on the slide and warns, so the author can decide
    whether to simplify it or move it to the handout where it belongs.
    """
    figures.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5(latex.encode()).hexdigest()[:10]
    out = figures / f"eq_{digest}.png"
    if out.exists():
        return out

    size = DISPLAY_PT if display else INLINE_PT
    drawn = _draw_in_container(f"${_normalise(latex)}$", out, size)
    if drawn is True:
        return out
    if drawn is False:
        out.unlink(missing_ok=True)
        return None

    _drawn_on_host.append(out.name)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${_normalise(latex)}$", fontsize=size, color="#1A1A1A")
    try:
        fig.savefig(
            out, dpi=220, transparent=True, bbox_inches="tight", pad_inches=0.06
        )
    except ValueError:
        out.unlink(missing_ok=True)
        return None
    finally:
        plt.close(fig)
    return out


def figures_dir(source: Path) -> Path:
    """Where equation images belong for a given slide source.

    Inside a lesson the slides and the handout both live one level below
    the lesson folder, and share its``Figures/`` with the notebooks.
    Originally this covered only `Slides/` and share the lesson's
    `Figures/` with the notebooks. Anywhere else - the syntax example in
    `Course/`, say - keep the images beside the source instead of inventing a
    `Figures/` folder one level up.
    """
    if source.parent.name in ("Slides", "Docs"):
        return source.parent.parent / "Figures"
    return source.parent / "Figures"


def process(source: Path, dest: Path | None = None) -> tuple[int, int, list[str]]:
    """Convert the math in a slide source. Returns (unicode, images, warnings).

    Writes to `dest`, leaving `source` untouched: the author's markdown keeps
    real LaTeX, which is what gets maintained and what the handout build needs.
    Defaults to rewriting in place only when explicitly asked.
    """
    dest = dest or source
    text = source.read_text(encoding="utf8")
    figures = figures_dir(source)
    stats = {"unicode": 0, "images": 0}
    warnings: list[str] = []
    _drawn_on_host.clear()

    def display_repl(match: re.Match) -> str:
        latex = match.group(1).strip()
        path = render_image(latex, figures, display=True)
        if path is None:
            warnings.append(
                f"could not render, left as text: {latex[:60]}"
                " - simplify it or move it to the handout"
            )
            return f"\n`{latex}`\n"
        stats["images"] += 1
        return f"\n![]({path.name})\n"

    text = re.sub(r"\$\$(.+?)\$\$", display_repl, text, flags=re.S)

    def inline_repl(match: re.Match) -> str:
        latex = match.group(1).strip()
        unicode_form = to_unicode(latex)
        if unicode_form is not None:
            stats["unicode"] += 1
            return unicode_form
        # Never fall back to an image here. An image inside a paragraph does
        # not survive the pptx conversion - pandoc drops it and leaves a gap in
        # the sentence. Keep readable text and say so, so the author can
        # simplify the expression or move it to display position.
        warnings.append(
            f"inline maths has no Unicode form, left as text: ${latex}$"
            " - simplify it, or set it on its own line as display maths"
        )
        return latex

    text = re.sub(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", inline_repl, text)
    warnings.extend(_check_display_position(text))
    warnings.extend(_check_display_alone(text))

    if _drawn_on_host:
        warnings.append(
            f"{len(_drawn_on_host)} equation(s) drawn on this machine, not in "
            f"the {CONTAINER} container - start it and delete Figures/eq_*.png "
            "before committing, or the images will not match the ones a "
            "rebuild produces"
        )

    dest.write_text(text, encoding="utf8")
    return stats["unicode"], stats["images"], warnings


def _check_display_position(text: str) -> list[str]:
    """Warn when prose follows display maths inside one slide.

    A block image ends the slide as far as pandoc is concerned: anything after
    it starts a new, untitled one. So display maths has to be the last thing on
    its slide, or the sentence following it is orphaned.
    """
    problems = []
    for section in re.split(r"^# ", text, flags=re.M)[1:]:
        title = section.splitlines()[0].strip()
        after = re.split(r"^!\[\]\(eq_\w+\.png\)$", section, flags=re.M)
        if len(after) < 2:
            continue
        trailing = after[-1].strip()
        if trailing and not trailing.startswith(":::"):
            problems.append(
                f'slide "{title}": text after display maths will land on a new'
                " untitled slide - move the equation to the end of the slide"
            )
    return problems


def _check_display_alone(text: str) -> list[str]:
    """Warn when a display equation shares its slide with a bullet list.

    pandoc then reaches for the two-column "Content with Caption" layout: the
    title shrinks and left-aligns, and the equation is scaled into whatever
    height the bullets left it. A short lead-in sentence is fine and common;
    it is the list that does the damage. Lesson 6's soft-margin slide is the
    case this was written from.
    """
    problems = []
    for section in re.split(r"^# ", text, flags=re.M)[1:]:
        title = section.splitlines()[0].strip()
        body = re.sub(r"::: notes.*?\n:::\n?", "", section, flags=re.S)
        body = "\n".join(body.splitlines()[1:])
        if not re.search(r"^!\[\]\(eq_\w+\.png\)$", body, flags=re.M):
            continue
        if any(line.lstrip().startswith(("- ", "* ")) for line in body.splitlines()):
            problems.append(
                f'slide "{title}": a bullet list shares the slide with display'
                " maths - the title shrinks and the equation is scaled into"
                " what the bullets left. Give the equation its own slide"
            )
    return problems


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    for name in argv:
        path = Path(name)
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 1
        uni, img, warns = process(path)
        print(f"  {path.name}: {uni} to Unicode, {img} to image")
        for warning in warns:
            print(f"    note: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
