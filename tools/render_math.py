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
import sys
from pathlib import Path

SUPERSCRIPT = str.maketrans("0123456789+-=()aeioruvxn", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᵃᵉⁱᵒʳᵘᵛˣⁿ")
SUBSCRIPT = str.maketrans("0123456789+-=()aeioruvxn", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢₒᵣᵤᵥₓₙ")

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
    r"\hat": "", r"\,": " ", r"\;": " ", r"\!": "", r"\ ": " ",
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


def to_unicode(latex: str) -> str | None:
    """Render simple inline math as Unicode, or None if it cannot be done."""
    if TOO_COMPLEX.search(latex):
        return None

    text = _flatten_fractions(latex)
    text = ROOT.sub(lambda m: f"√{m.group(1)}", text)
    for command, symbol in sorted(SYMBOLS.items(), key=lambda kv: -len(kv[0])):
        text = text.replace(command, symbol)

    def script(match: re.Match, table: dict) -> str:
        body = match.group(1) or match.group(2) or ""
        converted = body.translate(table)
        # Refuse a partial conversion - a stray ^ looks worse than an image.
        return converted if converted != body or not body else body

    text = re.sub(r"\^\{([^{}]*)\}|\^(\w)", lambda m: script(m, SUPERSCRIPT), text)
    text = re.sub(r"_\{([^{}]*)\}|_(\w)", lambda m: script(m, SUBSCRIPT), text)

    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text).strip()
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


def _normalise(latex: str) -> str:
    for command, replacement in sorted(
        MATHTEXT_ALIASES.items(), key=lambda kv: -len(kv[0])
    ):
        latex = latex.replace(command, replacement)
    return latex


def render_image(latex: str, figures: Path, display: bool = True) -> Path | None:
    """Render one equation to a transparent PNG, or None if mathtext refuses.

    A formula we cannot draw must never break the build: the caller falls back
    to leaving readable text on the slide and warns, so the author can decide
    whether to simplify it or move it to the handout where it belongs.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5(latex.encode()).hexdigest()[:10]
    out = figures / f"eq_{digest}.png"
    if out.exists():
        return out

    size = 26 if display else 18
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
