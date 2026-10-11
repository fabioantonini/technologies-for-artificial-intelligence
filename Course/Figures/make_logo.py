"""The course logo, for the top of the repository's README.

    python Course/Figures/make_logo.py

Writes course_logo.svg (for a light page) and course_logo_dark.svg (for GitHub's
dark theme) beside this file. The emblem is the course in one picture: two
classes of points that no straight line separates, and the curved boundary a
model learns between them - the pumps of lesson 6, the fence of lesson 9. The
colours are the ones every figure in the course uses.
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BLUE, TEAL, RUST, GOLD = "#1F4E79", "#2E7D8A", "#9C4221", "#F0BE50"
SIZE, RADIUS = 168, 46          # emblem side; radius of the learned boundary


def points(rng):
    """Inner class inside the boundary, outer class in a ring around it."""
    out = []
    while len(out) < 34:
        x, y = rng.uniform(-RADIUS + 6, RADIUS - 6, 2)
        if x * x + y * y < (RADIUS - 9) ** 2:
            out.append((x, y, "inner"))
    while len(out) < 34 + 40:
        a, r = rng.uniform(0, 2 * np.pi), rng.uniform(RADIUS + 9, 68)
        x, y = r * np.cos(a), r * np.sin(a)
        if abs(x) < 68 and abs(y) < 68:
            out.append((x, y, "outer"))
    return out


def emblem(theme):
    c = SIZE / 2
    tile = BLUE if theme == "light" else "#16212C"
    edge = "" if theme == "light" else ' stroke="#3B5368" stroke-width="2"'
    inner = "#7CC4CF"
    dots = "\n".join(
        f'    <circle cx="{c + x:.1f}" cy="{c + y:.1f}" r="4.2" '
        f'fill="{inner if kind == "inner" else "#D9825F"}"/>'
        for x, y, kind in points(np.random.default_rng(7))
    )
    # The boundary is drawn slightly irregular: a learned curve, not a compass circle.
    t = np.linspace(0, 2 * np.pi, 121)
    r = RADIUS + 2.2 * np.sin(3 * t + 0.6) + 1.4 * np.cos(5 * t)
    path = " ".join(f"{'M' if i == 0 else 'L'}{c + ri * np.cos(ti):.1f},{c + ri * np.sin(ti):.1f}"
                    for i, (ti, ri) in enumerate(zip(t, r))) + " Z"
    return f"""  <g>
    <rect x="0" y="0" width="{SIZE}" height="{SIZE}" rx="30" fill="{tile}"{edge}/>
{dots}
    <path d="{path}" fill="none" stroke="{GOLD}" stroke-width="4.5" stroke-linejoin="round"/>
  </g>"""


def logo(theme):
    ink = BLUE if theme == "light" else "#E6EDF3"
    soft = "#5B6B7A" if theme == "light" else "#9FB0C0"
    accent = TEAL if theme == "light" else "#7CC4CF"
    font = "'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif"
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="760" height="{SIZE}" viewBox="0 0 760 {SIZE}" role="img" aria-label="Technologies for Artificial Intelligence">
  <title>Technologies for Artificial Intelligence</title>
{emblem(theme)}
  <g font-family="{font}">
    <text x="196" y="58" font-size="26" fill="{accent}" font-weight="600" letter-spacing="0.5">Technologies for</text>
    <text x="194" y="104" font-size="44" fill="{ink}" font-weight="700">Artificial Intelligence</text>
    <rect x="196" y="120" width="64" height="4" rx="2" fill="{GOLD}"/>
    <text x="196" y="150" font-size="17" fill="{soft}">MSc in Computer Science · Università degli Studi dell'Aquila</text>
  </g>
</svg>
"""


if __name__ == "__main__":
    for theme, name in (("light", "course_logo.svg"), ("dark", "course_logo_dark.svg")):
        (HERE / name).write_text(logo(theme), encoding="utf-8")
        print("wrote", name)
