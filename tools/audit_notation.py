"""Check that one symbol means one thing across the whole course.

    python tools/audit_notation.py            # every lesson
    python tools/audit_notation.py 03 07      # only these

`tools/verify_lesson.py` recomputes the numbers a handout works out by hand. It
does not read the symbols those numbers are attached to, and that gap has cost
something twice. Lesson 1 defined the empirical risk with $n$ and stated the
unbiasedness theorem with $m$ sixty lines later, while CLAUDE.md reserves $n$ for
the number of features — so, read with the notation table in hand, its Section
2.4 claimed the standard error falls with the root of the *feature* count.
Lesson 7 was worse: $n$ meant examples in the split criterion and features in
`max_features`, five hundred lines apart, and one line used $p$ for the feature
count while the same lesson's table gave $p$ to a predicted probability.

Neither failed a test. Both were found by reading a formula. This makes the
reading mechanical.

Two passes:

  1. **Sentence level.** Where a reserved symbol appears in the same sentence as
     words that contradict its meaning — $n$ next to "examples", $\\alpha$ next to
     "penalty" — report it.

  2. **Table level.** Every lesson that declares its own notation table is
     compared against the shared table in CLAUDE.md, and a row that redefines a
     shared symbol is reported.

Exits non-zero when anything is reported, so it can gate a commit.

**It reports suspects, not verdicts.** A local redefinition is legitimate when a
lesson declares it — lesson 10's $n$ is an input's spatial size and says so. Read
every hit; the point is that nothing reaches a student unread, not that the tool
is right.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXAMPLES = r"examples?|samples?|rows?|observations?|data ?points?|instances?"
FEATURES = r"features?|columns?|dimensions?|predictors?|covariates?|inputs?"
LEARNING_RATE = r"learning rate|step size|step length"
REGULARISATION = r"regularisation|regularization|penalt|shrinkage"

# Reported as "<what was seen>": symbol pattern, contradicting words, excusing words.
SENTENCE_RULES = [
    ("n used for examples", r"\$n\$", EXAMPLES, FEATURES),
    ("m used for features", r"\$m\$", FEATURES, EXAMPLES),
    ("p used for features", r"\$p\$", FEATURES, r"probabilit|proportion|fraction|padding"),
    ("alpha used for regularisation", r"\\alpha", REGULARISATION,
     LEARNING_RATE + r"|scikit|`alpha`|reserves"),
    ("lambda used for a learning rate", r"\\lambda", LEARNING_RATE,
     REGULARISATION + r"|reserves|spells"),
]


# Suspects read by hand on 24 August 2026 and accepted, with the reason. A new
# finding is a finding; these four are not, and burying them would stop the tool
# being usable as a gate. Delete a row and it comes back.
ACCEPTED = {
    ("06", "local table redefines $x$"):
        "a feature vector is one input; lesson 6 needs a second one, $x'$, for distances",
    ("06", "m used for features"):
        "the Resources note reads '$O(mn)$ for $m$ stored items in $n$ dimensions', "
        "which is the shared meaning - 'items' is simply not in the word list",
    ("08", "local table redefines $J$"):
        "the within-cluster sum of squares IS the cost k-means minimises",
    ("10", "local table redefines $n$"):
        "declared locally as an image's spatial size; a convolutional lesson has no "
        "separate feature count to clash with",
}


def sentences(text):
    """Prose is hard-wrapped; unwrap it so a sentence is one string."""
    text = re.sub(r"\n(?!\n)", " ", text)
    for paragraph in text.split("\n\n"):
        for sentence in re.split(r"(?<=[.:;])\s+", paragraph):
            yield sentence.strip()


def shared_table():
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    start = text.index("| Symbol | Meaning | Where it appears |")
    block = text[start:text.index("\n\n", start)]
    table = {}
    for row in block.splitlines()[2:]:
        cells = [c.strip() for c in row.strip("|").split("|")]
        for symbol in re.findall(r"\$([^$]+)\$", cells[0]):
            table[symbol] = cells[1]
    return table


def local_table(text):
    """The lesson's own '### Notation used in this lesson', if it has one."""
    start = text.find("Notation used in this lesson")
    if start < 0:
        return None
    table = {}
    for row in text[start:].splitlines():
        row = row.strip()
        if not row.startswith("|"):
            if table:
                break
            continue
        cells = [c.strip() for c in row.strip("|").split("|")]
        if len(cells) < 2 or not cells[0].startswith("$"):
            continue
        for symbol in re.findall(r"\$([^$]+)\$", cells[0]):
            table[symbol] = cells[1]
    return table


def overlapping(a, b):
    """Do two meanings share any content word? Crude, and deliberately so."""
    words = lambda s: set(re.sub(r"[^a-z ]", "", s.lower()).split()) - {
        "the", "a", "an", "of", "and", "or", "in", "its", "one", "number"}
    return bool(words(a) & words(b))


def audit(lessons):
    master = shared_table()
    findings = []

    for folder in sorted((ROOT / "Lessons").iterdir()):
        if not folder.is_dir() or folder.name[:2] not in lessons:
            continue
        number = folder.name[:2]

        for path in sorted(folder.rglob("*.md")):
            text = path.read_text(encoding="utf-8")
            where = path.relative_to(ROOT)

            for label, symbol, against, excused in SENTENCE_RULES:
                for sentence in sentences(text):
                    if (re.search(symbol, sentence)
                            and re.search(against, sentence, re.I)
                            and not re.search(excused, sentence, re.I)):
                        findings.append((number, label, str(where), sentence[:170]))

            # an average over examples should be 1/m; k and B are ensemble counts
            for match in re.finditer(r"\\frac\{1\}\{([a-zA-Z])\}\s*\\sum", text):
                if match.group(1) not in ("m", "k", "B", "N", "T"):
                    findings.append((number, "average uses 1/%s, not 1/m" % match.group(1),
                                     "%s:%d" % (where, text[:match.start()].count("\n") + 1),
                                     match.group(0)))

            for match in re.finditer(r"\\mathcal\{D\}\^\{?([a-zA-Z])\}?", text):
                if match.group(1) != "m":
                    findings.append((number, "D raised to %s, not m" % match.group(1),
                                     "%s:%d" % (where, text[:match.start()].count("\n") + 1),
                                     match.group(0)))

            if path.parent.name == "Docs" and "worked_examples" not in path.name:
                local = local_table(text)
                if local is None:
                    findings.append((number, "no notation table in this handout",
                                     str(where), ""))
                    continue
                for symbol, meaning in local.items():
                    if symbol in master and not overlapping(master[symbol], meaning):
                        findings.append((
                            number, "local table redefines $%s$" % symbol, str(where),
                            "CLAUDE.md: %s   |   lesson: %s" % (master[symbol], meaning)))
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lessons", nargs="*", help="lesson numbers, e.g. 03 07")
    parser.add_argument("--show-accepted", action="store_true",
                        help="also list the suspects already read and accepted")
    args = parser.parse_args()
    wanted = set(args.lessons) or {f.name[:2] for f in (ROOT / "Lessons").iterdir()
                                   if f.is_dir()}

    findings = audit(wanted)
    accepted = [f for f in findings if (f[0], f[1]) in ACCEPTED]
    findings = [f for f in findings if (f[0], f[1]) not in ACCEPTED]

    if args.show_accepted and accepted:
        print("read and accepted:")
        for number, label, where, _ in accepted:
            print("  lesson %s  %s" % (number, label))
            print("    %s" % ACCEPTED[(number, label)])
        print()

    if not findings:
        if accepted and not args.show_accepted:
            print("notation: %d accepted exception(s) not shown "
                  "(--show-accepted)" % len(accepted))
        print("notation: nothing to look at in %d lesson(s)" % len(wanted))
        return 0

    current = None
    for number, label, where, detail in findings:
        if number != current:
            print("\nlesson %s" % number)
            current = number
        print("  %s" % label)
        print("    %s" % where)
        if detail:
            print("    %s" % detail)
    print("\n%d thing(s) to read. Each is a suspect, not a verdict." % len(findings))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
