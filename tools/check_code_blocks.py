"""Bind every call in a lesson's printed code against the real signature.

    python tools/check_code_blocks.py 05

Run by ``tools/verify_lesson.py``, inside the course container, so the
signatures are the ones the image ships rather than whatever the host venv has
drifted to.

**Why binding rather than running.** Most printed blocks are fragments: the
lesson-5 deck writes ``sel.fit(noise, y)`` where ``noise`` was built three
slides earlier, and a slide that had to carry its own setup would stop being a
slide. So the blocks cannot be executed, and a checker that tried would report
a NameError for every one of them and be switched off within a week.

What *can* be checked without any data is whether each call could ever succeed:
that the block parses, that a name it imports exists, and that its arguments
bind to the callable's signature. That is a narrow question, and it is the one
that matters, because the failure it catches is the failure a student meets
first — they copy the four lines off the slide and get a traceback before any
of the lesson's content is in play.

The fault that prompted it: lesson 5's deck printed

    sel = SelectKBest(f_classif, 10)

which has been a TypeError since ``k`` became keyword-only. The handout and the
notebook both wrote ``k=10``; only the deck, the one artefact students copy
from, carried it. It survived a full review pass because reading it proves
nothing - the signature is not in the reader's head.

Anything the checker cannot resolve it ignores. A name it has never heard of is
not evidence of a fault, and a checker that guesses is a checker that gets
disabled.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Where a bare name in a printed block is looked up. The course draws on a
#: small, stable part of scikit-learn; anything outside this list is skipped
#: rather than guessed at.
SEARCH_MODULES = (
    "sklearn.cluster", "sklearn.compose", "sklearn.datasets",
    "sklearn.decomposition", "sklearn.ensemble", "sklearn.feature_selection",
    "sklearn.impute", "sklearn.linear_model", "sklearn.metrics",
    "sklearn.model_selection", "sklearn.naive_bayes", "sklearn.neighbors",
    "sklearn.pipeline", "sklearn.preprocessing", "sklearn.svm", "sklearn.tree",
)

#: Directories whose markdown reaches a student.
ARTEFACTS = ("Docs", "Slides", "Exercises", "Resources")

BLOCK = re.compile(r'^```python\n(.*?)^```', re.S | re.M)


def library_symbols() -> dict:
    """Public names the course uses without importing them in the block."""
    table: dict[str, object] = {}
    for name in SEARCH_MODULES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        for attribute in dir(module):
            if not attribute.startswith("_"):
                table.setdefault(attribute, getattr(module, attribute))
    return table


def blocks(lesson: Path):
    """Every printed python block, with the file and line it starts on."""
    for folder in ARTEFACTS:
        for path in sorted((lesson / folder).glob("*.md")):
            text = path.read_text(encoding="utf-8")
            for match in BLOCK.finditer(text):
                line = text[:match.start()].count("\n") + 2
                yield path, line, match.group(1)


def imported(tree: ast.AST, problems: list, where: str) -> dict:
    """Resolve the block's own imports, so its local names win over the table.

    An import that cannot be resolved is reported: a brief telling a student to
    write ``from turbine_data import load_turbines`` is making a promise about a
    file in the repository, and that promise is checkable even when the rest of
    the block is not.
    """
    local: dict[str, object] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            try:
                module = importlib.import_module(node.module)
            except Exception as error:                       # noqa: BLE001
                problems.append(f"{where}: cannot import {node.module} "
                                f"({type(error).__name__}: {error})")
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                if not hasattr(module, alias.name):
                    problems.append(f"{where}: {node.module} has no "
                                    f"{alias.name!r}")
                    continue
                local[alias.asname or alias.name] = getattr(module, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                try:
                    local[alias.asname or alias.name.split(".")[0]] = \
                        importlib.import_module(alias.name)
                except ImportError:
                    pass
    return local


def rebound(tree: ast.AST) -> set:
    """Names the block assigns itself, which therefore are not the library's."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                names.update(n.id for n in ast.walk(target)
                             if isinstance(n, ast.Name))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                               ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.For):
            names.update(n.id for n in ast.walk(node.target)
                         if isinstance(n, ast.Name))
    return names


def resolve(node: ast.expr, local: dict, table: dict, shadowed: set):
    """The object a call's callee refers to, or None if it cannot be known."""
    if isinstance(node, ast.Name):
        if node.id in local:
            return local[node.id]
        if node.id in shadowed:
            return None
        return table.get(node.id)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        owner = local.get(node.value.id)
        if inspect.ismodule(owner):
            return getattr(owner, node.attr, None)
    return None


def elided(call: ast.Call) -> bool:
    """Blocks that deliberately write `...` are showing a shape, not a call."""
    for argument in list(call.args) + [k.value for k in call.keywords]:
        if isinstance(argument, ast.Constant) and argument.value is Ellipsis:
            return True
    if any(isinstance(a, ast.Starred) for a in call.args):
        return True
    return any(k.arg is None for k in call.keywords)


def check_block(source: str, where: str, table: dict) -> list[str]:
    problems: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [f"{where}: does not parse - {error.msg} on its line {error.lineno}"]

    local = imported(tree, problems, where)
    shadowed = rebound(tree)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or elided(node):
            continue
        target = resolve(node.func, local, table, shadowed)
        if target is None or not callable(target):
            continue
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            continue
        keywords = {k.arg: object() for k in node.keywords if k.arg}
        try:
            signature.bind(*[object()] * len(node.args), **keywords)
        except TypeError as error:
            name = ast.unparse(node.func)
            problems.append(f"{where}: {name}(...) will not run - {error}; "
                            f"the signature is {name}{readable(signature)}")
    return problems


def readable(signature: inspect.Signature) -> str:
    """The signature without the repr of a default that carries an address.

    ``SelectKBest``'s default score_func prints as
    ``<function f_classif at 0x7f...>``, which changes every run and would put
    a different byte in the report each time it fired.
    """
    class Elided:
        def __repr__(self) -> str:
            return "..."

    parameters = []
    for parameter in signature.parameters.values():
        if parameter.default is not inspect.Parameter.empty \
                and " at 0x" in repr(parameter.default):
            parameter = parameter.replace(default=Elided())
        parameters.append(parameter)
    return str(signature.replace(parameters=parameters))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("lesson", help="lesson number, e.g. 05")
    args = parser.parse_args()

    matches = sorted(ROOT.glob(f"Lessons/{int(args.lesson):02d}_*"))
    if not matches:
        print(f"no lesson {args.lesson} under Lessons/")
        return 2
    lesson = matches[0]

    table = library_symbols()

    problems, counted = [], 0
    for path, line, source in blocks(lesson):
        counted += 1
        # A brief's loader snippet imports a module shipped beside it - the
        # exercise data generators live in Exercises/, the lesson's in
        # Notebooks/ - so a block is resolved from where a student would run
        # it, which is the folder it was printed in.
        for folder in (path.parent, lesson / "Notebooks"):
            if str(folder) not in sys.path:
                sys.path.insert(0, str(folder))
        where = f"{path.relative_to(ROOT).as_posix()}:{line}"
        problems.extend(check_block(source, where, table))

    for problem in problems:
        print(problem)
    if not problems:
        print(f"{lesson.name}: {counted} printed code block(s), every call binds")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
