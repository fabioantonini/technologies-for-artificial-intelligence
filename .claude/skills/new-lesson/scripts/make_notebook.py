#!/usr/bin/env python3
"""Build a .ipynb from a list of (kind, source) pairs.

    import sys; sys.path.insert(0, ".claude/skills/new-lesson/scripts")
    from make_notebook import build

    build([("md",   "# Decision trees from scratch"),
           ("md",   "## 1. Setup"),
           ("code", "import numpy as np"),
          ], "Lessons/07_trees_and_ensembles/Notebooks/01_trees.ipynb")

Lessons 7 and 8 each wrote a script like this from scratch before authoring
their notebooks, which is the signal to ship one rather than let a third
session write a third.

Authoring cells as Python data beats editing .ipynb JSON directly: the source
stays readable and diffable while it is being written, and a notebook can be
regenerated wholesale after an edit instead of patched in place. Keep the
generator script beside the notebook while drafting - name it with a leading
underscore so `lesson_state.py` does not mistake it for the dataset module -
and delete it once the notebook is final.

The notebook this writes has no outputs. Execute it in the container, which is
also what puts real outputs in for committing:

    docker exec tai_course bash -lc 'cd /home/jovyan/work/Lessons/NN_*/Notebooks \\
        && jupyter nbconvert --to notebook --execute --inplace 01_*.ipynb'

Then fix ownership, because the container runs as root and everything it writes
lands on the host owned by root:

    docker exec tai_course chown -R 1000:100 /home/jovyan/work/Lessons/NN_*
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

KINDS = {"md": "markdown", "markdown": "markdown", "code": "code"}

METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}


def build(cells: list[tuple[str, str]], path: str | Path) -> Path:
    """Write `cells` to `path` as a notebook. Returns the path written."""
    out = []
    for kind, source in cells:
        if kind not in KINDS:
            raise ValueError(f"unknown cell kind {kind!r}; use md or code")
        cell = {
            "cell_type": KINDS[kind],
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            # A trailing newline on the last line makes git diffs behave and
            # matches what nbconvert writes back after execution.
            "source": source.strip("\n").splitlines(keepends=True),
        }
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        out.append(cell)

    notebook = {"cells": out, "metadata": METADATA,
                "nbformat": 4, "nbformat_minor": 5}
    path = Path(path)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf8")
    return path


if __name__ == "__main__":
    print(__doc__)
