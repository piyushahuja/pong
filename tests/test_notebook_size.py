"""
Keep notebook outputs small enough to live in git.

The worksheets save their outputs on purpose, so they can be read before being
run. That is worth the diff noise for text and small figures. It is not worth it
for `replay.watch()`, which embeds every frame of an episode as base64 PNG: that
one cell once made a notebook 7.5 MB, and because every re-run writes a fresh
blob, five copies ended up in history at roughly 37 MB.

So the rule is: save outputs, except the animation. These budgets are generous
enough for several matplotlib figures and tight enough that an embedded animation
fails immediately.
"""

import json
from pathlib import Path

import pytest

from pong.utils import PROJECT_ROOT

MAX_NOTEBOOK_BYTES = 256 * 1024
MAX_SINGLE_OUTPUT_BYTES = 100 * 1024

NOTEBOOKS = sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb"))


def test_there_are_notebooks_to_check():
    assert NOTEBOOKS, "no notebooks found, so the budget checks below prove nothing"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_small_enough_for_git(path):
    size = path.stat().st_size
    assert size <= MAX_NOTEBOOK_BYTES, (
        f"{path.name} is {size / 1024:.0f} KB, over the {MAX_NOTEBOOK_BYTES / 1024:.0f} KB "
        f"budget. Usually this means a cell saved an animation or a very large figure. "
        f"Run `uv run scripts/strip_heavy_outputs.py` and commit again."
    )


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_single_output_is_huge(path):
    notebook = json.loads(path.read_text())

    for index, cell in enumerate(notebook["cells"]):
        for output in cell.get("outputs", []):
            for mime, payload in (output.get("data") or {}).items():
                text = "".join(payload) if isinstance(payload, list) else str(payload)
                assert len(text) <= MAX_SINGLE_OUTPUT_BYTES, (
                    f"{path.name} cell {index} has a {len(text) / 1024:.0f} KB {mime} "
                    f"output, over the {MAX_SINGLE_OUTPUT_BYTES / 1024:.0f} KB budget. "
                    f"If this is replay.watch(), clear the cell: the animation does not "
                    f"render on GitHub anyway, so saving it costs megabytes for nothing. Fix: "
                    f"`uv run scripts/strip_heavy_outputs.py`."
                )
