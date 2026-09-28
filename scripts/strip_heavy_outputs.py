"""
Clear notebook outputs that are too big to commit.

Re-running a worksheet re-saves every output, including `replay.watch()`, whose
animation embeds a frame per PNG and comes to several megabytes. The size budget in
tests/test_notebook_size.py catches that, and this is the fix it tells you to apply.

    uv run scripts/strip_heavy_outputs.py

Outputs under the budget are left alone, because the worksheets save their outputs
deliberately so they can be read before being run.
"""

import json
import sys
from pathlib import Path

from pong.utils import PROJECT_ROOT

MAX_OUTPUT_BYTES = 100 * 1024


def strip(path):
    """Clear over-budget outputs. Returns the number of cells cleared."""
    notebook = json.loads(path.read_text())
    indent = 1 if path.read_text().startswith('{\n "') else 2
    cleared = 0

    for cell in notebook["cells"]:
        biggest = 0
        for output in cell.get("outputs", []):
            for payload in (output.get("data") or {}).values():
                text = "".join(payload) if isinstance(payload, list) else str(payload)
                biggest = max(biggest, len(text))
        if biggest > MAX_OUTPUT_BYTES:
            cell["outputs"] = []
            cell["execution_count"] = None
            cleared += 1
            first = "".join(cell["source"]).splitlines()[0][:60]
            print(f"  cleared {biggest / 1024:>7.0f} KB from: {first}")

    if cleared:
        path.write_text(json.dumps(notebook, indent=indent, ensure_ascii=False) + "\n")
    return cleared


def main():
    total = 0
    for path in sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb")):
        total += strip(path)
    print(f"{total} cell(s) cleared" if total else "nothing over budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())
