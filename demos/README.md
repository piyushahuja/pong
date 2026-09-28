# Demos

Interactive explanations that run in a browser, no install. The worksheets link to them at
the point where they help, so they stay out of the notebooks and add nothing to their size.

Served at <https://piyushahuja.com/pong/demos/>, or open the files directly from a
clone.

| Demo | What it shows | Linked from |
|---|---|---|
| `forward-pass.html` | A real rally through the trained policy: the difference image on the court, 200 hidden units as bars, the two committees adding up their votes. Every number computed in the browser from the actual weights. | worksheets 2 and 4 |
| `reinforce-gridworld.html` | A policy learning from nothing, in a gridworld small enough to watch. | worksheet 3 |
| `weights-as-matrix.html` | A layer read two ways: as neurons with their own weights, or as one matrix whose rows are neurons. | worksheet 2, inline |

`three.min.js` is here because `forward-pass.html` needs it. Keep it alongside it.

`weights-as-matrix.html` is 17 KB and is embedded directly in worksheet 2 with
`HTML(...read_text())`. `forward-pass.html` is far larger, about 690 KB with three.js, so it
is linked rather than embedded: inlining it would blow the size budget that
`tests/test_notebook_size.py` enforces.
