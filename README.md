# 175_sokoban

CS 175 · Group 24 · Spring 2026.
A comparative study of classical search and learning-based methods for
Sokoban.

This repository hosts the shared codebase for the project. It is split
along the work division agreed in `Sokoban_Work_Division_Elite.docx`:

| Module path                | Owner   | Responsibility                                          |
| -------------------------- | ------- | ------------------------------------------------------- |
| `sokoban/env/`             | Vedant  | XSB parser, board, move and push generation             |
| `sokoban/viz/`             | Vedant  | ASCII + matplotlib rendering and solution animation     |
| `sokoban/bench/`           | Vedant  | Benchmark harness, CSV schema, batch runner             |
| `sokoban/data/`            | Vedant  | Boxoban loader + 30 curated XSokoban levels             |
| `sokoban/solvers/bfs.py`   | Vedant  | Reference BFS solver (sanity check, not a contribution) |
| `sokoban/solvers/astar*`   | Lakshya | A* / IDA* with Hungarian heuristic and pruning          |
| `sokoban/solvers/ppo*`     | Jakob   | PPO policy and hybrid policy-guided A*                  |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Inspect the 30 curated XSokoban levels
sokoban info xsokoban

# Print one level
sokoban play xsokoban --index 0

# Solve a single level with the reference BFS solver
sokoban solve xsokoban --index 0

# Solve + write a GIF of the step-by-step solution
sokoban solve xsokoban --index 0 --animate level_01.gif

# Run a full benchmark, write per-level rows to CSV
sokoban benchmark xsokoban boxoban:medium@50 \
    --solver bfs-push --csv bench_results/baseline.csv

# Run tests
pytest
```

## Level sources

The CLI accepts the following source strings:

* `xsokoban` — the 30 curated XSokoban levels (`sokoban/data/levels/xsokoban_curated.xsb`).
* `boxoban:<tier>` — defaults to 200 levels from `<tier>` (`unfiltered`,
  `medium`, or `hard`) of the DeepMind Boxoban dataset.
* `boxoban:<tier>:<split>@<n>` — explicit split (`train`/`valid`/`test`)
  and sample size.
* Any path to a `.xsb` file (single or multi-level).

The Boxoban dataset is cloned on first use into
`~/.cache/sokoban175/boxoban-levels` (overridable via
`SOKOBAN_DATA_DIR`).

## Solution-string convention

Following Section 2 of the proposal, the solver output uses lowercase
`u/d/l/r` for plain moves and UPPERCASE `U/D/L/R` for box pushes. The
benchmark harness records the trace verbatim in the `solution_len_chars`
and `pushes` columns.

## CSV schema

Each benchmark run appends rows to a CSV with the columns documented in
`report/REPORT.md` (Appendix A). The same schema is shared across all
solver families.

## Repo layout

```
sokoban/
  env/        XSB parser, board, move/push generation, dead-square map
  viz/        ASCII + matplotlib rendering, step-by-step animation
  bench/      Benchmark harness, CSV I/O, per-tier time budgets
  data/       Boxoban loader, curated XSokoban set
    levels/   xsokoban_curated.xsb (30 levels, packaged with the wheel)
  solvers/    Common solver interface + reference BFS
  cli.py      sokoban {info,play,solve,animate,benchmark}
tests/        pytest suite
report/       REPORT.md scaffold for final submission
```
