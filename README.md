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
| `sokoban/solvers/{astar,idastar,heuristics,deadlock,macros,transposition}.py` | Lakshya | A* / IDA* with Hungarian heuristic, dead-square / freeze / tunnel / PI-corral pruning |
| `sokoban/rl/`              | Jakob   | Gymnasium wrapper, DRC ConvLSTM policy, PPO training + eval pipeline |
| `sokoban/solvers/{ppo,hybrid}.py` | Jakob | `PPOSolver` (inference-time) + `HybridSolver` (policy-guided A*) |

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

# Solve a single level with any registered solver
sokoban solve xsokoban --index 0                       # default: bfs-push
sokoban solve xsokoban --index 3 --solver astar+all
sokoban solve xsokoban --index 3 --solver idastar+all

# Solve + write a GIF of the step-by-step solution
sokoban solve xsokoban --index 0 --animate level_01.gif

# Run a full benchmark, write per-level rows to CSV
sokoban benchmark xsokoban boxoban:medium@50 \
    --solver astar+all --csv bench_results/astar_full.csv

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

## Learning-based stack (optional)

The RL / hybrid solvers are gated behind an optional extra so the
core package stays lightweight. To install:

```bash
pip install -e '.[rl]'
```

This pulls in PyTorch, Stable-Baselines3, and Gymnasium (~1.5 GB).
After installation:

```bash
# Train PPO on the Boxoban unfiltered split (single-GPU run)
sokoban rl-train --tier unfiltered --total-timesteps 5_000_000 --save runs/ppo

# Evaluate a trained checkpoint on all three Boxoban tiers
sokoban rl-eval --checkpoint runs/ppo/ppo_final.zip --csv bench_results/ppo.csv

# Use the policy as a heuristic prior inside A*
# (programmatic for now; CLI flag pending Jakob's checkpoint)
python -c "
from sokoban.solvers.hybrid import hybrid_with_checkpoint
solver = hybrid_with_checkpoint('runs/ppo/ppo_final.zip', alpha=0.5)
"
```

The full DRC architecture lives in `sokoban/rl/policy.py` and defaults
to a scaled-down `D=2, R=3, C=32` configuration that fits a consumer
GPU in ~12 hours per tier.

## CSV schema

Each benchmark run appends rows to a CSV with the columns documented in
`report/REPORT.md` (Appendix A). The same schema is shared across all
solver families.

## Available solvers

| name              | family    | notes |
| ----------------- | --------- | ----- |
| `bfs-push`        | reference | optimal in pushes; sanity-check baseline (Vedant) |
| `astar`           | classical | A* + Hungarian heuristic, no pruning (Lakshya) |
| `astar+dead`      | classical | + precomputed dead-square map |
| `astar+freeze`    | classical | + recursive freeze-deadlock check |
| `astar+tunnels`   | classical | + tunnel macro collapsing |
| `astar+all`       | classical | full pruning suite (dead + freeze + tunnels + PI-corral) |
| `idastar`         | classical | IDA* + Hungarian, no pruning |
| `idastar+all`     | classical | IDA* + full pruning suite |
| `hybrid-uniform`  | hybrid    | A*+all with no policy prior (sanity-check baseline) |
| `ppo` *(planned)* | learned   | trained PPO checkpoint; needs `pip install -e '.[rl]'` and a saved checkpoint |
| `hybrid-ppo` *(planned)* | hybrid | PPO-policy-guided A* once Jakob ships a checkpoint |

Each variant is recorded under its own `solver` column in the
benchmark CSV so the report can quote the marginal contribution of
each pruning technique (proposal Section 4.2).

## Repo layout

```
sokoban/
  env/             XSB parser, board, move/push generation, dead-square map
  viz/             ASCII + matplotlib rendering, step-by-step animation
  bench/           Benchmark harness, CSV I/O, per-tier time budgets
  data/            Boxoban loader, curated XSokoban set
    levels/        xsokoban_curated.xsb (30 levels, packaged with the wheel)
  solvers/
    base.py        Solver protocol + SolveResult
    bfs.py         Reference push-BFS (Vedant)
    heuristics.py  Hungarian-assignment + push-distance maps (Lakshya)
    deadlock.py    Dead squares, freeze-deadlock, PI-corral (Lakshya)
    macros.py      Tunnel macro collapsing (Lakshya)
    transposition.py  Push-state TT keyed by (boxes, canonical player) (Lakshya)
    astar.py       A* with configurable pruning toggles (Lakshya)
    idastar.py     IDA* sharing heuristic + pruning (Lakshya)
  cli.py           sokoban {info,play,solve,animate,benchmark}
tests/             pytest suite (38 tests)
report/            REPORT.md scaffold for final submission
```
