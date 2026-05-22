# A Comparative Study of Classical Search and Learning-Based Methods for Sokoban

CS 175 · Spring 2026 · Group 24
Lakshya Malik, Vedant Khatri, Jakob Groh

> Final-report scaffold. Each section lists the figures, tables, and
> CSV-derived numbers that need to live there at submission time. The
> bullet points are inputs from each owner, not prose to keep — replace
> them with the actual writeup as it lands.

## 1. Introduction
- Restate the project framing: Sokoban as a PSPACE-complete planning
  benchmark with irreversible actions.
- Motivation: why Sokoban over Sudoku, anchored to proposal Section 1.
- Contributions, one bullet per solver family.

## 2. Environment and Infrastructure (Vedant)
- XSB level parser, board representation, push/move generator (`sokoban/env`).
- Solution-string convention: lowercase `u/d/l/r`, UPPERCASE on push.
- Visualizer: static ASCII + matplotlib, plus per-step animation.
- Benchmark harness: CSV schema (`solver, tier, level_name, status,
  pushes, moves, nodes_expanded, nodes_generated, time_seconds,
  peak_memory_bytes, optimal, solution_len_chars, error`).
- Datasets: Boxoban loader (deterministic `seed=175` sampler), 30
  curated XSokoban levels shipped in-tree.
- Figure 1: example level + solution overlay (animation frame strip).

## 3. Classical Search (Lakshya)
- A* with Hungarian-assignment heuristic — baseline.
- IDA* — memory-bounded variant.
- Pruning: dead squares, freeze deadlocks, tunnel macros, PI-corral.
- Table 1: per-tier solve rate and median time for `astar`,
  `astar+freeze`, `astar+freeze+tunnels`, `astar+all-pruning`.
- Figure 2: cumulative-solved curve vs time budget on Boxoban hard.

## 4. Learning-Based and Hybrid Solvers (Jakob)

### 4.1 Infrastructure
- `sokoban/rl/env_wrapper.py`: Gymnasium env over our Sokoban
  environment, 5-channel binary observation (walls / goals / boxes /
  player / box-on-goal), step-penalty + box-on-goal-reward shaping.
- `sokoban/rl/policy.py`: scaled-down Deep Repeating ConvLSTM
  (default `D=2`, `R=3`, `C=32`) ≈ 0.6 M parameters, fits a single
  consumer GPU.
- `sokoban/rl/train.py`: PPO training script using SB3 with a custom
  features extractor over the DRC backbone.
- `sokoban/rl/evaluate.py`: per-tier standalone evaluation that
  writes to the shared benchmark CSV.
- `sokoban/solvers/ppo.py`: `PPOSolver` implementing the common
  `Solver` protocol; greedy + stochastic rollout, illegal-action
  fallback.
- `sokoban/solvers/hybrid.py`: `HybridSolver` extending `AStarSolver`
  with a policy-entropy heuristic bias; degrades cleanly to plain
  A*+all when no checkpoint is loaded.

### 4.2 Training procedure
- Boxoban training split, level shuffle per episode.
- Reward shaping: -0.1 per step, ±1.0 per box on/off a goal, +10.0
  on solve.
- PPO defaults: lr 3e-4, n_steps 128, batch 256, 4 epochs, γ 0.99,
  λ 0.95, entropy coefficient 0.01. Total 5 M timesteps fits in a
  single overnight run.
- Checkpoints every 200 K steps; report quotes the best validation
  checkpoint.

### 4.3 Standalone PPO results
- Table 2a: success rate per tier (unfiltered / medium / hard).
- Figure 3a: training curve (mean episode reward vs. timesteps).

### 4.4 Hybrid policy-guided A* results
- `HybridSolver` adds α·entropy(π) to the Hungarian heuristic;
  α defaults to 0.5. The optimality flag is recorded as `False`
  because the prior breaks strict admissibility — the empirical
  push counts are typically within 1 of the optimum on solved
  levels (cross-check against `astar+all`).
- Table 2b: PPO vs `astar+all` vs `hybrid-ppo`, all three tiers.
- Figure 3b: scatter of pushes-to-solve vs. nodes-expanded by
  solver.

## 5. Evaluation (joint)
- Benchmark set: 200 × {unfiltered, medium, hard} + 30 XSokoban =
  630 levels (proposal Section 4.2).
- Time budgets: 60s easier tiers, 300s hard.
- Aggregate Table 3 — success rate (%), median wall-clock, median
  nodes expanded, median pushes — one row per solver, one column per
  tier.
- Discussion: where each family wins/loses, the trade-off curve.

## 6. Qualitative Analysis
- Hand-picked levels where solvers disagree.
- Animations for: BFS reference (Vedant), best classical
  (Lakshya), hybrid (Jakob).
- Failure-mode breakdown (timeout vs deadlock-pruned vs unsolvable).

## 7. Conclusion and Future Work
- Whether the moonshot landed (hybrid > classical on hard tier).
- Caveats: parameter budget for PPO, deterministic Boxoban order,
  Hungarian heuristic limits.

## Reproducibility Checklist
- All CSVs live in `bench_results/` under deterministic filenames.
- Sampling seed: `175`.
- The 30 XSokoban curated levels are version-controlled in
  `sokoban/data/levels/xsokoban_curated.xsb`.
- Boxoban revision: pinned at first `ensure_boxoban()` call; record
  the commit hash in the final report appendix.
- Final-run command lines, one per row in the appendix.

## Appendix A. Per-Level CSV Schema

| field                | type   | meaning                                         |
| -------------------- | ------ | ----------------------------------------------- |
| `solver`             | str    | solver name (e.g. `astar+freeze`)               |
| `tier`               | str    | dataset tier (`unfiltered`/`medium`/`hard`/`xsokoban`) |
| `level_name`         | str    | unique level id                                 |
| `status`             | str    | `solved`/`timeout`/`memory`/`unsolvable`/`error` |
| `pushes`             | int    | box pushes in the returned plan                  |
| `moves`              | int    | total moves (including walks)                    |
| `nodes_expanded`     | int    | search nodes expanded                            |
| `nodes_generated`    | int    | search nodes generated                           |
| `time_seconds`       | float  | wall-clock                                       |
| `peak_memory_bytes`  | int    | tracemalloc peak (Python only)                  |
| `optimal`            | bool   | self-reported optimality                         |
| `solution_len_chars` | int    | length of returned solution string               |
| `error`              | str    | exception text on failure                        |
