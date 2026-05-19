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
- PPO + ConvLSTM architecture, training curve, single-GPU budget.
- Standalone policy success rate per tier.
- Hybrid policy-guided A*: how the policy ranks pushes; integration
  with Lakshya's transposition table.
- Table 2: PPO vs hybrid vs best classical, all three tiers.
- Figure 3: scatter of pushes-to-solve vs nodes-expanded by solver.

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
