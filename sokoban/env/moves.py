"""Move and push generation, deadlock primitives, and trace utilities.

The transition function is deliberately split into two layers:

* :func:`apply_action` operates in the *move* space (player walks one
  cell, possibly pushing one box). This is what the visualiser and the
  RL policy step in, and it is what produces the proposal's solution
  string (``u/d/l/r`` vs ``U/D/L/R``).

* :func:`apply_push` operates in the *push* space (player teleports to
  the cell next to a box and pushes it once). Classical search planners
  generally search in this compressed space because move-only steps are
  irrelevant for optimality in pushes. We expose both so Lakshya's A*
  / IDA* and Jakob's PPO can each use the representation that fits.

The ``is_simple_deadlock`` helper here is the cheap, layout-only
dead-square test based on the static board. Freeze deadlocks, PI-corral
pruning, and the recursive deadlock checks live in Lakshya's solver
module — they need the search context that this layer deliberately
does not see.
"""

from __future__ import annotations

from collections import deque
from typing import FrozenSet, Iterator, List, Tuple

from sokoban.env.board import (
    ACTION_NAMES,
    ACTIONS,
    Action,
    Board,
    DXDY,
    Pos,
    State,
)


# ---------------------------------------------------------------------------
# Move space
# ---------------------------------------------------------------------------


def _step(pos: Pos, action: Action) -> Pos:
    dr, dc = DXDY[action]
    return (pos[0] + dr, pos[1] + dc)


def legal_actions(board: Board, state: State) -> List[Action]:
    """Return the list of directional actions the player can take.

    A direction is legal if either the target cell is free floor, or
    the target contains a box whose far side is free floor (i.e. the
    box can be pushed). Walls and "double-stacked" boxes block movement.
    """

    out: list[Action] = []
    for a in ACTIONS:
        target = _step(state.player, a)
        if target not in board.floor:
            continue
        if target in state.boxes:
            beyond = _step(target, a)
            if beyond in board.floor and beyond not in state.boxes:
                out.append(a)
        else:
            out.append(a)
    return out


def apply_action(
    board: Board, state: State, action: Action
) -> Tuple[State, bool]:
    """Apply a single move-space action.

    Returns the resulting state and a flag indicating whether the step
    pushed a box. Raises ``ValueError`` if the action is not legal.
    """

    target = _step(state.player, action)
    if target not in board.floor:
        raise ValueError(f"cannot step into non-floor cell {target}")
    if target in state.boxes:
        beyond = _step(target, action)
        if beyond not in board.floor or beyond in state.boxes:
            raise ValueError(f"cannot push box from {target} to {beyond}")
        new_boxes = (state.boxes - {target}) | {beyond}
        return State(player=target, boxes=frozenset(new_boxes)), True
    return State(player=target, boxes=state.boxes), False


# ---------------------------------------------------------------------------
# Push space
# ---------------------------------------------------------------------------


def player_reachable(board: Board, state: State) -> FrozenSet[Pos]:
    """Cells the player can reach without pushing any box.

    This is the canonical "player component" used as part of the
    push-state-space key in A*/IDA*: states whose box positions match
    and whose player sits in the same component are equivalent for
    search.
    """

    start = state.player
    seen = {start}
    queue = deque([start])
    while queue:
        p = queue.popleft()
        for a in ACTIONS:
            nxt = _step(p, a)
            if nxt in seen:
                continue
            if nxt not in board.floor:
                continue
            if nxt in state.boxes:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return frozenset(seen)


def canonical_player(reachable: FrozenSet[Pos]) -> Pos:
    """Pick a deterministic representative cell from a player component.

    Used as the search-space canonical form. We use the lexicographic
    minimum so the same set always maps to the same key.
    """
    return min(reachable)


def legal_pushes(
    board: Board, state: State
) -> Iterator[Tuple[Pos, Action, State]]:
    """Yield every legal box-push reachable from the current state.

    Each yielded tuple is ``(box, direction, next_state)`` where ``box``
    is the *current* cell of the box being pushed and ``direction`` is
    the push direction (from the player's side toward the destination).
    The returned ``next_state`` already includes the player walking to
    the push-from cell and performing the push.
    """

    reachable = player_reachable(board, state)
    for box in state.boxes:
        for a in ACTIONS:
            dr, dc = DXDY[a]
            from_cell = (box[0] - dr, box[1] - dc)
            to_cell = (box[0] + dr, box[1] + dc)
            if from_cell not in reachable:
                continue
            if to_cell not in board.floor:
                continue
            if to_cell in state.boxes:
                continue
            new_boxes = (state.boxes - {box}) | {to_cell}
            yield box, a, State(player=box, boxes=frozenset(new_boxes))


def apply_push(board: Board, state: State, box: Pos, action: Action) -> State:
    """Apply a single push to ``box`` in direction ``action``.

    Raises ``ValueError`` if the push is not currently legal.
    """
    reachable = player_reachable(board, state)
    dr, dc = DXDY[action]
    from_cell = (box[0] - dr, box[1] - dc)
    to_cell = (box[0] + dr, box[1] + dc)
    if box not in state.boxes:
        raise ValueError(f"no box at {box}")
    if from_cell not in reachable:
        raise ValueError(f"player cannot reach push-from cell {from_cell}")
    if to_cell not in board.floor or to_cell in state.boxes:
        raise ValueError(f"destination {to_cell} is blocked")
    new_boxes = (state.boxes - {box}) | {to_cell}
    return State(player=box, boxes=frozenset(new_boxes))


# ---------------------------------------------------------------------------
# Termination + cheap deadlock test
# ---------------------------------------------------------------------------


def is_solved(board: Board, state: State) -> bool:
    return state.boxes == board.goals


def _simple_dead_squares(board: Board) -> FrozenSet[Pos]:
    """Static dead-square map: cells from which a single box, alone on
    the board, cannot possibly be pushed onto any goal.

    We compute this by BFS *backwards* from every goal in the push
    space, assuming an idealised player that can stand wherever needed.
    Any floor cell not reached is a dead square. Goals themselves are
    never dead.

    This is a deliberately conservative subset of what Lakshya's full
    dead-square detector will do, but it is cheap enough to run on
    every state and is enough to make BFS-style search behave sanely on
    smoke-test levels.
    """

    reachable: set[Pos] = set(board.goals)
    frontier = deque(board.goals)
    while frontier:
        cell = frontier.popleft()
        for a in ACTIONS:
            dr, dc = DXDY[a]
            # To "pull" the box backwards: the box was at ``prev`` and
            # got pushed into ``cell``. For that push to be legal in
            # the single-box idealisation, the player has to stand at
            # ``prev - d`` (i.e. one cell further behind the box along
            # the push axis) and that cell has to be floor. Both
            # ``prev`` and the player cell must lie on floor.
            prev = (cell[0] - dr, cell[1] - dc)
            player_side = (prev[0] - dr, prev[1] - dc)
            if prev in reachable:
                continue
            if prev not in board.floor:
                continue
            if player_side not in board.floor:
                continue
            reachable.add(prev)
            frontier.append(prev)
    return frozenset(c for c in board.floor if c not in reachable)


_DEAD_SQUARE_CACHE: dict[Board, FrozenSet[Pos]] = {}


def dead_squares(board: Board) -> FrozenSet[Pos]:
    """Cached accessor for the static dead-square map of ``board``.

    Keyed by the (frozen, hashable) ``Board`` itself; we deliberately
    avoid ``id(board)`` because Python recycles ids across short-lived
    ``Board`` instances, which would cause the cache to return the
    wrong map for a new level that happens to land at a reused
    address.
    """
    cached = _DEAD_SQUARE_CACHE.get(board)
    if cached is None:
        cached = _simple_dead_squares(board)
        _DEAD_SQUARE_CACHE[board] = cached
    return cached


def is_simple_deadlock(board: Board, state: State) -> bool:
    """Cheap necessary-but-not-sufficient deadlock check.

    Returns ``True`` if any box currently sits on a static dead square
    (and is not already on a goal). Combined with the deeper deadlock
    checks Lakshya will add, this forms a tiered pruning chain: the
    cheap test runs on every expansion, the expensive ones only on
    states that pass it.
    """
    dead = dead_squares(board)
    for box in state.boxes:
        if box in dead and box not in board.goals:
            return True
    return False


# ---------------------------------------------------------------------------
# Solution-string serialisation (proposal Section 2 output convention)
# ---------------------------------------------------------------------------


def trace_to_string(steps: List[Tuple[Action, bool]]) -> str:
    """Encode a move-space trace as the proposal's solution string.

    Each step is ``(action, pushed_box)``: pushes use UPPERCASE letters
    and pure moves use lowercase letters, matching Section 2 of the
    proposal.
    """
    out: list[str] = []
    for action, pushed in steps:
        ch = ACTION_NAMES[action]
        out.append(ch if pushed else ch.lower())
    return "".join(out)


def replay_trace(
    board: Board, state: State, solution: str
) -> List[State]:
    """Replay a solution string against a starting state.

    Returns the list of intermediate states including the start. Raises
    ``ValueError`` if the trace is inconsistent with the board.
    """
    letter_to_action = {ACTION_NAMES[a]: a for a in ACTIONS}
    states = [state]
    cur = state
    for ch in solution:
        upper = ch.upper()
        if upper not in letter_to_action:
            raise ValueError(f"bad solution char {ch!r}")
        action = letter_to_action[upper]
        cur, pushed = apply_action(board, cur, action)
        # Sanity-check the case bit against the actual transition.
        if pushed != ch.isupper():
            raise ValueError(
                f"trace inconsistency: step {ch!r} push-case does not match "
                f"actual push={pushed}"
            )
        states.append(cur)
    return states


def push_path_to_moves(
    board: Board, start: State, pushes: List[Tuple[Pos, Action]]
) -> List[Tuple[Action, bool]]:
    """Expand a sequence of pushes back into a sequence of moves.

    A push solver returns a list of ``(box, direction)`` pairs. To
    animate or score it under the proposal's metrics we need to fill in
    the player's walk between pushes. We do that via BFS on the
    move graph between each push's "from" cell and the previous player
    position. Returns the full move trace including push steps tagged
    with ``pushed=True`` and walking steps tagged with ``pushed=False``.
    """
    moves: list[tuple[Action, bool]] = []
    cur = start
    for box, action in pushes:
        dr, dc = DXDY[action]
        push_from = (box[0] - dr, box[1] - dc)
        walk = _bfs_walk(board, cur, push_from)
        for step in walk:
            cur, pushed = apply_action(board, cur, step)
            assert not pushed
            moves.append((step, False))
        cur, pushed = apply_action(board, cur, action)
        assert pushed
        moves.append((action, True))
    return moves


def _bfs_walk(board: Board, state: State, target: Pos) -> List[Action]:
    """Shortest action sequence that walks the player to ``target``
    without pushing any box. Returns ``[]`` if already there.
    """
    if state.player == target:
        return []
    parents: dict[Pos, tuple[Pos, Action]] = {}
    queue = deque([state.player])
    seen = {state.player}
    while queue:
        p = queue.popleft()
        if p == target:
            break
        for a in ACTIONS:
            nxt = _step(p, a)
            if nxt in seen:
                continue
            if nxt not in board.floor:
                continue
            if nxt in state.boxes:
                continue
            seen.add(nxt)
            parents[nxt] = (p, a)
            queue.append(nxt)
    if target not in parents and state.player != target:
        raise ValueError(f"target {target} not reachable from {state.player}")
    path: list[Action] = []
    cur = target
    while cur != state.player:
        prev, action = parents[cur]
        path.append(action)
        cur = prev
    return list(reversed(path))
