"""Shortest path over an *implicit* graph -- you give the neighbours, it finds the path.

The graph is never built. You pass a ``neighbours(node)`` oracle and the search
expands it lazily, so the graph may be huge or infinite and defined by anything: a
kd-tree range/kNN query, IK feasibility, a simulator step, an on-the-fly product of
factors. The only contract is that oracle.

Search is A* -- Dijkstra when no ``heuristic`` is given. A* pops the goal optimally
when the heuristic is admissible *and consistent* (monotone); the joint-space
distances used here are, so a closed set is safe.

    from graph import shortest_path
    path, cost = shortest_path(neighbours, start, goal)
"""

import heapq
import itertools


def shortest_path(neighbours, start, goal, *, heuristic=None):
    """Least-cost path from ``start`` to ``goal`` over an implicit graph.

    Args:
        neighbours: ``node -> iterable`` of successors. Each item is either a bare
            ``neighbour`` (weight defaults to 1.0) or a ``(neighbour, weight)`` pair
            with ``weight >= 0``. Nodes may be any hashable.
        start: the source node.
        goal: the target node, or a predicate ``node -> bool`` to reach any node in
            a set.
        heuristic: optional ``node -> float`` lower bound on the remaining cost to
            the goal. ``None`` (the default) makes this Dijkstra. Must be admissible
            and consistent for A* to stay optimal.

    Returns:
        ``(path, cost)`` -- the list of nodes from ``start`` to the reached goal and
        its total cost, or ``(None, inf)`` if the goal is unreachable.
    """
    is_goal = goal if callable(goal) else (lambda n: n == goal)
    h = heuristic if heuristic is not None else (lambda n: 0.0)

    g = {start: 0.0}                       # best known cost from start to node
    pred = {start: None}                   # predecessor on that best path
    counter = itertools.count()            # tie-breaker: never compare nodes
    heap = [(h(start), next(counter), start)]
    closed = set()

    while heap:
        _, _, node = heapq.heappop(heap)
        if node in closed:                 # a stale, worse entry
            continue
        if is_goal(node):
            return _reconstruct(pred, node), g[node]
        closed.add(node)

        for item in neighbours(node):
            nb, w = item if isinstance(item, tuple) else (item, 1.0)
            if w < 0:
                raise ValueError("negative edge weight %r (%s -> %s)" % (w, node, nb))
            cost = g[node] + w
            if nb not in g or cost < g[nb]:
                g[nb] = cost
                pred[nb] = node
                heapq.heappush(heap, (cost + h(nb), next(counter), nb))

    return None, float("inf")


def _reconstruct(pred, node):
    """Walk predecessors back to the source; return the path source-first."""
    path = []
    while node is not None:
        path.append(node)
        node = pred[node]
    return path[::-1]


if __name__ == "__main__":
    # Smoke test: a 4-connected grid, shortest path corner to corner. With the
    # Manhattan heuristic A* must return a monotone staircase of length 2*(W-1).
    W = 25

    def grid_neighbours(node):
        x, y = node
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < W:
                yield (nx, ny), 1.0

    def manhattan(node):
        return abs(node[0] - (W - 1)) + abs(node[1] - (W - 1))

    path, cost = shortest_path(grid_neighbours, (0, 0), (W - 1, W - 1),
                               heuristic=manhattan)
    assert cost == 2 * (W - 1), cost
    assert path[0] == (0, 0) and path[-1] == (W - 1, W - 1)
    print("ok: %d-node path, cost %.0f" % (len(path), cost))
