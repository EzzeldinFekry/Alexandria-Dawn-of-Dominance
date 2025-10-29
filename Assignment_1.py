import time, math, heapq
import itertools
from collections import deque

Goal = (0,1,2,3,4,5,6,7,8)  # goal configuration

# Format board for readable printing
def format_board(state):
    s = [" " if x==0 else str(x) for x in state]
    return "\n".join(["| " + "  ".join(s[i:i+3]) + " |" for i in range(0,9,3)])

# Node structure for search tree
class Node:
    def __init__(self, state, parent=None, action=None, g=0, depth=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.g = g
        self.depth = depth

# Trace back the solution path from goal to start
def extract_path(node):
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return path[::-1]

# Generate all valid moves from the current state (order: Up, Down, Left, Right)
def successors(state):
    s = list(state)
    i = s.index(0)  # locate blank tile
    r, c = divmod(i, 3)
    moves = []

    if r > 0:
        new = s.copy(); new[i], new[i-3] = new[i-3], new[i]; moves.append((tuple(new), "Up"))
    if r < 2:
        new = s.copy(); new[i], new[i+3] = new[i+3], new[i]; moves.append((tuple(new), "Down"))
    if c > 0:
        new = s.copy(); new[i], new[i-1] = new[i-1], new[i]; moves.append((tuple(new), "Left"))
    if c < 2:
        new = s.copy(); new[i], new[i+1] = new[i+1], new[i]; moves.append((tuple(new), "Right"))

    return moves

# Check if puzzle can be solved by counting inversions
def is_solvable(state):
    s = [x for x in state if x != 0]
    inv = sum(1 for i in range(len(s)) for j in range(i+1, len(s)) if s[i] > s[j])
    return inv % 2 == 0  # even → solvable

# BFS explores all nodes level by level
def bfs(initial):
    start = Node(initial)
    frontier = deque([start])
    frontier_states = {initial}
    explored = set()
    nodes = 0
    start_time = time.perf_counter()

    while frontier:
        node = frontier.popleft()
        frontier_states.remove(node.state)
        nodes += 1
        explored.add(node.state)

        if node.state == Goal:
            return extract_path(node), nodes, node.depth, time.perf_counter() - start_time

        for s, _ in successors(node.state):
            if s not in explored and s not in frontier_states:
                child = Node(s, node, g=node.g+1, depth=node.depth+1)
                frontier.append(child)
                frontier_states.add(s)
    return None

# DFS explores one branch deeply before backtracking
def dfs(initial, max_nodes=100000, max_trace_states=2000):
    start = Node(initial)
    stack = [start]
    stack_states = {initial}
    explored = set()
    nodes = 0
    start_time = time.perf_counter()

    while stack and nodes < max_nodes:
        node = stack.pop()
        stack_states.discard(node.state)
        nodes += 1

        if node.state == Goal:
            return extract_path(node), nodes, node.depth, time.perf_counter() - start_time

        if node.state in explored:
            continue
        explored.add(node.state)

        # push successors in normal order but reversed because stack LIFO
        for s, _ in reversed(successors(node.state)):
            if s not in explored and s not in stack_states:
                child = Node(s, node, g=node.g+1, depth=node.depth+1)
                stack.append(child)
                stack_states.add(s)

    # If reached here: no solution within node limit
    return None

# IDDFS repeats DFS with increasing depth limit (returns path similar to others)
def iddfs(initial, limit=20):
    def dls(state, parent_map, depth):
        if state == Goal:
            return True
        if depth == 0:
            return False
        for s, _ in successors(state):
            if s not in parent_map:  # avoid cycles in DLS search tree
                parent_map[s] = state
                if dls(s, parent_map, depth - 1):
                    return True
        return False

    start_time = time.perf_counter()
    for d in range(limit + 1):
        parent_map = {initial: None}
        found = dls(initial, parent_map, d)
        if found:
            # reconstruct path:
            path = []
            cur = Goal
            while cur is not None:
                path.append(cur)
                cur = parent_map[cur]
            path = path[::-1]
            return path, len(path)-1, d, time.perf_counter() - start_time
    return None

# Compute heuristics for A*
goal_pos = {Goal[i]: (i // 3, i % 3) for i in range(9)}

def manhattan(s):
    return sum(abs(i//3 - goal_pos[v][0]) + abs(i%3 - goal_pos[v][1]) for i, v in enumerate(s) if v != 0)

def euclidean(s):
    return sum(math.sqrt((i//3 - goal_pos[v][0])**2 + (i%3 - goal_pos[v][1])**2) for i, v in enumerate(s) if v != 0)

# A* search algorithm (with tie-breaker counter + g_score to avoid duplicates)
def a_star(initial, heuristic):
    start = Node(initial)
    counter = itertools.count()  # unique sequence count to break ties in heapq
    # heap entries: (f_score, count, node)
    open_heap = []
    heapq.heappush(open_heap, (heuristic(initial), next(counter), start))
    g_score = {initial: 0}
    closed = set()
    nodes = 0
    start_time = time.perf_counter()

    while open_heap:
        _, _, node = heapq.heappop(open_heap)
        nodes += 1

        if node.state == Goal:
            return extract_path(node), nodes, node.depth, time.perf_counter() - start_time

        if node.state in closed:
            continue
        closed.add(node.state)

        for s, _ in successors(node.state):
            tentative_g = node.g + 1
            if s in closed and tentative_g >= g_score.get(s, float('inf')):
                continue
            # if this path to s is better than any previous one
            if tentative_g < g_score.get(s, float('inf')):
                g_score[s] = tentative_g
                child = Node(s, node, g=tentative_g, depth=node.depth+1)
                f = tentative_g + heuristic(s)
                heapq.heappush(open_heap, (f, next(counter), child))

    return None

# Utility to nicely print algorithm result
def print_result(name, result, show_path_boards=True, max_boards=50):
    print(f"\n{name} Result:")
    if not result:
        print("  No solution found or limit reached.")
        return
    path, nodes, depth, duration = result
    print(f"  Path length (states): {len(path)}")
    print(f"  Nodes expanded: {nodes}")
    print(f"  Depth: {depth}")
    print(f"  Time (s): {duration:.6f}")

    if show_path_boards:
        print("\n  Traceable Solution Path:")
        for i, state in enumerate(path):
            if i >= max_boards:
                print("   ... (path too long to display further; increase max_boards to show more)")
                break
            print(f"\n  Step {i}:")
            print("  " + format_board(state).replace("\n", "\n  "))

# ---------- Example usage ----------
if __name__ == "__main__":
    # Default initial (you can change this)
    initial = (7, 0, 2, 8, 5, 3, 6, 1, 4)

    print("Initial state:")
    print(format_board(initial))
    print()

    if not is_solvable(initial):
        print("Unsolvable Puzzle")
    else:
        print("Solvable Puzzle\n")

        # BFS
        res_bfs = bfs(initial)
        print_result("BFS", res_bfs)

        # DFS (with a node limit to avoid runaway) - adjust max_nodes as needed
        res_dfs = dfs(initial, max_nodes=200000)
        print_result("DFS", res_dfs, max_boards=80)

        # IDDFS (limit can be increased if required)
        res_iddfs = iddfs(initial, limit=20)
        print_result("IDDFS", res_iddfs)

        # A* Manhattan
        res_astar_man = a_star(initial, manhattan)
        print_result("A* (Manhattan)", res_astar_man)

        # A* Euclidean
        res_astar_euc = a_star(initial, euclidean)
        print_result("A* (Euclidean)", res_astar_euc)
