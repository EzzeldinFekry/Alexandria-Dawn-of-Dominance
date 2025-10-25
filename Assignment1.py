# 8-PUZZLE SOLVER — FULLY COMMENTED AND EXPLAINED
# ------------------------------------------------
# This program solves the classic 8-puzzle problem using
# BFS, DFS, IDDFS, and A* search (with Manhattan and Euclidean heuristics).

import time, math, heapq, itertools
from collections import deque

# ------------------------------------------------
# Goal state definition
# 0 represents the blank tile.
Goal = (0, 1, 2, 3, 4, 5, 6, 7, 8)

# ------------------------------------------------
# Function to display a 3×3 puzzle board nicely
def format_board(state):
    # Replace 0 (blank) with a space for readability
    s = [" " if x == 0 else str(x) for x in state]
    # Make rows of 3 tiles and join them with lines
    return "\n".join(["| " + "  ".join(s[i:i+3]) + " |" for i in range(0, 9, 3)])

# ------------------------------------------------
# Node class represents a single state of the puzzle
class Node:
    def __init__(self, state, parent=None, action=None, g=0, depth=0):
        self.state = state      # Current board configuration
        self.parent = parent    # Parent node (previous state)
        self.action = action    # Move made to reach this node ("Up", "Down", etc.)
        self.g = g              # Cost from the start node (used in A*)
        self.depth = depth      # Depth level from the root

# ------------------------------------------------
# Function to check if a puzzle is solvable
def is_solvable(state):
    # Exclude 0 because blank tile doesn’t count for inversions
    s = [x for x in state if x != 0]
    # Count number of inversions (pairs of tiles out of order)
    inv = sum(1 for i in range(len(s)) for j in range(i + 1, len(s)) if s[i] > s[j])
    # A puzzle is solvable only if number of inversions is even
    return inv % 2 == 0

# ------------------------------------------------
# Helper function to swap two tiles (used when moving the blank)
def swap(s, i, j):
    s = s.copy()        # Make a copy so we don’t modify the original list
    s[i], s[j] = s[j], s[i]  # Swap the two positions
    return s

# ------------------------------------------------
# Generate all possible next moves from the current state
def successors(state):
    s = list(state)
    i = s.index(0)             # Find where the blank tile (0) is
    r, c = divmod(i, 3)        # Convert index into (row, column)
    moves = []

    # Try moving the blank up, down, left, right (if allowed)
    if r > 0:  # Can move up
        moves.append((tuple(swap(s, i, i - 3)), "Up"))
    if r < 2:  # Can move down
        moves.append((tuple(swap(s, i, i + 3)), "Down"))
    if c > 0:  # Can move left
        moves.append((tuple(swap(s, i, i - 1)), "Left"))
    if c < 2:  # Can move right
        moves.append((tuple(swap(s, i, i + 1)), "Right"))

    return moves  # Return list of (new_state, move_name)

# ------------------------------------------------
# Breadth-First Search (BFS)
# Explores all nodes at the current depth before moving deeper
def bfs(initial):
    start = Node(initial)               # Create the start node
    frontier = deque([start])           # Queue for BFS
    frontier_states = {initial}         # Track which states are already in the frontier
    explored = set()                    # Set of explored (visited) states
    nodes = 0                           # Counter for expanded nodes

    while frontier:                     # Continue until queue is empty
        node = frontier.popleft()       # Remove the front node
        frontier_states.remove(node.state)
        nodes += 1                      # Count node expansion

        if node.state == Goal:          # Goal test
            return node, nodes

        # Generate next possible moves
        for (state, action) in successors(node.state):
            if state not in explored and state not in frontier_states:
                child = Node(state, node, action, depth=node.depth + 1)
                frontier.append(child)
                frontier_states.add(state)
        explored.add(node.state)

    return None, nodes                  # If no solution found

# ------------------------------------------------
# Depth-First Search (DFS)
# Explores as far as possible down one branch before backtracking
def dfs(initial, max_nodes=100000):
    stack = [Node(initial)]             # Stack for DFS (LIFO)
    stack_states = {initial}
    explored = set()
    nodes = 0

    while stack and nodes < max_nodes:
        node = stack.pop()              # Take the last inserted node
        stack_states.remove(node.state)
        nodes += 1

        if node.state == Goal:          # Goal test
            return node, nodes

        explored.add(node.state)
        # Expand children and push them onto the stack
        for (state, action) in successors(node.state):
            if state not in explored and state not in stack_states:
                child = Node(state, node, action, depth=node.depth + 1)
                stack.append(child)
                stack_states.add(state)

    return None, nodes

# ------------------------------------------------
# Iterative Deepening DFS (IDDFS)
# Runs DFS with increasing depth limits until the goal is found
def iddfs(initial, limit=20):
    def dls(state, parent_map, depth):
        # Depth limit check
        if depth > limit:
            return None
        # Goal test
        if state == Goal:
            return state
        # Expand children recursively
        for (succ, _) in successors(state):
            if succ not in parent_map:
                parent_map[succ] = state
                found = dls(succ, parent_map, depth + 1)
                if found:
                    return found
        return None

    parent_map = {initial: None}
    return dls(initial, parent_map, 0)

# ------------------------------------------------
# Precompute goal positions for heuristic calculations
goal_pos = {Goal[i]: (i // 3, i % 3) for i in range(9)}

# Manhattan distance heuristic: sum of horizontal and vertical distances
def manhattan(s):
    return sum(abs(i // 3 - goal_pos[v][0]) + abs(i % 3 - goal_pos[v][1])
               for i, v in enumerate(s) if v != 0)

# Euclidean distance heuristic: straight-line distance
def euclidean(s):
    return sum(math.sqrt((i // 3 - goal_pos[v][0]) ** 2 +
                         (i % 3 - goal_pos[v][1]) ** 2)
               for i, v in enumerate(s) if v != 0)

# ------------------------------------------------
# A* Search Algorithm
# Combines cost so far (g) and estimated cost to goal (h)
def a_star(initial, heuristic):
    counter = itertools.count()     # Counter to break ties in heap
    open_heap = []                  # Priority queue (min-heap)
    start = Node(initial)
    # Push the start node with f = g + h
    heapq.heappush(open_heap, (heuristic(initial), next(counter), start))
    closed = set()                  # Visited states
    nodes = 0

    while open_heap:
        _, _, node = heapq.heappop(open_heap)  # Pop node with lowest f-score
        nodes += 1

        if node.state == Goal:      # Goal found
            return node, nodes

        closed.add(node.state)
        # Expand successors
        for (state, action) in successors(node.state):
            if state in closed:
                continue
            child = Node(state, node, action, g=node.g + 1)
            f = child.g + heuristic(state)
            heapq.heappush(open_heap, (f, next(counter), child))

    return None, nodes

# ------------------------------------------------
# Print the final results and (optionally) the path of boards
def print_result(name, result, show_path_boards=True, max_boards=50):
    node, nodes = result
    path = []
    # Reconstruct path by following parent pointers
    while node:
        path.append(node)
        node = node.parent
    path.reverse()  # Reverse so it goes from start → goal

    print(f"\n{name} Results:")
    print(f"Path length (states): {len(path)}")
    print(f"Nodes expanded: {nodes}")
    print(f"Depth: {len(path) - 1}")

    # Display the sequence of board states
    if show_path_boards:
        for i, n in enumerate(path[:max_boards]):
            print(f"\nStep {i}:\n{format_board(n.state)}")

# ------------------------------------------------
# Example usage
if __name__ == "__main__":
    # Define an initial puzzle state (0 = blank)
    initial = (1, 2, 3, 4, 0, 5, 6, 7, 8)

    # Check solvability before running algorithms
    if not is_solvable(initial):
        print("This puzzle configuration is unsolvable.")
    else:
        print("Initial board:")
        print(format_board(initial))

        start_time = time.time()
        result = bfs(initial)
        print_result("BFS", result)
        print(f"Time: {time.time() - start_time:.2f} sec")

        start_time = time.time()
        result = dfs(initial)
        print_result("DFS", result)
        print(f"Time: {time.time() - start_time:.2f} sec")

        start_time = time.time()
        result = a_star(initial, manhattan)
        print_result("A* (Manhattan)", result)
        print(f"Time: {time.time() - start_time:.2f} sec")