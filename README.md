##8-Puzzle Solver

A Python implementation of various search algorithms to solve the classic 8-puzzle problem.

Overview

This program solves the 8-puzzle (3x3 sliding tile puzzle) using four different search algorithms:

· BFS (Breadth-First Search)
· DFS (Depth-First Search)
· IDDFS (Iterative Deepening DFS)
· A* (with Manhattan and Euclidean heuristics)

Features

· Multiple Algorithms: Compare different search strategies
· Solvability Check: Automatically detects if a puzzle is solvable
· Visualization: Displays the puzzle board and solution path
· Performance Metrics: Tracks nodes expanded, path length, and execution time

How to Use

1. Set your initial puzzle state in the initial variable (0 represents the blank tile)
2. Run the program: python puzzle_solver.py
3. View results for each algorithm including solution path and performance statistics

Algorithm Details

· BFS: Guarantees optimal solution, explores level by level
· DFS: Memory efficient but may not find optimal solution
· IDDFS: Combines DFS space efficiency with BFS optimality
· A*: Most efficient with admissible heuristics (Manhattan/Euclidean distance)

Example

```python
initial = (1, 2, 3, 4, 0, 5, 6, 7, 8)  # 0 is the blank space
```

The program will solve this configuration and display the step-by-step solution for each algorithm.

Requirements

· Python 3.x
· No external dependencies

The code is fully commented and includes helper functions for board formatting, move generation, and solution path reconstruction.