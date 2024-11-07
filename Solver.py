import tkinter as tk
from tkinter import messagebox, filedialog
import random
import heapq
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import json
import sys


BG_COLOR = "#1DB954"  
DARK_BG = "#121212"   
TEXT_COLOR = "white"


DEFAULT_WIDTH, DEFAULT_HEIGHT = 20, 20
WALL, PATH, START, END = '#', ' ', 'S', 'E'

ALGO_EXPLANATIONS = {
    "DFS": "Depth-First Search (DFS) explores paths until it hits a wall, then backtracks. It creates single-solution mazes.",
    "Prim": "Prim's algorithm adds walls to a frontier and selects paths that connect parts of the maze, creating open mazes.",
    "Kruskal": "Kruskal's algorithm generates mazes by treating each cell as an individual set and merging cells randomly until the maze is complete.",
    "BFS": "Breadth-First Search (BFS) is a simple algorithm that explores each level of the maze evenly. It finds the shortest path but can be slower than A*."
}

class Maze:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.maze = [[WALL for _ in range(width)] for _ in range(height)]
        self.start = (1, 1)
        self.end = (height - 2, width - 2)

    def generate_maze_dfs(self):
        def carve(x, y):
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
            random.shuffle(directions)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 1 <= nx < self.height - 1 and 1 <= ny < self.width - 1:
                    if self.maze[nx][ny] == WALL:
                        self.maze[nx][ny] = PATH
                        self.maze[x + dx // 2][y + dy // 2] = PATH
                        if app.visualize:
                            self._visualize()
                        carve(nx, ny)

        self.maze[1][1] = PATH
        carve(1, 1)
        self.maze[self.start[0]][self.start[1]] = START
        self.maze[self.end[0]][self.end[1]] = END

    def generate_maze_prim(self):
        self.maze[1][1] = PATH
        walls = [(1, 1)]
        
        while walls:
            x, y = walls.pop(random.randint(0, len(walls) - 1))
            if self.maze[x][y] == PATH:
                neighbors = [(x + dx, y + dy) for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]]
                valid_neighbors = [(nx, ny) for nx, ny in neighbors if 1 <= nx < self.height - 1 and 1 <= ny < self.width - 1]
                random.shuffle(valid_neighbors)

                for nx, ny in valid_neighbors:
                    if self.maze[nx][ny] == WALL:
                        self.maze[(x + nx) // 2][(y + ny) // 2] = PATH
                        self.maze[nx][ny] = PATH
                        walls.extend([(nx + dx, ny + dy) for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)] if (0 < nx + dx < self.height - 1 and 0 < ny + dy < self.width - 1)])
                        if app.visualize:
                            self._visualize()
                        break

        self.maze[self.start[0]][self.start[1]] = START
        self.maze[self.end[0]][self.end[1]] = END

    def generate_maze_kruskal(self):
        sets = {(x, y): (x, y) for x in range(1, self.height, 2) for y in range(1, self.width, 2)}
        edges = [(x, y, nx, ny) for x in range(1, self.height, 2) for y in range(1, self.width, 2)
                 for nx, ny in [(x + 2, y), (x, y + 2)] if nx < self.height and ny < self.width]
        random.shuffle(edges)

        def find(cell):
            if sets[cell] != cell:
                sets[cell] = find(sets[cell])
            return sets[cell]

        def union(cell1, cell2):
            root1, root2 = find(cell1), find(cell2)
            if root1 != root2:
                sets[root2] = root1
                return True
            return False

        for x, y, nx, ny in edges:
            if union((x, y), (nx, ny)):
                self.maze[x][y] = PATH
                self.maze[(x + nx) // 2][(y + ny) // 2] = PATH
                self.maze[nx][ny] = PATH
                if app.visualize:
                    self._visualize()

        self.maze[self.start[0]][self.start[1]] = START
        self.maze[self.end[0]][self.end[1]] = END

    def _visualize(self):
        cmap = colors.ListedColormap(['black', 'white', 'red', 'green'])
        maze_numeric = [[0 if cell == WALL else 1 for cell in row] for row in self.maze]
        maze_numeric[self.start[0]][self.start[1]] = 2  
        maze_numeric[self.end[0]][self.end[1]] = 3     
        plt.imshow(maze_numeric, cmap=cmap)
        plt.pause(0.01)

    def is_solvable(self, solver):
        return solver.solve() is not None

    def display(self):
        plt.show()


class MazeSolver:
    def __init__(self, maze):
        self.maze = maze
        self.start = maze.start
        self.end = maze.end

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def solve(self, algorithm="A*"):
        if algorithm == "BFS":
            return self.solve_bfs()
        return self.solve_a_star()

    def solve_a_star(self):
        width, height = self.maze.width, self.maze.height
        open_set = []
        heapq.heappush(open_set, (0, self.start))
        came_from = {}
        g_score = {self.start: 0}
        f_score = {self.start: self.heuristic(self.start, self.end)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == self.end:
                return self.reconstruct_path(came_from)

            x, y = current
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (x + dx, y + dy)
                if 0 <= neighbor[0] < height and 0 <= neighbor[1] < width:
                    if self.maze.maze[neighbor[0]][neighbor[1]] != WALL:
                        tentative_g_score = g_score[current] + 1
                        if tentative_g_score < g_score.get(neighbor, float('inf')):
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g_score
                            f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, self.end)
                            if neighbor not in [i[1] for i in open_set]:
                                heapq.heappush(open_set, (f_score[neighbor], neighbor))
            if app.visualize:
                self._visualize_step(g_score)
        return None

    def solve_bfs(self):
        from collections import deque
        queue = deque([self.start])
        came_from = {self.start: None}
        while queue:
            current = queue.popleft()
            if current == self.end:
                return self.reconstruct_path(came_from)

            x, y = current
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (x + dx, y + dy)
                if 0 <= neighbor[0] < self.maze.height and 0 <= neighbor[1] < self.maze.width:
                    if self.maze.maze[neighbor[0]][neighbor[1]] != WALL and neighbor not in came_from:
                        queue.append(neighbor)
                        came_from[neighbor] = current
            if app.visualize:
                self._visualize_step(came_from)
        return None

    def _visualize_step(self, visited_cells):
        plt.clf()  
        cmap = colors.ListedColormap(['black', 'white', 'red', 'green', 'blue'])
        maze_numeric = [[0 if cell == WALL else 1 for cell in row] for row in self.maze.maze]
        for cell in visited_cells:
            maze_numeric[cell[0]][cell[1]] = 4  
        maze_numeric[self.start[0]][self.start[1]] = 2  
        maze_numeric[self.end[0]][self.end[1]] = 3      
        plt.imshow(maze_numeric, cmap=cmap)
        plt.pause(0.05)

    def reconstruct_path(self, came_from):
        current = self.end
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def display_final_path(self, path):
        plt.ioff()
        cmap = colors.ListedColormap(['black', 'white', 'red', 'green', 'blue', 'yellow'])
        maze_numeric = [[0 if cell == WALL else 1 for cell in row] for row in self.maze.maze]
        for x, y in path:
            maze_numeric[x][y] = 5
        maze_numeric[self.start[0]][self.start[1]] = 2
        maze_numeric[self.end[0]][self.end[1]] = 3
        plt.imshow(maze_numeric, cmap=cmap)
        plt.title("Final Path in Green Line")
        plt.show()

class MazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Maze Solver")
        self.root.geometry("400x500")
        self.root.configure(bg=DARK_BG)
        self.root.iconbitmap(default="icon.ico")
        
        self.algorithm = tk.StringVar(value="DFS")
        self.width_var = tk.StringVar(value=str(DEFAULT_WIDTH))
        self.height_var = tk.StringVar(value=str(DEFAULT_HEIGHT))
        self.solver_algorithm = tk.StringVar(value="A*")
        self.visualize = tk.BooleanVar(value=True)

        tk.Label(root, text="Maze Generator Algorithm", bg=DARK_BG, fg=TEXT_COLOR).pack(pady=5)
        tk.OptionMenu(root, self.algorithm, "DFS", "Prim", "Kruskal").pack()
        
        tk.Label(root, text="Maze Width", bg=DARK_BG, fg=TEXT_COLOR).pack(pady=5)
        tk.Entry(root, textvariable=self.width_var, width=5).pack()

        tk.Label(root, text="Maze Height", bg=DARK_BG, fg=TEXT_COLOR).pack(pady=5)
        tk.Entry(root, textvariable=self.height_var, width=5).pack()

        tk.Label(root, text="Solver Algorithm", bg=DARK_BG, fg=TEXT_COLOR).pack(pady=5)
        tk.OptionMenu(root, self.solver_algorithm, "A*", "BFS").pack()

        tk.Checkbutton(root, text="Visualize Solution", variable=self.visualize, bg=DARK_BG, fg=TEXT_COLOR).pack(pady=5)

        tk.Button(root, text="Generate Maze", command=self.generate_maze, bg=BG_COLOR, fg=TEXT_COLOR).pack(pady=10)
        tk.Button(root, text="Solve Maze", command=self.solve_maze, bg=BG_COLOR, fg=TEXT_COLOR).pack(pady=5)
        tk.Button(root, text="Restart", command=self.restart_maze, bg=BG_COLOR, fg=TEXT_COLOR).pack(pady=5)
        tk.Button(root, text="Help", command=self.show_help, bg=BG_COLOR, fg=TEXT_COLOR).pack(pady=5)

        self.maze = None
        self.solver = None

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        plt.close("all")
        self.root.destroy()
        sys.exit()

    def generate_maze(self):
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
            algorithm = self.algorithm.get()

            self.maze = Maze(width, height)
            if algorithm == "DFS":
                self.maze.generate_maze_dfs()
            elif algorithm == "Prim":
                self.maze.generate_maze_prim()
            elif algorithm == "Kruskal":
                self.maze.generate_maze_kruskal()

            self.solver = MazeSolver(self.maze)
            while not self.maze.is_solvable(self.solver):
                self.maze = Maze(width, height)
                if algorithm == "DFS":
                    self.maze.generate_maze_dfs()
                elif algorithm == "Prim":
                    self.maze.generate_maze_prim()
                elif algorithm == "Kruskal":
                    self.maze.generate_maze_kruskal()

            self.maze.display()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid integer dimensions.")

    def solve_maze(self):
        if not self.maze:
            messagebox.showwarning("Warning", "Please generate a maze first!")
            return

        algorithm = self.solver_algorithm.get()
        solution_path = self.solver.solve(algorithm=algorithm)
        if solution_path:
            self.solver.display_final_path(solution_path)
        else:
            messagebox.showwarning("Warning", "No solution found.")

    def restart_maze(self):
        plt.close("all")  
        self.maze = None
        self.solver = None
        messagebox.showinfo("Restarted", "The maze has been reset. Generate a new maze to start.")

    def show_help(self):
        algorithm = self.algorithm.get()
        explanation = ALGO_EXPLANATIONS.get(algorithm, "No information available for this algorithm.")
        messagebox.showinfo("Algorithm Explanation", explanation)


root = tk.Tk()
app = MazeApp(root)
root.mainloop()
