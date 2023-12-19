import math
import heapq

class MapSolver:
    def __init__(self, map_str):
        self.map_arr = [list(row.strip()) for row in map_str.strip().split("\n")]
        self.start_node = (0, 0)
        self.end_node = (len(self.map_arr) - 1, len(self.map_arr[0]) - 1)

    def get_neighbors(self, node):
        x, y = node
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= len(self.map_arr) or ny >= len(self.map_arr[0]):
                    continue
                if self.map_arr[nx][ny] == "X":
                    continue
                if dx != 0 and dy != 0:
                    cost = math.sqrt(2)
                else:
                    cost = 1
                neighbors.append(((nx, ny), cost))
        return neighbors

    def get_path(self):
        visited = set()
        queue = [(0, self.start_node, [])]
        while queue:
            cost, node, path = heapq.heappop(queue)
            if node == self.end_node:
                return path + [node], cost
            if node in visited:
                continue
            visited.add(node)
            for neighbor, neighbor_cost in self.get_neighbors(node):
                if neighbor in visited:
                    continue
                heapq.heappush(queue, (cost + neighbor_cost, neighbor, path + [node]))
        return None, None

    def solve(self):
        shortest_path, cost = self.get_path()
        if shortest_path:
            for i, node in enumerate(shortest_path):
                self.map_arr[node[0]][node[1]] = chr(ord("A") + i)
            print("Shortest path:")
            for row in self.map_arr:
                print(" ".join(row))
            print(f"Cost: {cost}")
        else:
            print("No shortest path found.")

# Example usage:
map_str = """
..X..
..X..
..X..
..X..
.....
"""
solver = MapSolver(map_str)
solver.solve()
