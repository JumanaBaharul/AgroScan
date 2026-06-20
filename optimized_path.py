import numpy as np
import heapq
import random
import math
import time
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import label


# ══════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def path_distance(path):
    if len(path) < 2:
        return 0.0
    return sum(
        np.linalg.norm(np.array(path[i]) - np.array(path[i - 1]))
        for i in range(1, len(path))
    )


def disease_gain(path, prob_map):
    return sum(prob_map[y, x] for y, x in path)


def count_turns(path):
    turns = 0
    for i in range(2, len(path)):
        dy1 = path[i-1][0] - path[i-2][0]
        dx1 = path[i-1][1] - path[i-2][1]
        dy2 = path[i][0]   - path[i-1][0]
        dx2 = path[i][1]   - path[i-1][1]
        if (dy1, dx1) != (dy2, dx2):
            turns += 1
    return turns


def get_centroids(mask, min_size=40):
    labeled, num = label(mask)
    centroids = []
    for i in range(1, num + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) >= min_size:
            centroids.append((int(np.mean(ys)), int(np.mean(xs))))
    return centroids


# ══════════════════════════════════════════════════════════
# A* SEARCH (shared primitive)
# ══════════════════════════════════════════════════════════

def astar(cost_map, start, goal):
    H, W = cost_map.shape

    def h(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    pq     = [(0, start)]
    parent = {}
    g      = {start: 0}

    while pq:
        _, cur = heapq.heappop(pq)
        if cur == goal:
            path = []
            while cur in parent:
                path.append(cur)
                cur = parent[cur]
            path.append(start)
            return path[::-1]

        y, x = cur
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W:
                nb      = (ny, nx)
                new_g   = g[cur] + cost_map[ny, nx]
                if nb not in g or new_g < g[nb]:
                    g[nb]      = new_g
                    parent[nb] = cur
                    heapq.heappush(pq, (new_g + h(nb, goal), nb))
    return []


# ══════════════════════════════════════════════════════════
# ALGORITHMS
# ══════════════════════════════════════════════════════════

def run_astar(prob_map, spray_mask, start):
    """Baseline: visit centroids in discovery order."""
    cost_map  = 1 - normalize(prob_map)
    centroids = get_centroids(spray_mask)
    if not centroids:
        return []
    path, cur = [], start
    for node in centroids:
        path.extend(astar(cost_map, cur, node))
        cur = node
    return path


def run_mst(prob_map, spray_mask, start):
    """Prim's MST on centroids, then A* between consecutive nodes."""
    centroids = get_centroids(spray_mask)
    nodes     = [start] + centroids
    cost_map  = 1 - normalize(prob_map)

    visited, order = {0}, []
    while len(visited) < len(nodes):
        best, min_d = None, 1e9
        for u in visited:
            for v in range(len(nodes)):
                if v not in visited:
                    d = np.linalg.norm(np.array(nodes[u]) - np.array(nodes[v]))
                    if d < min_d:
                        min_d, best = d, v
        visited.add(best)
        order.append(nodes[best])

    path, cur = [], start
    for node in order:
        path.extend(astar(cost_map, cur, node))
        cur = node
    return path


def run_tsp(prob_map, spray_mask, start):
    """Nearest-neighbour TSP heuristic."""
    centroids = get_centroids(spray_mask)
    nodes     = [start] + centroids
    cost_map  = 1 - normalize(prob_map)

    unvisited = nodes.copy()
    route     = [unvisited.pop(0)]
    while unvisited:
        last = route[-1]
        nxt  = min(unvisited, key=lambda n: np.linalg.norm(np.array(last)-np.array(n)))
        route.append(nxt)
        unvisited.remove(nxt)

    path = []
    for i in range(len(route)-1):
        path.extend(astar(cost_map, route[i], route[i+1]))
    return path


def run_dijkstra(prob_map, spray_mask, start):
    """Dijkstra between centroids (same cost map as A*)."""
    cost_map  = 1 - normalize(prob_map)
    centroids = get_centroids(spray_mask)
    nodes     = [start] + centroids

    def dijkstra(cost_map, s, g):
        H, W   = cost_map.shape
        pq     = [(0, s)]
        dist   = {s: 0}
        parent = {}
        while pq:
            cd, cur = heapq.heappop(pq)
            if cur == g:
                path = []
                while cur in parent:
                    path.append(cur)
                    cur = parent[cur]
                path.append(s)
                return path[::-1]
            y, x = cur
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y+dy, x+dx
                if 0 <= ny < H and 0 <= nx < W:
                    nb    = (ny, nx)
                    nc    = cd + cost_map[ny, nx]
                    if nb not in dist or nc < dist[nb]:
                        dist[nb]   = nc
                        parent[nb] = cur
                        heapq.heappush(pq, (nc, nb))
        return []

    path, cur = [], start
    for node in nodes[1:]:
        path.extend(dijkstra(cost_map, cur, node))
        cur = node
    return path


def run_sa(prob_map, spray_mask, start):
    """Simulated Annealing for route order, then A* for pixel path."""
    centroids = get_centroids(spray_mask)
    nodes     = [start] + centroids

    def route_len(r):
        return sum(np.linalg.norm(np.array(r[i])-np.array(r[i-1]))
                   for i in range(1, len(r)))

    current = nodes.copy()
    best    = current.copy()
    temp    = 1000.0
    cooling = 0.995

    while temp > 1 and len(nodes) > 2:
        i, j = sorted(random.sample(range(1, len(nodes)), 2))
        new = current[:]
        new[i:j] = reversed(new[i:j])
        if (route_len(new) < route_len(current) or
                random.random() < math.exp((route_len(current)-route_len(new))/temp)):
            current = new
            if route_len(current) < route_len(best):
                best = current
        temp *= cooling

    cost_map = 1 - normalize(prob_map)
    path = []
    for i in range(len(best)-1):
        path.extend(astar(cost_map, best[i], best[i+1]))
    return path


# ══════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════

def plot_all_algorithms(prob_map, spray_mask, paths):
    plt.figure(figsize=(18, 10))
    for i, (name, path) in enumerate(paths.items()):
        plt.subplot(2, 3, i+1)
        plt.imshow(prob_map, cmap="Reds")
        plt.imshow(spray_mask.astype(np.uint8), cmap="gray", alpha=0.3)
        if path:
            ys = [p[0] for p in path]
            xs = [p[1] for p in path]
            plt.plot(xs, ys, color="cyan", linewidth=1.5)
            plt.scatter(xs[0], ys[0], color="blue", s=80, zorder=5)
        plt.title(name, fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════
# COMPARISON ENTRY POINT
# ══════════════════════════════════════════════════════════

def compare_algorithms(prob_map, spray_mask, start=(0, 0)):
    algorithms = {
        "Direct A*":          run_astar,
        "MST + A*":           run_mst,
        "TSP":                run_tsp,
        "Dijkstra":           run_dijkstra,
        "Simulated Annealing": run_sa,
    }

    results, paths = [], {}

    for name, func in algorithms.items():
        t0   = time.time()
        path = func(prob_map, spray_mask, start)
        rt   = time.time() - t0

        dist     = path_distance(path)
        gain     = disease_gain(path, prob_map)
        eff      = gain / (dist + 1e-6)
        turns    = count_turns(path)

        results.append({
            "Algorithm":   name,
            "Distance":    round(dist, 2),
            "Gain":        round(gain, 2),
            "Efficiency":  round(eff, 4),
            "Turns":       turns,
            "Runtime (s)": round(rt, 4),
        })
        paths[name] = path

    df = pd.DataFrame(results).sort_values("Efficiency", ascending=False)
    print("\n===== ALGORITHM COMPARISON =====\n")
    print(df.to_string(index=False))

    plot_all_algorithms(prob_map, spray_mask, paths)

    return df, paths
