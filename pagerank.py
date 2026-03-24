from __future__ import annotations

import argparse
import math
from collections import deque, defaultdict
from pathlib import Path


BETA = 0.85
ITERATIONS = 10
DEFAULT_DATASET = "web-Google_10k.txt"
VALID_DATASETS = {"web-Google_10k.txt", "web-Google.txt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PageRank for the Google web graph datasets."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=DEFAULT_DATASET,
        help="Input dataset file name or path.",
    )
    return parser.parse_args()


def resolve_input_path(raw_input: str) -> Path:
    candidate = Path(raw_input)
    if candidate.is_file():
        return candidate

    dataset_name = candidate.name
    if dataset_name not in VALID_DATASETS:
        valid = ", ".join(sorted(VALID_DATASETS))
        raise ValueError(f"Unsupported dataset '{raw_input}'. Choose one of: {valid}")

    dataset_path = Path("dataset") / dataset_name
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    return dataset_path


"""
Output: 
- Adjacency List L, where L[i] represesnts the set of nodes that link to i (incoming nodes)
- Degree D, where D[i] represents the number of links from i to other nodes
"""
def load_graph(file_path: Path):
    L = defaultdict(set)
    D = defaultdict(int)
    nodes = set()
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            source_node, target_node = map(int, line.split())
            
            if source_node not in L[target_node]:
                # Build incoming adjacency list
                L[target_node].add(source_node)

                # Build degree
                D[source_node] += 1

            nodes.add(source_node)
            nodes.add(target_node)
    return L, D, nodes


"""
Output:
- core_nodes: nodes remaining after recursively removing dangling nodes
- core_D: out-degree in the reduced core graph
- removed_order: dangling nodes in order of removal
"""
def remove_dangling_nodes(L, D, nodes):
    core_nodes = set(nodes)
    core_D = {node: D[node] for node in nodes}
    removed_order = []

    dangling_queue = deque(node for node in core_nodes if core_D[node] == 0)

    while dangling_queue:
        node = dangling_queue.popleft()

        if node not in core_nodes or core_D[node] != 0:
            continue

        core_nodes.remove(node)
        removed_order.append(node)

        for incoming_node in L[node]:
            if incoming_node not in core_nodes:
                continue

            core_D[incoming_node] -= 1
            if core_D[incoming_node] == 0:
                dangling_queue.append(incoming_node)

    return core_nodes, core_D, removed_order


"""
Output:
- ranks: PageRank scores for nodes in the reduced core graph
"""
def compute_pagerank_core(L, core_D, core_nodes):
    core_size = len(core_nodes)

    # initialisation
    ranks = {node: 1 / core_size for node in core_nodes}

    # teleporation value
    teleport = (1 - BETA) / core_size 

    for _ in range(ITERATIONS):
        updated_ranks = {node: teleport for node in core_nodes}

        for node in core_nodes:
            for incoming_node in L[node]:
                if incoming_node not in core_nodes:
                    continue

                updated_ranks[node] += BETA * (ranks[incoming_node] / core_D[incoming_node])

        ranks = updated_ranks

    return ranks


"""
Output:
- full_ranks: PageRank scores for all nodes after reinserting dangling nodes
"""
def reinsert_dangling_nodes(L, D, nodes, removed_order, core_ranks):
    full_ranks = dict(core_ranks)
    active_nodes = set(nodes)
    teleport = (1 - BETA) / (len(nodes) + len(removed_order))

    while removed_order:
        node = removed_order.pop()
        active_nodes.add(node)
        score = teleport
    
        for incoming_node in L[node]:
            if incoming_node not in active_nodes:
                continue
            D[incoming_node] += 1
            score += BETA * (full_ranks[incoming_node] / D[incoming_node])
        full_ranks[node] = score

    return full_ranks


"""
Output:
- variance: population variance of the PageRank scores
- std_dev: population standard deviation of the PageRank scores
"""
def compute_rank_distribution_stats(ranks):
    scores = list(ranks.values())
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    std_dev = math.sqrt(variance)
    return variance, std_dev



def main():
    args = parse_args()
    input_path = resolve_input_path(args.input_file)

    # load graph
    L, D, nodes = load_graph(input_path)

    # remove dangling nodes
    core_nodes, core_D, removed_order = remove_dangling_nodes(L, D, nodes)

    # get the PageRank for the remaining graph
    core_ranks = compute_pagerank_core(L, core_D, core_nodes)

    # update score for removed nodes in reverse order
    full_ranks = reinsert_dangling_nodes(L, core_D, core_nodes, removed_order, core_ranks)
    variance, std_dev = compute_rank_distribution_stats(full_ranks)

    top_nodes = sorted(full_ranks.items(), key=lambda item: item[1], reverse=True)[:10]

    print(f"Total nodes: {len(nodes)}")
    print(f"Core nodes: {len(core_nodes)}")
    print(f"Number of dead end nodes: {len(nodes) - len(core_nodes)}")
    print(f"p: {1-BETA:.2f}")
    print(f"Variance: {variance:.12e}")
    print(f"Standard deviation: {std_dev:.12e}")
    print("Top 10 nodes:")
    for node, score in top_nodes:
        print(f"{node}\t{score:.12f}")


if __name__ == "__main__":
    main()
