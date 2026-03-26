from __future__ import annotations

import argparse
import ast
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="GraphRAG retrieval using personalized PageRank from provided query entities.",
    )
    parser.add_argument(
        "graph_file",
        help="Path to the knowledge-graph file. Each line should be 'source target' or 'source relation target'.",
    )
    parser.add_argument(
        "--entities",
        nargs="+",
        required=True,
        help="Query entities to use as personalized PageRank queries.",
    )
    parser.add_argument(
        "--damp",
        type=float,
        default=0.85,
        help="Probability of following graph edges during PageRank propagation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of most relevant nodes to return.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum number of personalized PageRank iterations.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-10,
        help="Convergence tolerance based on L1 difference between iterations.",
    )
    parser.add_argument(
        "--include-query-entity",
        action="store_true",
        help="Include the query entities in the returned ranking.",
    )
    return parser.parse_args()

def parse_kg_line(line, line_number):
    if line.startswith("("):
        parsed = ast.literal_eval(line.rstrip(","))
        if not isinstance(parsed, tuple) or len(parsed) != 3:
            raise ValueError(
                f"Invalid graph line {line_number}: tuple input must be (source, relation, target)",
            )
        source, _relation, target = parsed
        return str(source).strip(), str(target).strip()

    parts = line.split("\t")
    if len(parts) == 1:
        parts = line.split()

    if len(parts) == 2:
        source, target = (part.strip() for part in parts)
        return source, target

    if len(parts) == 3:
        source, _relation, target = (part.strip() for part in parts)
        return source, target

    raise ValueError(
        f"Invalid graph line {line_number}: expected 2 or 3 columns, got {len(parts)}",
    )


"""
Output:
- Adjacency List L, where L[i] represents the set of nodes that link to i
- Degree D, where D[i] represents the number of links from i to other nodes
"""
def load_knowledge_graph(file_path):
    L = defaultdict(set)
    D = defaultdict(int)
    nodes = set()

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            source, target = parse_kg_line(line, line_number)

            if source not in L[target]:
                L[target].add(source)
                D[source] += 1

            nodes.add(source)
            nodes.add(target)

    for node in nodes:
        L.setdefault(node, set())
        D.setdefault(node, 0)

    return L, D, sorted(nodes)


# Initial PageRank (distributed among query entities only)
def build_query_vector(nodes, query_entities):
    node_set = set(nodes)
    noramlised_query = [query.strip() for query in query_entities]
    missing = [query for query in noramlised_query if query not in node_set]
    if missing:
        raise ValueError(f"Seed entities not found in graph: {', '.join(missing)}")

    weight = 1.0 / len(noramlised_query)
    return {node: weight if node in noramlised_query else 0.0 for node in nodes}


def personalized_pagerank(L, D, nodes, queries, damp, max_iterations, tolerance):
    query_vector = build_query_vector(nodes, queries)
    ranks = dict(query_vector)

    for iteration in range(1, max_iterations + 1):
        # teleportation vector distributed among query entities only
        new_ranks = {node: (1.0 - damp) * query_vector[node] for node in nodes}
        dangling_mass = 0.0

        for node in nodes:
            if D[node] == 0:
                dangling_mass += ranks[node]

        for node in nodes:
            incoming_score = 0.0
            for incoming_node in L[node]:
                incoming_score += ranks[incoming_node] / D[incoming_node]

            new_ranks[node] += damp * incoming_score
            new_ranks[node] += damp * dangling_mass * query_vector[node]

        rank_sum = sum(new_ranks.values())
        if rank_sum > 0.0:
            new_ranks = {node: score / rank_sum for node, score in new_ranks.items()}

        delta = sum(abs(new_ranks[node] - ranks[node]) for node in nodes)
        ranks = new_ranks
        if delta < tolerance:
            return ranks, iteration

    return ranks, max_iterations


def top_k_nodes(ranks, query_entities, k, include_seeds):
    query_set = {entity.strip() for entity in query_entities}
    if include_seeds:
        filtered = list(ranks.items())
    else:
        filtered = [(node, score) for node, score in ranks.items() if node not in query_set]
    return sorted(filtered, key=lambda item: item[1], reverse=True)[:k]


def print_results(graph_file, nodes, query_entities, damp, iterations, ranks, top_k, include_query_entity):
    print(f"Graph file: {graph_file}")
    print(f"Total nodes: {len(nodes)}")
    print(f"Query entities: {', '.join(query_entities)}")
    print(f"Damping Factor: {damp:.2f}")
    print(f"Iterations: {iterations}")
    print()
    print(f"Top {top_k} key entities:")

    for index, (node, score) in enumerate(top_k_nodes(ranks, query_entities, top_k, include_query_entity), start=1):
        print(f"{index}. {node}  score={score:.12e}")


def main():
    args = parse_args()
    L, D, nodes = load_knowledge_graph(args.graph_file)
    ranks, iterations = personalized_pagerank(L, D, nodes, args.entities, args.damp, args.max_iterations, args.tol)
    print_results(args.graph_file, nodes, args.entities, args.damp, iterations, ranks, args.top_k, args.include_query_entity)


if __name__ == "__main__":
    main()