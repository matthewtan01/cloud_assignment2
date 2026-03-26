# PageRank And GraphRAG

This repository contains two scripts:

- `pagerank.py`: computes PageRank on the Google web graph dataset
- `graphrag.py`: computes personalized PageRank on a knowledge graph using the query entities as the restart distribution

## Requirements

- Python 3

## Run `pagerank.py`

`pagerank.py` expects one of the Google web graph files:

- `web-Google_10k.txt`
- `web-Google.txt`

Place these files inside the `dataset/` folder before running the script.

If you are using the dataset folder layout expected by the script, run:

```bash
python3 pagerank.py web-Google_10k.txt
```

or

```bash
python3 pagerank.py web-Google.txt
```

You can also pass a direct file path:

```bash
python3 pagerank.py dataset/web-Google_10k.txt
```

### What `pagerank.py` prints

- total number of nodes
- number of core nodes after removing dangling nodes
- number of dead end nodes
- variance and standard deviation of the PageRank scores
- top 10 highest-ranked nodes

## Run `graphrag.py`

`graphrag.py` expects:

1. a knowledge graph file
2. one or more query entities

For this repository, use the provided `example_kg.txt` file when running `graphrag.py`.
The script expects tuple-style triples only, like the ones in `example_kg.txt`.

### Required knowledge graph format

Each line must be a tuple-style triple:

```text
("Singapore", "hasPrimeMinister", "Lawrence Wong"),
("Singapore", "locatedIn", "Southeast Asia"),
```

### Example command

Run on the provided example knowledge graph file:

```bash
python3 graphrag.py example_kg.txt --entities "Singapore" "Government" --top-k 5
```

### Optional arguments

- `--damp`: damping factor for personalized PageRank, default `0.85`
- `--top-k`: number of top entities to return, default `3`
- `--max-iterations`: maximum iterations, default `100`
- `--tol`: convergence tolerance, default `1e-10`
- `--include-query-entity`: include the query entities in the final ranking

Example with optional arguments:

```bash
python3 graphrag.py example_kg.txt --entities "Singapore" "Government" --top-k 8 --damp 0.85 --max-iterations 100 --tol 1e-10
```

### What `graphrag.py` prints

- graph file name
- total number of nodes
- query entities
- damping factor
- number of iterations
- top `k` key entities ranked by personalized PageRank

## Notes

- In `graphrag.py`, the initial probability distribution is assigned only to the query entities.
- The teleportation vector is also assigned only to the query entities.
- Dangling-node probability mass is redistributed back through the same query-entity distribution.
