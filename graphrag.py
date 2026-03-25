"""
Objective: 
Use GraphRAG to answer a multi-hop query, for instance, “What discoveries by Marie Curie led to advances in medical imaging?"

How: 
During retrieval stage, inject Pagerank scores starting from seed entities ("Marie Curie", "medical imaging"),
and propagating importance through the graph's connections, effectively ranking nodes based on their relevance
to the specific query context

Inputs:
1. knowledge graph
2. set of query entities -> seed entities

Output:
top k most relevant nodes
"""