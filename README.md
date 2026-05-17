# SignalRank

## Goal
This system takes natural language queries from `data/queries.csv`, ranks relevant people from `data/random_actors.json`, and writes the final results to `outputs/results.json`.

## How to Run
Install the dependencies and start Memgraph locally:

```bash
pip install -r requirements.txt
docker compose up -d
python -m src.main
```

OpenRouter remains optional. Memgraph is required for the graph retrieval layer.

## Architecture
The retrieval flow is:

`queries -> query parser -> HNSW vector retrieval + Memgraph graph retrieval -> graph-aware reranker -> JSON output`

For each query, the system:
- parses deterministic intent signals
- compares the query against one actor-level embedding per profile through a local HNSW index
- computes exact relational matches through a graph index over companies, locations, roles, domains, schools, and signals
- applies structured boosts for especially important matches
- writes final ranked results to JSON

## Actor Transformation
- Raw nested JSON is flattened from `profile`, `professional.current_position`, `work_experience`, `education`, and accomplishments.
- A field-labeled `search_text` document is created instead of embedding raw JSON.
- Rule-based tags and signals are generated from the combined profile text.
- One normalized document is created per actor.
- Graph edges are created from profile relationships.

Each transformed actor includes:
- `actor_id`
- `name`
- `location`
- `normalized_location`
- `headline`
- `bio`
- `current_title`
- `current_company`
- `companies`
- `titles`
- `education`
- `schools`
- `raw_text`
- `search_text`
- `tags`

## Embeddings and HNSW
The vector text is the normalized actor profile document built from:
- name
- location
- headline
- bio
- current role
- work experience
- education
- accomplishments
- generated signals

I represent each actor as one profile-level vector instead of splitting into chunks because the search target is a person, not a document passage.

The default local embedding path is `all-MiniLM-L6-v2`. HNSW is used because it gives fast local semantic retrieval while keeping the vector layer simple and swappable. Query vectors and actor vectors are compared through the HNSW index and converted into a normalized `0.0-1.0` `vector_score`.

## Graph Retrieval with Memgraph
Actor profiles are relational, not just textual. Each person connects to companies, locations, roles, domains, schools, and profile signals through graph edges such as `WORKED_AT`, `LOCATED_IN`, `HAS_ROLE`, `HAS_DOMAIN`, `HAS_SIGNAL`, and `STUDIED_AT`.

This graph layer helps with exact relationship queries like:
- worked at Google
- founders in San Francisco
- marketing experts in fintech
- ML people in Bangalore
- mentor or advisor profiles with computer vision and startup signals

Memgraph is the graph retrieval layer used by the prototype. It stores the actor relationship graph and handles exact constraint matching over companies, locations, roles, domains, schools, and signals.

## Ranking
The final ranking score combines three pieces:
- vector score from HNSW semantic retrieval
- graph score from exact graph constraint matches
- structured boost score from rule-based query-aware matches

Weighted formula:

```txt
final_score =
  0.55 * vector_score
+ 0.35 * graph_score
+ 0.10 * structured_boost
```

## Why No BM25 / TF-IDF
I intentionally skipped BM25/TF-IDF because exact relational constraints are handled through the graph index, while fuzzy semantic intent is handled through the HNSW vector index. This separates structured matching from semantic similarity and keeps the retrieval logic explainable.

## Why No External Vector DB
The dataset is small, so a local HNSW index is enough for this prototype. The vector retrieval layer is modular and can later be swapped with Pinecone, Qdrant, or another vector database if the corpus grows.

## OpenRouter Usage
OpenRouter is optional only. It can be used as an alternative embedding provider and as an optional post-ranking layer for `debug_results.json` explanation text. The core ranking path stays local and deterministic: OpenRouter does not decide which actors rank where, and the system falls back to local explanations if the key or request is unavailable.

## Example
For the query `Marketing experts in fintech`, the top result in this run was `David Thompson`. That result makes sense because his profile aligns in all three layers: strong semantic similarity in the HNSW vector search, graph matches for marketing and fintech relationships, and rule-based boosts for relevant tags.

## Outputs
- `outputs/results.json`
- optional `outputs/debug_results.json`
