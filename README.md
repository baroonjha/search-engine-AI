Build Your Own Search Engine
================================

This repository contains code and notes for a "Build Your Own Search Engine" workshop. It demonstrates a simple pipeline for turning documents into searchable vectors, indexing them, and answering queries with approximate nearest-neighbor search and optional reranking.

Contents
--------
- `ingest.py` — (suggested) scripts to read documents and produce a corpus (JSONL with text + metadata).
- `build_index.py` — (suggested) scripts to create vector representations and write the vector index and metadata to disk.
- `search.py` — (suggested) load index and metadata and run queries from CLI or API.
- `notebooks/` — interactive demos (optional).

Architecture (high level)
------------------------
- Ingest: read documents (text, PDF, markdown) and store raw text and metadata.
- Preprocess: clean, normalize, and optionally split long documents into passages.
- Vectorize / Embed: convert text to vectors using either sparse vectorizers (TF-IDF) or dense embeddings (pretrained models).
- Index: store vectors in a vector store (FAISS / Annoy / HNSW / in-memory) and save metadata for each vector.
- Retrieve: given a query, compute its vector and perform nearest-neighbor search to get candidate documents.
- Rerank: optionally apply a reranker (cross-encoder, BM25) to reorder candidates for higher quality.
- Serve: expose a simple CLI or web API to run queries and show results.

How search actually happens (brief)
---------------------------------
1. A user issues a text query.
2. The query is preprocessed the same way as documents (tokenization, lowercasing, optional stopword handling).
3. The query is converted into a numeric vector: either a TF-IDF sparse vector or a dense embedding from a neural model.
4. The vector is compared with indexed document vectors using a similarity measure (commonly cosine similarity for dense vectors; cosine or dot-product for normalized vectors).
5. The nearest vectors (top-K) are returned as candidates along with their metadata (document id, offset, score).
6. Optionally, a reranker computes a more accurate relevance score on the query + candidate pairs and the results are returned to the user.

Embeddings vs Vectorizers
-------------------------
- TF-IDF / CountVectorizer (sparse):
	- Represent text as high-dimensional sparse vectors (one dimension per token/term).
	- Fast and interpretable, works well for exact token matching and keyword search.
	- Good for small corpora and when keyword matches dominate relevance.
- Dense Embeddings (neural):
	- Map text into lower-dimensional dense vectors using models like Sentence Transformers.
	- Capture semantic similarity — paraphrases and synonyms map to nearby vectors.
	- Require ANN indexes (Faiss, Annoy, HNSW) for efficient large-scale search.

Similarity & Indexing
---------------------
- Similarity: cosine similarity on normalized vectors or dot product (after normalization) is common for embeddings.
- ANN indexes: use FAISS, Annoy, hnswlib for sublinear nearest-neighbor retrieval on large datasets.
- Hybrid search: combine sparse scores (BM25 / TF-IDF) with dense embedding scores to get the best of both worlds.

Minimal code examples
---------------------
TF-IDF example (scikit-learn):

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = ["document one text", "document two text"]
vec = TfidfVectorizer(max_features=5000)
X = vec.fit_transform(corpus)

q = "search text"
qv = vec.transform([q])
scores = cosine_similarity(qv, X).ravel()
topk = scores.argsort()[::-1][:5]
```

Dense embedding + FAISS example:

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

model = SentenceTransformer('all-MiniLM-L6-v2')
corpus = ["document one text", "document two text"]
embs = model.encode(corpus, convert_to_numpy=True)

# normalize for cosine similarity
faiss.normalize_L2(embs)
dim = embs.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embs)

q = "search text"
qv = model.encode([q], convert_to_numpy=True)
faiss.normalize_L2(qv)
scores, ids = index.search(qv, k=5)
```

How to run (suggested scripts)
-------------------------------
All the code are given in notebook.ipynb .So run this notebook cell one by one and understand it .

Suggested dependencies
----------------------
```
numpy
scikit-learn
pandas
request
sentence-transformers
torch
```

Tips and next steps
-------------------
- Evaluate retrieval quality with a small set of test queries and measure precision@k.
- Use passage-level splitting for better granularity on long documents.
- Consider hybrid search (BM25 + embeddings) for robust results.
- Add a simple web UI (Streamlit or FastAPI + template) to demo results interactively.


