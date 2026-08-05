# Retrieval Comparison

## Goal

The goal of this experiment is to compare different retrieval strategies for the CHRIS RAG system.

The current comparison includes:

- TF-IDF 
- BM25 keyword retrieval
- Semantic embedding retrieval
- Hybrid retrieval = BM25 + semantic embeddings

Both methods are evaluated on the same set of manually curated questions.

## Data

The retrieval corpus is built from PDF documents extracted and chunked from the CHRIS document collection.

Current input files include:

- governance documents
- published papers
- CHRIS baseline codebook

The current chunking strategy uses:

- chunk size: 1200 characters
- chunk overlap: 200 characters

Total number of chunks:

```text
955

## Results

| | TF-IDF | BM25| Embeddings |
| Hit@1 | 0.667 | 0.611 | 0.389
| Hit@3 | 0.722 | 0.889 | 0.611
| Hit@5 | 0.778 | 0.944 | 0.722
| MRR | 0.706 | 0.733 | 0.491

## Evaluation 05/08/2026

BM25 currently performs better than semantic embeddings on the evaluation set. 
This is likely because many questions require exact factual evidence, such as participant counts, policy names, 
and specific CHRIS terminology. Semantic retrieval often retrieves topically related chunks but may miss the exact answer string.
