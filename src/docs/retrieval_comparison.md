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

| Method | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| TF-IDF | 0.667 | 0.722 | 0.778 | 0.706 |
| BM25 | 0.611 | 0.889 | 0.944 | 0.733 |
| MiniLM embeddings | 0.333 | 0.611 | 0.667 | 0.468 |
| Naive hybrid MiniLM | 0.500 | 0.611 | 0.833 | 0.606 |
| RRF hybrid MiniLM | 0.500 | 0.722 | 0.778 | 0.613 |
| E5-base embeddings | 0.667 | 0.778 | 0.944 | 0.746 |
| RRF hybrid E5-base | 0.667 | 0.944 | 1.000 | 0.798 |

## Evaluation 05/08/2026

BM25 currently performs better than semantic embeddings on the evaluation set. 
This is likely because many questions require exact factual evidence, such as participant counts, policy names, 
and specific CHRIS terminology. Semantic retrieval often retrieves topically related chunks but may miss the exact answer string.

The retrieval-oriented multilingual E5 model substantially improves semantic retrieval compared with the MiniLM embedding baseline. 
Combining BM25 and E5 embeddings with Reciprocal Rank Fusion gives the best overall performance, reaching Hit@5 = 1.000 and the highest MRR.

The first answer prototype avoids hand-written question-specific rules. It performs sentence-level semantic ranking 
over the retrieved evidence and returns the most relevant answer sentences with file, page, and chunk citations. 
This provides an extractive, citation-grounded baseline before introducing a generative LLM.


