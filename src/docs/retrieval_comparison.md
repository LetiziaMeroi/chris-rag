# Retrieval Comparison

## Goal

The goal of this experiment is to compare different retrieval strategies for the CHRIS RAG system.

The current comparison includes:

* TF-IDF retrieval
* BM25 keyword retrieval
* semantic embedding retrieval
* naive hybrid retrieval
* Reciprocal Rank Fusion (RRF) hybrid retrieval
* extractive answer sentence ranking

All retrieval methods are evaluated on the same set of manually curated questions.

## Data

The retrieval corpus is built from PDF documents extracted and chunked from the CHRIS document collection.

Current input files include:

* governance documents
* published papers
* CHRIS baseline codebook

The current chunking strategy uses:

* chunk size: 1200 characters
* chunk overlap: 200 characters

Total number of chunks:

```text
955
```

## Retrieval methods

### TF-IDF

TF-IDF is used as a simple lexical baseline. It ranks chunks according to weighted term overlap between the query and the document chunks.

### BM25

BM25 is a stronger sparse retrieval baseline. It is keyword-based and works well when the query contains exact terms also present in the documents, such as CHRIS-specific terminology, policy names, variables, or numerical facts.

### MiniLM embeddings

The first semantic retrieval baseline used:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

This model supports multilingual semantic similarity, but in this experiment it often retrieved topically related chunks that did not contain the exact expected answer.

### E5-base embeddings

The second semantic retrieval model used:

```text
intfloat/multilingual-e5-base
```

This model is retrieval-oriented and uses different prefixes for queries and passages:

```text
query: <user question>
passage: <document chunk>
```

Switching from MiniLM to E5-base substantially improved semantic retrieval performance.

### Naive hybrid retrieval

The naive hybrid method combines normalized BM25 and embedding scores:

```text
final_score = alpha * bm25_score + (1 - alpha) * embedding_score
```

The tested configuration used:

```text
alpha = 0.7
```

This gave more weight to BM25 because BM25 performed better than MiniLM embeddings on the initial evaluation set.

### RRF hybrid retrieval

Reciprocal Rank Fusion combines rank positions instead of raw scores:

```text
RRF score = 1 / (k + BM25 rank) + 1 / (k + embedding rank)
```

The tested configuration used:

```text
k = 60
```

RRF avoids directly combining BM25 scores and cosine similarity scores, which have different scales.

## Retrieval results

The following table reports retrieval performance on 18 manually curated evaluation questions.

| Method              | Hit@1 | Hit@3 | Hit@5 |   MRR |
| ------------------- | ----: | ----: | ----: | ----: |
| TF-IDF              | 0.667 | 0.722 | 0.778 | 0.706 |
| BM25                | 0.611 | 0.889 | 0.944 | 0.733 |
| MiniLM embeddings   | 0.333 | 0.611 | 0.667 | 0.468 |
| Naive hybrid MiniLM | 0.500 | 0.611 | 0.833 | 0.606 |
| RRF hybrid MiniLM   | 0.500 | 0.722 | 0.778 | 0.613 |
| E5-base embeddings  | 0.667 | 0.778 | 0.944 | 0.746 |
| RRF hybrid E5-base  | 0.667 | 0.944 | 1.000 | 0.798 |

## Retrieval interpretation

BM25 performs strongly because many questions require exact factual evidence, such as participant counts, policy names, and CHRIS-specific terminology.

The first embedding model, MiniLM, performs worse than BM25. It often retrieves passages that are semantically related to the query but do not contain the exact answer evidence.

The retrieval-oriented multilingual E5 model substantially improves semantic retrieval compared with the MiniLM embedding baseline.

The best overall retrieval configuration is:

```text
RRF hybrid E5-base
```

This method reaches:

```text
Hit@1 = 0.667
Hit@3 = 0.944
Hit@5 = 1.000
MRR   = 0.798
```

This means that, for all evaluation questions, a relevant chunk is retrieved within the top 5 results.

## Answer extraction prototype

After selecting the best retrieval configuration, the system was extended with a first extractive answer prototype.

The current answer pipeline is:

1. retrieve top chunks using RRF hybrid retrieval with BM25 and E5-base embeddings;
2. split retrieved chunks into candidate sentences;
3. remove low-information sentences, such as table-of-contents fragments;
4. encode the user query with the E5 `query:` prefix;
5. encode candidate sentences with the E5 `passage:` prefix;
6. rank answer sentence candidates by cosine similarity;
7. return the top answer candidates with file, page, and chunk citations.

This approach does not generate new text. It is an extractive and citation-grounded baseline. This is safer than generative answering because every answer candidate is copied from a retrieved source passage.

## Answer extraction evaluation

The answer extraction step was evaluated using the same 18 manually curated questions. The evaluation checks whether one of the expected answer strings appears in the top-ranked answer sentence candidates.

| Method                    | Answer Hit@1 | Answer Hit@3 | Answer MRR |
| ------------------------- | -----------: | -----------: | ---------: |
| RRF E5 + sentence ranking |        0.556 |        0.833 |      0.694 |

## Answer extraction interpretation

The answer extraction performance is lower than the retrieval performance.

This is expected because answer extraction is a stricter task:

```text
retrieval evaluation:
Is a relevant chunk retrieved?

answer extraction evaluation:
Is the exact answer sentence selected?
```

The RRF E5 retriever reaches Hit@5 = 1.000, meaning that the relevant evidence is present in the retrieved context for all evaluated questions. However, the sentence-level ranking sometimes selects a semantically related sentence that does not contain the exact expected answer string.

This shows that the main remaining gap is not only retrieval, but answer selection and answer generation.

## Example

Question:

```text
How many participants are in the CHRIS baseline study?
```

Top answer candidate:

```text
The CHRIS baseline study has 13,393 participants.
```

Source:

```text
1772024710-chris-citation-guide_v1-4.pdf, page 3, chunk 1772024710-chris-citation-guide_v1-4_p3_c1
```

## Grounded prompt construction

The next prototype step builds a grounded prompt for generative question answering.

The prompt contains:

* the user question;
* the top answer sentence candidates;
* the retrieved evidence chunks;
* strict instructions to answer only from the provided evidence;
* citation labels for every source passage.

This step does not yet call a generative model. It prepares the input that will later be passed to an LLM and makes the grounding process inspectable.

The prompt ends with:

```text
Final answer:
```

This field is intentionally empty because no LLM is currently called. A future generative RAG component will complete this section using only the retrieved evidence.

## Current limitations

The current system has several limitations:

* The evaluation set is still small.
* Expected string matching is a simple approximation of relevance.
* Some correct or partially correct answers may be counted as wrong if they do not contain the exact expected string.
* Sentence-level extraction can fail when the answer requires combining multiple sentences.
* Tables are not yet handled as first-class retrieval units.
* Codebook and GWAS-style tabular data are not yet evaluated separately.
* The current system does not yet generate natural-language final answers.
* The current system does not yet include a reranker or LLM-based answer generation.

## Next steps

The next planned steps are:

1. Add structured JSON output for RAG results.
2. Continue improving the answer extraction evaluation.
3. Add expected source metadata, such as expected file, page, or chunk ID.
4. Build a final extractive answer formatter.
5. Add a grounded generative answer module.
6. Evaluate whether generated answers are faithful to the retrieved evidence.
7. Extend retrieval and evaluation to tables, codebook variables, and GWAS files.
