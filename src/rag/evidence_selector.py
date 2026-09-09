from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
import re

from rank_bm25 import BM25Okapi

MODEL_NAME = "intfloat/multilingual-e5-base"

_MODEL = None

def tokenize(text: str):
    return re.findall(
        r"\b\w+\b",
        text.lower(),
        flags=re.UNICODE,
    )


def get_ranks(scores):
    """
    Convert scores into 1-based ranks.
    Rank 1 = highest score.
    """

    order = np.argsort(scores)[::-1]

    ranks = np.empty(
        len(scores),
        dtype=int,
    )

    for rank, idx in enumerate(
        order,
        start=1,
    ):
        ranks[idx] = rank

    return ranks


def _valid_table_line(line: str) -> bool:
    line = line.strip()

    if not line:
        return False

    if line.startswith("#"):
        return False

    if line in {
        "[NO CAPTION]",
        "[NO LINKED FOOTNOTES]",
        "[NO NEARBY EXPLANATORY TEXT]",
    }:
        return False

    # Markdown separator:
    # |:------|:------|
    compact = line.replace(
        "|", ""
    ).replace(
        ":", ""
    ).replace(
        "-", ""
    ).strip()

    if not compact:
        return False

    # We are mainly interested in table rows.
    return "|" in line


def select_table_rows(
    query: str,
    results: List[Dict],
    top_n: int = 12,
    max_rows_per_chunk: int = 3,
) -> List[Dict]:
    """
    Select precise table rows using hybrid retrieval:

        BM25 + E5 + Reciprocal Rank Fusion

    The input consists of table chunks already retrieved
    by the document retrieval layer.

    This stage ranks individual rows so that precise
    evidence such as devices, variables and measurement
    protocols can be passed to the answer generator.
    """

    if not results:
        return []

    model = get_model()

    retrieval_query = query

    # --------------------------------------------------
    # Build row-level candidate collection
    # --------------------------------------------------

    candidates = []
    seen = set()

    for item in results:

        caption = item.get("caption") or ""

        for line in item["text"].splitlines():

            line = " ".join(
                line.strip().split()
            )

            if not _valid_table_line(line):
                continue

            key = (
                item["source"]["chunk_id"],
                line,
            )

            if key in seen:
                continue

            seen.add(key)

            passage = line

            if caption:
                passage = (
                    caption
                    + " "
                    + line
                )

            candidates.append({
                "item": item,
                "line": line,
                "passage": passage,
            })

    if not candidates:
        return []

    # --------------------------------------------------
    # BM25 row ranking
    # --------------------------------------------------

    tokenized_passages = [
        tokenize(candidate["passage"])
        for candidate in candidates
    ]

    bm25 = BM25Okapi(
        tokenized_passages
    )

    tokenized_query = tokenize(
        retrieval_query
    )

    bm25_scores = bm25.get_scores(
        tokenized_query
    )

    # --------------------------------------------------
    # E5 row ranking
    # --------------------------------------------------

    query_embedding = model.encode(
        [
            "query: "
            + retrieval_query
        ],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    passages = [
        "passage: "
        + candidate["passage"]
        for candidate in candidates
    ]

    passage_embeddings = model.encode(
        passages,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embedding_scores = (
        passage_embeddings
        @ query_embedding
    )

    # --------------------------------------------------
    # Reciprocal Rank Fusion
    # --------------------------------------------------

    bm25_ranks = get_ranks(
        bm25_scores
    )

    embedding_ranks = get_ranks(
        embedding_scores
    )

    rrf_k = 60

    rrf_scores = (
        1.0 / (
            rrf_k
            + bm25_ranks
        )
        +
        1.0 / (
            rrf_k
            + embedding_ranks
        )
    )

    ranked_indices = np.argsort(
        rrf_scores
    )[::-1]

    # --------------------------------------------------
    # Select final evidence rows
    # --------------------------------------------------

    selected = []

    rows_per_chunk = {}

    for idx in ranked_indices:

        candidate = candidates[idx]

        original = candidate["item"]

        chunk_id = (
            original["source"]["chunk_id"]
        )

        current_count = rows_per_chunk.get(
            chunk_id,
            0,
        )

        # Avoid one large table dominating
        # all the evidence.
        if current_count >= max_rows_per_chunk:
            continue

        result = dict(original)

        result["text"] = (
            candidate["line"]
        )

        result["selection_score"] = float(
            rrf_scores[idx]
        )

        result["row_bm25_rank"] = int(
            bm25_ranks[idx]
        )

        result["row_embedding_rank"] = int(
            embedding_ranks[idx]
        )

        result["row_embedding_score"] = float(
            embedding_scores[idx]
        )

        result["retrieval_query"] = (
            retrieval_query
        )

        selected.append(result)

        rows_per_chunk[chunk_id] = (
            current_count + 1
        )

        if len(selected) >= top_n:
            break

    return selected

def get_model():
    global _MODEL

    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME)

    return _MODEL


def expand_query(query: str) -> str:
    """
    Lightweight deterministic query expansion.

    The goal is not to answer the query,
    only to improve evidence retrieval.
    """

    lower = query.lower()

    expansions = []

    if "blood pressure" in lower:
        expansions.extend([
            "systolic blood pressure",
            "diastolic blood pressure",
            "blood pressure device",
            "blood pressure protocol",
            "repeated measurements",
            "office blood pressure",
            "continuous blood pressure",
        ])

    if "how" in lower or "measured" in lower:
        expansions.extend([
            "measurement method",
            "measurement protocol",
            "device",
            "instrument",
            "repeated measurement",
        ])

    if not expansions:
        return query

    return query + " " + " ".join(expansions)


def select_evidence(
    query: str,
    results: List[Dict],
    top_n: int = 8,
) -> List[Dict]:

    if not results:
        return []

    model = get_model()

    retrieval_query = query

    query_embedding = model.encode(
        ["query: " + retrieval_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    passages = [
        "passage: " + item["text"]
        for item in results
    ]

    passage_embeddings = model.encode(
        passages,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    scores = (
        passage_embeddings
        @ query_embedding
    )

    ranked_indices = (
        np.argsort(scores)[::-1][:top_n]
    )

    selected = []

    for idx in ranked_indices:
        item = dict(results[idx])

        item["selection_score"] = float(
            scores[idx]
        )

        item["retrieval_query"] = retrieval_query

        selected.append(item)

    return selected