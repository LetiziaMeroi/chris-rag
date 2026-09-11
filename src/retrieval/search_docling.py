from pathlib import Path
import argparse
import json
import re
from typing import Dict, List
import os

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


BASE_DATA_ROOT = Path(
    os.getenv(
        "CHRIS_DATA_ROOT",
        "/storage/data/chris-rag",
    )
)

DATA_ROOT = BASE_DATA_ROOT / "processed"

MODEL_NAME = "intfloat/multilingual-e5-base"

DEFAULT_TOP_K = 5
DEFAULT_RRF_K = 60
CANDIDATE_K = 100


LAYER_CONFIG = {
    "text": {
        "chunks": (
            DATA_ROOT
            / "chunks_docling"
            / "text_chunks.jsonl"
        ),
        "embeddings": (
            DATA_ROOT
            / "embeddings_docling_e5_base"
            / "text_embeddings.npy"
        ),
    },

    "table": {
        "chunks": (
            DATA_ROOT
            / "chunks_docling"
            / "table_chunks.jsonl"
        ),
        "embeddings": (
            DATA_ROOT
            / "embeddings_docling_e5_base"
            / "table_embeddings.npy"
        ),
    },
}


STOPWORDS = {
    "the", "a", "an", "and", "or",
    "of", "to", "in", "on", "for",
    "is", "are", "was", "were",
    "be", "been", "being",
    "what", "which", "who", "how",
    "many", "much",
    "does", "do", "did",
    "with", "from", "by", "as", "at",
}

_MODEL = None


def get_model():
    global _MODEL

    if _MODEL is None:
        print(f"Loading embedding model: {MODEL_NAME}")
        _MODEL = SentenceTransformer(MODEL_NAME)

    return _MODEL

def tokenize(text: str) -> List[str]:

    text = text.lower()

    # Unicode-aware tokenization.
    tokens = re.findall(
        r"\b\w+\b",
        text,
        flags=re.UNICODE,
    )

    return [
        token
        for token in tokens
        if (
            token not in STOPWORDS
            and len(token) > 1
        )
    ]


def load_chunks(path: Path) -> List[Dict]:

    chunks = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if line.strip():
                chunks.append(
                    json.loads(line)
                )

    return chunks


def rank_indices(
    scores: np.ndarray,
    candidate_k: int,
) -> List[int]:

    candidate_k = min(
        candidate_k,
        len(scores),
    )

    return list(
        np.argsort(scores)[::-1][
            :candidate_k
        ]
    )


def build_rank_map(
    indices: List[int],
) -> Dict[int, int]:

    return {
        idx: rank
        for rank, idx
        in enumerate(
            indices,
            start=1,
        )
    }


def reciprocal_rank_fusion(
    bm25_ranked: List[int],
    embedding_ranked: List[int],
    rrf_k: int,
):

    bm25_ranks = build_rank_map(
        bm25_ranked
    )

    embedding_ranks = build_rank_map(
        embedding_ranked
    )

    all_indices = (
        set(bm25_ranks)
        | set(embedding_ranks)
    )

    scores = {}

    for idx in all_indices:

        score = 0.0

        if idx in bm25_ranks:
            score += (
                1.0
                / (
                    rrf_k
                    + bm25_ranks[idx]
                )
            )

        if idx in embedding_ranks:
            score += (
                1.0
                / (
                    rrf_k
                    + embedding_ranks[idx]
                )
            )

        scores[idx] = score

    return scores


def clean_snippet(
    text: str,
    max_chars: int = 1200,
):

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars]
        .rsplit(" ", 1)[0]
        + "..."
    )


def search(
    query: str,
    layer: str = "text",
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = DEFAULT_RRF_K,
):

    if layer not in LAYER_CONFIG:
        raise ValueError(
            f"Unknown layer: {layer}"
        )

    config = LAYER_CONFIG[layer]

    chunks = load_chunks(
        config["chunks"]
    )

    embeddings = np.load(
        config["embeddings"]
    )

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Chunk/embedding count mismatch: "
            f"{len(chunks)} chunks vs "
            f"{len(embeddings)} embeddings"
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # -------------------------------------------------
    # BM25
    # -------------------------------------------------

    tokenized_corpus = [
        tokenize(text)
        for text in texts
    ]

    bm25 = BM25Okapi(
        tokenized_corpus
    )

    bm25_scores = np.array(
        bm25.get_scores(
            tokenize(query)
        )
    )

    bm25_ranked = rank_indices(
        bm25_scores,
        CANDIDATE_K,
    )

    bm25_rank_map = build_rank_map(
        bm25_ranked
    )

    # -------------------------------------------------
    # E5
    # -------------------------------------------------

    model = get_model()

    query_embedding = model.encode(
        [
            "query: " + query
        ],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    embedding_scores = (
        embeddings
        @ query_embedding
    )

    embedding_ranked = rank_indices(
        embedding_scores,
        CANDIDATE_K,
    )

    embedding_rank_map = (
        build_rank_map(
            embedding_ranked
        )
    )

    # -------------------------------------------------
    # Reciprocal Rank Fusion
    # -------------------------------------------------

    rrf_scores = (
        reciprocal_rank_fusion(
            bm25_ranked,
            embedding_ranked,
            rrf_k,
        )
    )

    top_indices = sorted(
        rrf_scores,
        key=lambda idx: (
            rrf_scores[idx]
        ),
        reverse=True,
    )[:top_k]

    results = []

    for rank, idx in enumerate(
        top_indices,
        start=1,
    ):

        chunk = chunks[idx]

        result = {
            "rank": rank,
            "rrf_score": float(
                rrf_scores[idx]
            ),
            "bm25_rank": (
                bm25_rank_map.get(idx)
            ),
            "embedding_rank": (
                embedding_rank_map.get(idx)
            ),
            "embedding_score": float(
                embedding_scores[idx]
            ),
            "file_name": (
                chunk.get("file_name")
            ),
            "collection": (
                chunk.get("collection")
            ),
            "page": (
                chunk.get("page")
            ),
            "pages": (
                chunk.get("pages", [])
            ),
            "chunk_id": (
                chunk.get("chunk_id")
            ),
            "chunk_type": (
                chunk.get("chunk_type")
            ),
            "text": chunk.get(
                "text",
                "",
            ),
        }

        if layer == "table":
            result["table_index"] = (
                chunk.get(
                    "table_index"
                )
            )

            result["caption"] = (
                chunk.get(
                    "caption"
                )
            )

        results.append(result)

    return results


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Hybrid BM25 + E5 + RRF "
            "retrieval over Docling chunks."
        )
    )

    parser.add_argument(
        "query",
        type=str,
    )

    parser.add_argument(
        "--layer",
        choices=[
            "text",
            "table",
        ],
        default="text",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    args = parser.parse_args()

    results = search(
        query=args.query,
        layer=args.layer,
        top_k=args.top_k,
    )

    print()
    print("=" * 80)
    print("DOCLING HYBRID RETRIEVAL")
    print("=" * 80)

    print(f"Query: {args.query}")
    print(f"Layer: {args.layer}")
    print(f"Results: {len(results)}")

    for item in results:

        print()
        print(
            "=" * 80
        )

        print(
            f"Rank: {item['rank']}"
        )

        print(
            f"RRF score: "
            f"{item['rrf_score']:.6f}"
        )

        print(
            f"BM25 rank: "
            f"{item['bm25_rank']}"
        )

        print(
            f"Embedding rank: "
            f"{item['embedding_rank']}"
        )

        print(
            f"Embedding score: "
            f"{item['embedding_score']:.4f}"
        )

        print(
            f"File: "
            f"{item['file_name']}"
        )

        print(
            f"Page: "
            f"{item['page']}"
        )

        print(
            f"Chunk: "
            f"{item['chunk_id']}"
        )

        if args.layer == "table":

            print(
                f"Table: "
                f"{item.get('table_index')}"
            )

            print(
                "Caption: "
                f"{item.get('caption')}"
            )

        print()
        print(
            clean_snippet(
                item["text"]
            )
        )


if __name__ == "__main__":
    main()
