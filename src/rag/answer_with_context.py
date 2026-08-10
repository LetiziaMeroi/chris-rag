# print answers for debug
from pathlib import Path
import argparse
import json
import re
from typing import List, Dict

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


DATA_ROOT = Path("/storage/data/chris-rag")

CHUNKS_PATH = DATA_ROOT / "processed" / "chunks" / "document_chunks.jsonl"
EMBEDDINGS_PATH = DATA_ROOT / "processed" / "embeddings_e5_base" / "document_embeddings.npy"

MODEL_NAME = "intfloat/multilingual-e5-base"

TOP_K = 5
RRF_K = 60
CANDIDATE_K = 100


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
    "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "how", "many", "much",
    "does", "do", "did", "with", "from", "by", "as", "at",
}


def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z0-9]+", text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def load_chunks(path: Path) -> List[Dict]:
    chunks = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def rank_indices(scores: np.ndarray, candidate_k: int) -> List[int]:
    return list(np.argsort(scores)[::-1][:candidate_k])


def build_rank_map(indices: List[int]) -> Dict[int, int]:
    return {idx: rank for rank, idx in enumerate(indices, start=1)}


def reciprocal_rank_fusion(
    bm25_ranked: List[int],
    embedding_ranked: List[int],
    rrf_k: int = RRF_K,
) -> Dict[int, float]:
    scores = {}

    bm25_ranks = build_rank_map(bm25_ranked)
    embedding_ranks = build_rank_map(embedding_ranked)

    all_indices = set(bm25_ranks.keys()) | set(embedding_ranks.keys())

    for idx in all_indices:
        score = 0.0

        if idx in bm25_ranks:
            score += 1.0 / (rrf_k + bm25_ranks[idx])

        if idx in embedding_ranks:
            score += 1.0 / (rrf_k + embedding_ranks[idx])

        scores[idx] = score

    return scores


def clean_snippet(text: str, max_chars: int = 900) -> str:
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def retrieve(query: str, top_k: int = TOP_K) -> List[Dict]:
    chunks = load_chunks(CHUNKS_PATH)
    texts = [chunk["text"] for chunk in chunks]

    # BM25
    tokenized_corpus = [tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_scores = np.array(bm25.get_scores(tokenize(query)))
    bm25_ranked = rank_indices(bm25_scores, CANDIDATE_K)
    bm25_rank_map = build_rank_map(bm25_ranked)

    # E5 embeddings
    embeddings = np.load(EMBEDDINGS_PATH)
    model = SentenceTransformer(MODEL_NAME)

    query_embedding = model.encode(
        ["query: " + query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    embedding_scores = embeddings @ query_embedding
    embedding_ranked = rank_indices(embedding_scores, CANDIDATE_K)
    embedding_rank_map = build_rank_map(embedding_ranked)

    # RRF fusion
    rrf_scores = reciprocal_rank_fusion(
        bm25_ranked=bm25_ranked,
        embedding_ranked=embedding_ranked,
        rrf_k=RRF_K,
    )

    top_indices = sorted(
        rrf_scores.keys(),
        key=lambda idx: rrf_scores[idx],
        reverse=True,
    )[:top_k]

    results = []

    for rank, idx in enumerate(top_indices, start=1):
        chunk = chunks[idx]

        results.append({
            "rank": rank,
            "rrf_score": float(rrf_scores[idx]),
            "bm25_rank": bm25_rank_map.get(idx, ""),
            "embedding_rank": embedding_rank_map.get(idx, ""),
            "file_name": chunk["file_name"],
            "collection": chunk["collection"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
        })

    return results



def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip().split()) >= 8]


def is_low_information_sentence(sentence: str) -> bool:
    s = sentence.lower()

    bad_patterns = [
        "table of contents",
        "................................................................",
        "downloaded from",
        "copyright",
    ]

    if any(pattern in s for pattern in bad_patterns):
        return True

    letters = sum(ch.isalpha() for ch in sentence)
    if letters < 30:
        return True

    return False


def rank_answer_sentences(
    query: str,
    results: List[Dict],
    model: SentenceTransformer,
    top_n: int = 3,
) -> List[Dict]:
    candidates = []

    for item in results:
        for sentence in split_sentences(item["text"]):
            if is_low_information_sentence(sentence):
                continue

            candidates.append({
                "sentence": sentence,
                "source": item,
            })

    if not candidates:
        return []

    query_embedding = model.encode(
        ["query: " + query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    sentence_texts = [
        "passage: " + candidate["sentence"]
        for candidate in candidates
    ]

    sentence_embeddings = model.encode(
        sentence_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    scores = sentence_embeddings @ query_embedding

    ranked_indices = np.argsort(scores)[::-1][:top_n]

    ranked = []

    for idx in ranked_indices:
        candidate = candidates[idx]
        ranked.append({
            "sentence": candidate["sentence"],
            "score": float(scores[idx]),
            "source": candidate["source"],
        })

    return ranked

def build_output_object(
    query: str,
    results: List[Dict],
    answer_sentences: List[Dict],
) -> Dict:
    """
    Build a structured output object that can be saved as JSON.

    This is useful for:
    - evaluation
    - future UI
    - passing grounded evidence to a generative LLM
    """

    answer_candidates = []

    for rank, item in enumerate(answer_sentences, start=1):
        source = item["source"]

        answer_candidates.append({
            "rank": rank,
            "sentence": item["sentence"],
            "score": item["score"],
            "source": {
                "file_name": source["file_name"],
                "collection": source["collection"],
                "page": source["page"],
                "chunk_id": source["chunk_id"],
            },
        })

    retrieved_evidence = []

    for item in results:
        retrieved_evidence.append({
            "rank": item["rank"],
            "rrf_score": item["rrf_score"],
            "bm25_rank": item["bm25_rank"],
            "embedding_rank": item["embedding_rank"],
            "file_name": item["file_name"],
            "collection": item["collection"],
            "page": item["page"],
            "chunk_id": item["chunk_id"],
            "text": clean_snippet(item["text"], max_chars=1200),
        })

    return {
        "question": query,
        "answer_candidates": answer_candidates,
        "retrieved_evidence": retrieved_evidence,
    }

def print_answer(query: str, results: List[Dict], answer_sentences: List[Dict]):
    print("=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(query)

    print()
    print("=" * 80)
    print("ANSWER SENTENCE CANDIDATES")
    print("=" * 80)


    if not answer_sentences:
        print("No clear answer sentence found in the retrieved evidence.")
    else:
        for i, item in enumerate(answer_sentences, start=1):
            source = item["source"]
            print()
            print(f"[{i}] {item['sentence']}")
            print(
                f"Score: {item['score']:.4f} | "
                f"Source: {source['file_name']}, "
                f"page {source['page']}, "
                f"chunk {source['chunk_id']}"
            )

    print()
    print("=" * 80)
    print("RETRIEVED EVIDENCE")
    print("=" * 80)

    for item in results:
        citation = (
            f"[{item['rank']}] "
            f"{item['file_name']}, "
            f"page {item['page']}, "
            f"chunk {item['chunk_id']}"
        )

        print()
        print(citation)
        print(
            f"RRF={item['rrf_score']:.6f} | "
            f"BM25 rank={item['bm25_rank']} | "
            f"Embedding rank={item['embedding_rank']}"
        )
        print("-" * 80)
        print(clean_snippet(item["text"]))



def main():
    parser = argparse.ArgumentParser(
        description="Retrieve evidence for a question using RRF hybrid retrieval."
    )
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional path where the structured RAG output will be saved as JSON.",
    )

    args = parser.parse_args()

    results = retrieve(args.query, top_k=args.top_k)

    model = SentenceTransformer(MODEL_NAME)

    answer_sentences = rank_answer_sentences(
        query=args.query,
        results=results,
        model=model,
        top_n=3,
    )

    print_answer(
        query=args.query,
        results=results,
        answer_sentences=answer_sentences,
    )

    if args.json_output is not None:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_object = build_output_object(
            query=args.query,
            results=results,
            answer_sentences=answer_sentences,
        )

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_object, f, ensure_ascii=False, indent=2)

        print()
        print(f"Structured JSON output written to: {output_path}")

if __name__ == "__main__":
    main()