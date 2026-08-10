from pathlib import Path
import argparse
import sys
from typing import List, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer

from src.rag.answer_with_context import (
    MODEL_NAME,
    retrieve,
    rank_answer_sentences,
    clean_snippet,
)


DEFAULT_OUTPUT_PATH = Path("src/evaluation/sample_grounded_prompt.txt")


def format_evidence(results: List[Dict], max_chars_per_chunk: int = 1200) -> str:
    evidence_blocks = []

    for item in results:
        citation = (
            f"[{item['rank']}] "
            f"{item['file_name']}, "
            f"page {item['page']}, "
            f"chunk {item['chunk_id']}"
        )

        text = clean_snippet(item["text"], max_chars=max_chars_per_chunk)

        block = f"{citation}\n{text}"
        evidence_blocks.append(block)

    return "\n\n".join(evidence_blocks)


def format_answer_candidates(answer_candidates: List[Dict]) -> str:
    blocks = []

    for i, item in enumerate(answer_candidates, start=1):
        source = item["source"]

        citation = (
            f"[A{i}] "
            f"{source['file_name']}, "
            f"page {source['page']}, "
            f"chunk {source['chunk_id']}"
        )

        block = f"{citation}\n{item['sentence']}"
        blocks.append(block)

    return "\n\n".join(blocks)


def build_grounded_prompt(
    question: str,
    evidence: str,
    answer_candidates: str,
) -> str:
    prompt = f"""You are a careful assistant for the CHRIS RAG system.

Answer the user question using only the evidence provided below.

Rules:
- Do not use outside knowledge.
- If the evidence is insufficient, say: "The retrieved evidence is insufficient to answer this question."
- Keep the answer concise.
- Include citations using the source labels, for example [1] or [A1].
- Do not cite a source unless it directly supports the sentence.

User question:
{question}

Answer sentence candidates:
{answer_candidates}

Retrieved evidence:
{evidence}

Final answer:
"""
    return prompt


def main():
    parser = argparse.ArgumentParser(
        description="Build a grounded RAG prompt from retrieved evidence."
    )
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path where the grounded prompt will be saved.",
    )

    args = parser.parse_args()

    print("Retrieving evidence...")
    results = retrieve(args.query, top_k=args.top_k)

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    answer_candidates = rank_answer_sentences(
        query=args.query,
        results=results,
        model=model,
        top_n=3,
    )

    evidence_text = format_evidence(results)
    answer_candidate_text = format_answer_candidates(answer_candidates)

    prompt = build_grounded_prompt(
        question=args.query,
        evidence=evidence_text,
        answer_candidates=answer_candidate_text,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write(prompt)

    print(f"Prompt written to: {output_path}")


if __name__ == "__main__":
    main()
