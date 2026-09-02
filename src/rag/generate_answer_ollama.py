from pathlib import Path
import argparse
import json
import sys
from typing import List, Dict

from ollama import chat
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.rag.answer_with_context import (
    MODEL_NAME,
    retrieve,
    rank_answer_sentences,
    clean_snippet,
)


DEFAULT_OLLAMA_MODEL = "llama3.1:8b"


def format_answer_candidates(answer_candidates: List[Dict]) -> str:
    blocks = []

    for i, item in enumerate(answer_candidates, start=1):
        source = item["source"]
        label = f"A{i}"

        citation = (
            f"[{label}] "
            f"{source['file_name']}, "
            f"page {source['page']}, "
            f"chunk {source['chunk_id']}"
        )

        blocks.append(f"{citation}\n{item['sentence']}")

    return "\n\n".join(blocks)


def format_evidence(results: List[Dict], max_chars_per_chunk: int = 800) -> str:
    blocks = []

    for item in results:
        label = str(item["rank"])

        citation = (
            f"[{label}] "
            f"{item['file_name']}, "
            f"page {item['page']}, "
            f"chunk {item['chunk_id']}"
        )

        text = clean_snippet(item["text"], max_chars=max_chars_per_chunk)
        blocks.append(f"{citation}\n{text}")

    return "\n\n".join(blocks)


def build_messages(
    question: str,
    answer_candidates_text: str,
    evidence_text: str,
) -> List[Dict]:
    system_message = """You are a careful assistant for the CHRIS RAG system.

You must answer using only the provided evidence.
Do not use outside knowledge.

If the evidence is insufficient, say exactly:
"The retrieved evidence is insufficient to answer this question."

Rules:
- Prefer the answer candidate that most directly answers the user question.
- Usually this is [A1], unless another candidate is clearly more precise.
- Keep the answer concise: one or two sentences maximum.
- Use citations such as [A1], [A2], [1], or [2].
- Every factual claim must have a citation.
- Do not cite a source unless it directly supports the sentence.
- Do not answer with only a citation label or only a source name.
- The final answer must be a natural-language answer to the user question.
- Do not include file names, page numbers, or chunk IDs in the final answer unless explicitly asked.
- Do not invent numbers, dates, policies, links, or study details.
- If sources disagree, mention the disagreement.
- Do not show your reasoning.
- Do not include thinking.
- Do not include hidden reasoning.
- Do not use <think> tags.
- Output only the final answer.
"""

    user_message = f"""User question:
{question}

Answer sentence candidates:
{answer_candidates_text}

Retrieved evidence:
{evidence_text}

Write a concise final answer using only the answer candidates.
Prefer the shortest candidate that directly answers the question.
Cite the selected answer candidate, for example [A1].
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def generate_with_ollama(messages: List[Dict], model_name: str) -> str:
    response = chat(
        model=model_name,
        messages=messages,
        options={
            "temperature": 0,
            "top_p": 0.9,
            "num_predict": 80,
            "num_ctx": 4096,
        },
    )

    return response.message.content


def build_output_object(
    question: str,
    final_answer: str,
    answer_candidates: List[Dict],
    retrieved_evidence: List[Dict],
    model_name: str,
) -> Dict:
    candidates_json = []

    for i, item in enumerate(answer_candidates, start=1):
        source = item["source"]

        candidates_json.append({
            "rank": i,
            "label": f"A{i}",
            "sentence": item["sentence"],
            "score": item["score"],
            "source": {
                "file_name": source["file_name"],
                "collection": source["collection"],
                "page": source["page"],
                "chunk_id": source["chunk_id"],
            },
        })

    evidence_json = []

    for item in retrieved_evidence:
        evidence_json.append({
            "rank": item["rank"],
            "label": str(item["rank"]),
            "rrf_score": item["rrf_score"],
            "bm25_rank": item["bm25_rank"],
            "embedding_rank": item["embedding_rank"],
            "source": {
                "file_name": item["file_name"],
                "collection": item["collection"],
                "page": item["page"],
                "chunk_id": item["chunk_id"],
            },
            "text": clean_snippet(item["text"], max_chars=1200),
        })

    return {
        "question": question,
        "model": model_name,
        "final_answer": final_answer,
        "answer_candidates": candidates_json,
        "retrieved_evidence": evidence_json,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate a grounded RAG answer using local Ollama."
    )
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", type=str, default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional path where the full RAG output will be saved as JSON.",
    )

    args = parser.parse_args()

    print("Retrieving evidence...")
    retrieved_results = retrieve(args.query, top_k=args.top_k)

    print(f"Loading sentence model: {MODEL_NAME}")
    sentence_model = SentenceTransformer(MODEL_NAME)

    answer_candidates = rank_answer_sentences(
        query=args.query,
        results=retrieved_results,
        model=sentence_model,
        top_n=3,
    )

    answer_candidates_text = format_answer_candidates(answer_candidates)
    evidence_text = format_evidence(retrieved_results)

    messages = build_messages(
        question=args.query,
        answer_candidates_text=answer_candidates_text,
        evidence_text=evidence_text,
    )

    print(f"Generating answer with Ollama model: {args.model}")
    final_answer = generate_with_ollama(
        messages=messages,
        model_name=args.model,
    )

    print()
    print("=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)
    print(final_answer)

    print()
    print("=" * 80)
    print("ANSWER CANDIDATES")
    print("=" * 80)
    print(answer_candidates_text)

    if args.json_output is not None:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_object = build_output_object(
            question=args.query,
            final_answer=final_answer,
            answer_candidates=answer_candidates,
            retrieved_evidence=retrieved_results,
            model_name=args.model,
        )

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_object, f, ensure_ascii=False, indent=2)

        print()
        print(f"JSON output written to: {output_path}")


if __name__ == "__main__":
    main()