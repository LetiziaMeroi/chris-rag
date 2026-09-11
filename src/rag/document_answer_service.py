import os
from typing import Dict, List

from ollama import chat

from src.query.document_service import (
    DocumentRetrievalService,
)

from src.rag.evidence_selector import (
    select_evidence,
    select_table_rows,
)
from src.rag.query_rewriter import QueryRewriter


DEFAULT_MODEL = os.getenv(
    "CHRIS_LLM_MODEL",
    "llama3.1:8b",
)


class DocumentAnswerService:

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
    ):
        self.model_name = model_name
        self.retriever = (
            DocumentRetrievalService()
        )
        self.query_rewriter = QueryRewriter(
            model_name=self.model_name
        )

    def _format_text_evidence(
        self,
        results: List[Dict],
    ) -> str:

        blocks = []

        for i, item in enumerate(
            results,
            start=1,
        ):

            source = item["source"]
            label = f"T{i}"

            citation = (
                f"[{label}] "
                f"{source['file_name']}, "
                f"page {source['page']}, "
                f"chunk {source['chunk_id']}"
            )

            blocks.append(
                f"{citation}\n"
                f"{item['text']}"
            )

        return "\n\n".join(blocks)

    def _format_table_evidence(
        self,
        results: List[Dict],
    ) -> str:

        blocks = []

        for i, item in enumerate(
            results,
            start=1,
        ):

            source = item["source"]

            label = f"B{i}"

            citation = (
                f"[{label}] "
                f"{source['file_name']}, "
                f"page {source['page']}, "
                f"table {source.get('table_index')}, "
                f"chunk {source['chunk_id']}"
            )

            caption = item.get(
                "caption"
            )

            caption_text = ""

            if caption:
                caption_text = (
                    f"Caption: {caption}\n"
                )

            blocks.append(
                f"{citation}\n"
                f"{caption_text}"
                f"{item['text']}"
            )

        return "\n\n".join(blocks)

    def _build_messages(
        self,
        question: str,
        text_evidence: str,
        table_evidence: str,
    ):

        system_message = """
        You are the answer-generation component of the CHRIS RAG system.

        Answer the user's question using ONLY the supplied evidence.

        Rules:

        1. Do not use outside knowledge or infer unsupported facts.
        2. Answer only what the user asked.
        3. If the evidence explicitly answers the question, provide that answer
        and do NOT say that the evidence is insufficient.
        4. Do not call something an inference when it is explicitly stated
        in the evidence.
        5. Preserve exact names, numbers, units, devices, protocols, and
        study versions as they appear in the evidence.
        6. Prefer table/codebook evidence for structured information such as
        variables, devices, units, measurements, and protocols.
        7. Every factual claim must include a citation using only the supplied
        labels [T1], [T2], ... and [B1], [B2], ...
        Never include file names, page numbers, table numbers, chunk IDs,
        or other source metadata in the answer.
        Only output the citation label, for example [B3].
        8. Answer in the same language as the user's question.
        9. Keep the answer concise, factual, and non-repetitive.
        10. Do not repeat the user's question in the answer.

        Only when NO supplied evidence directly supports an answer, respond exactly:
        "The retrieved evidence is insufficient to answer this question."
        """
                
        user_message = f"""
        QUESTION
        {question}

        TEXT EVIDENCE
        {text_evidence}

        TABLE / CODEBOOK EVIDENCE
        {table_evidence}

        Answer only the question asked, using the evidence above.
        Use the minimum evidence needed and cite every factual claim.
        """

        return [
            {
                "role": "system",
                "content": system_message.strip(),
            },
            {
                "role": "user",
                "content": user_message.strip(),
            },
        ]

    def answer(self, query: str, text_top_k: int = 10, table_top_k: int = 30,) -> Dict:

        retrieval_query = (
            self.query_rewriter.rewrite_for_retrieval(
                query
            )
        )

        retrieval = self.retriever.search(
            query=retrieval_query,
            table_query=retrieval_query,
            text_top_k=text_top_k,
            table_top_k=table_top_k,
        )

        text_results = retrieval[
            "text_results"
        ]

        table_results = retrieval[
            "table_results"
        ]

        
        text_results = select_evidence(
            query=retrieval_query,
            results=text_results,
            top_n=5,
        )

        table_results = select_table_rows(
            query=retrieval_query,
            results=table_results,
            top_n=12,
        )
        

        text_evidence = (
            self._format_text_evidence(
                text_results
            )
        )

        table_evidence = (
            self._format_table_evidence(
                table_results
            )
        )

        answer_evidence = []

        for i, item in enumerate(
            text_results,
            start=1,
        ):
            answer_evidence.append({
                "label": f"T{i}",
                "type": "text",
                "source": item.get(
                    "source",
                    {},
                ),
                "text": item.get(
                    "text",
                    "",
                ),
            })


        for i, item in enumerate(
            table_results,
            start=1,
        ):
            answer_evidence.append({
                "label": f"B{i}",
                "type": "table",
                "source": item.get(
                    "source",
                    {},
                ),
                "text": item.get(
                    "text",
                    item.get("row", ""),
                ),
                "caption": item.get(
                    "caption",
                    "",
                ),
            })

        messages = self._build_messages(
            question=query,
            text_evidence=text_evidence,
            table_evidence=table_evidence,
        )

        response = chat(
            model=self.model_name,
            messages=messages,
            options={
                "temperature": 0,
                "top_p": 0.9,
                "num_predict": 400,
                "num_ctx": 8192,
            },
        )

        final_answer = (
            response.message.content.strip()
        )

        return {
            "status": "ok",
            "query": query,
            "model": self.model_name,
            "final_answer": final_answer,
            "answer_evidence": answer_evidence,
            "retrieval": retrieval,
        }
