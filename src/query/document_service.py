from typing import Dict, List

from src.retrieval.search_docling import search


class DocumentRetrievalService:
    """
    Retrieve evidence from both Docling layers:

    - text: narrative paragraphs
    - table: tables, codebooks and structured document data

    The two rankings are deliberately kept separate.
    """

    def __init__(
        self,
        default_text_top_k: int = 5,
        default_table_top_k: int = 5,
    ):
        self.default_text_top_k = default_text_top_k
        self.default_table_top_k = default_table_top_k

    def _format_results(
        self,
        results: List[Dict],
        layer: str,
    ) -> List[Dict]:

        formatted = []

        for item in results:

            source = {
                "file_name": item.get("file_name"),
                "collection": item.get("collection"),
                "page": item.get("page"),
                "pages": item.get("pages", []),
                "chunk_id": item.get("chunk_id"),
            }

            if layer == "table":
                source["table_index"] = item.get(
                    "table_index"
                )

            result = {
                "rank": item["rank"],
                "layer": layer,
                "rrf_score": item["rrf_score"],
                "bm25_rank": item["bm25_rank"],
                "embedding_rank": item["embedding_rank"],
                "embedding_score": item["embedding_score"],
                "source": source,
                "text": item["text"],
            }

            if layer == "table":
                result["caption"] = item.get(
                    "caption"
                )

            formatted.append(result)

        return formatted

    def search(
        self,
        query: str,
        text_top_k: int = None,
        table_top_k: int = None,
        table_query: str = None,
    ) -> Dict:

        if table_query is None:
            table_query = query

        if text_top_k is None:
            text_top_k = self.default_text_top_k

        if table_top_k is None:
            table_top_k = self.default_table_top_k

        # -------------------------------------------------
        # Narrative document retrieval
        # -------------------------------------------------

        text_results_raw = search(
            query=query,
            layer="text",
            top_k=text_top_k,
        )

        # -------------------------------------------------
        # Table / codebook retrieval
        # -------------------------------------------------

        table_results_raw = search(
            query=table_query,
            layer="table",
            top_k=table_top_k,
        )

        text_results = self._format_results(
            text_results_raw,
            layer="text",
        )

        table_results = self._format_results(
            table_results_raw,
            layer="table",
        )

        return {
            "status": "ok",
            "query": query,
            "retriever": "docling_hybrid_rrf_e5_base",
            "text_result_count": len(text_results),
            "table_result_count": len(table_results),
            "text_results": text_results,
            "table_results": table_results,
        }