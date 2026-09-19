import argparse
import json
import re

from src.rag.document_answer_service import (
    DocumentAnswerService,
)

from src.query.gwas_natural_language import (
    GWASNaturalLanguageService,
)
from src.router.query_router import route_query


class QueryOrchestrator:

    def __init__(self):
        self.gwas_service = (
            GWASNaturalLanguageService()
        )

        self.document_service = (
            DocumentAnswerService()
        )

    def _format_document_response(
        self,
        query: str,
        result: dict,
    ) -> dict:

        retrieval = result.get(
            "retrieval",
            {},
        )

        evidence = result.get(
            "answer_evidence",
            [],
        )

        final_answer = result.get(
            "final_answer",
            "",
        )

        normalized_answer = final_answer.lower()

        abstention_phrases = [
            "there is no evidence",
            "no evidence in the provided",
            "the provided evidence does not",
            "not enough evidence",
            "insufficient evidence",
            "evidence is insufficient",
            "retrieved evidence is insufficient",
            "cannot answer this question from the provided evidence",
            "cannot answer from the provided evidence",
            "not provide any information",
        ]

        abstained = any(
            phrase in normalized_answer
            for phrase in abstention_phrases
        )

        return {
            "route": "documents",
            "status": (
                "abstention"
                if abstained
                else "ok"
            ),
            "query": query,
            "answer": final_answer,
            "evidence": (
                []
                if abstained
                else evidence
            ),
            "abstained": abstained,
            "metadata": {
                "model": result.get(
                    "model"
                ),
                "retriever": retrieval.get(
                    "retriever"
                ),
            },
        }
    
    def _format_gwas_response(
        self,
        query: str,
        result: dict,
    ) -> dict:

        if result.get("status") != "ok":
            return {
                "route": "gwas",
                "status": result.get(
                    "status",
                    "error",
                ),
                "query": query,
                "answer": result.get(
                    "message"
                ),
                "evidence": [],
                "metadata": {
                    "parsed": result.get(
                        "parsed"
                    ),
                    "missing": result.get(
                        "missing"
                    ),
                },
            }

        gwas_result = result.get(
            "result",
            {},
        )

        rows = gwas_result.get(
            "rows",
            [],
        )

        evidence = []

        for row in rows:
            evidence.append({
                "type": "gwas_variant",
                "variant": {
                    "CHR": row.get("CHR"),
                    "POS": row.get("POS"),
                    "REF": row.get("REF"),
                    "ALT": row.get("ALT"),
                    "P": row.get("P"),
                    "BETA": row.get("BETA"),
                    "SE": row.get("SE"),
                    "ALT_AF": row.get("ALT_AF"),
                    "DIRECTION": row.get(
                        "DIRECTION"
                    ),
                },
            })

        return {
            "route": "gwas",
            "status": "ok",
            "query": query,
            "answer": result.get(
                "answer"
            ),
            "evidence": evidence,
            "metadata": {
                "parsed": result.get(
                    "parsed"
                ),
                "dataset": gwas_result.get(
                    "dataset"
                ),
                "filters": gwas_result.get(
                    "filters"
                ),
                "count": gwas_result.get(
                    "count"
                ),
            },
        }
    def run(self, query: str):

        routing = route_query(query)

        # =================================================
        # GWAS route
        # =================================================

        if routing.route == "gwas":

            result = self.gwas_service.run(
                query=query
            )

            return self._format_gwas_response(
                query=query,
                result=result,
            )

        # =================================================
        # Document route
        # =================================================

        if routing.route == "documents":

            result = self.document_service.answer(
                query=query
            )

            return self._format_document_response(
                query=query,
                result=result,
            )

        
def print_result(result):

    print("\n" + "=" * 80)
    print("CHRIS QUERY ORCHESTRATOR")
    print("=" * 80)

    print(
        f"Query: {result['query']}"
    )

    print(
        f"Route: {result['route']}"
    )

    print(
        f"Status: {result['status']}"
    )

    if result.get("answer"):
        print("\nAnswer:")
        print(
            result["answer"]
        )

    print("\nMetadata:")
    print(
        json.dumps(
            result.get(
                "metadata",
                {},
            ),
            indent=2,
            default=str,
        )
    )

    print("\nEvidence:")

    print(
        json.dumps(
            result.get(
                "evidence",
                [],
            ),
            indent=2,
            default=str,
        )
    )

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Route CHRIS natural-language "
            "queries to the appropriate backend."
        )
    )

    parser.add_argument(
        "query",
        type=str,
    )

    args = parser.parse_args()

    orchestrator = QueryOrchestrator()

    result = orchestrator.run(
        args.query
    )

    print_result(result)


if __name__ == "__main__":
    main()
