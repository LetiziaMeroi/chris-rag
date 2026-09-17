import argparse
import json

from src.query.gwas_service import GWASService
from src.router.gwas_query_parser import GWASQueryParser


class GWASNaturalLanguageService:
    def __init__(self):
        self.parser = GWASQueryParser()
        self.service = GWASService()

    def _build_answer(
        self,
        result: dict,
    ) -> str:
        """
        Build a deterministic natural-language summary
        from the structured GWAS query result.

        No LLM is used here.
        """

        dataset = result.get(
            "dataset",
            {},
        )

        count = result.get(
            "count",
            0,
        )

        rows = result.get(
            "rows",
            [],
        )

        trait = dataset.get(
            "trait",
            "unknown",
        )

        sex = dataset.get(
            "sex",
            "unknown",
        )

        build = dataset.get(
            "build",
            "unknown",
        )

        # No matching variants
        if count == 0:
            return (
                "No GWAS variants matched the requested filters "
                f"for trait {trait}, sex {sex}, build {build}."
            )

        answer = (
            f"Found {count:,} GWAS variants matching the requested filters "
            f"for trait {trait}, sex {sex}, build {build}."
        )

        # Add one representative top result
        if rows:

            top = rows[0]

            chromosome = top.get("CHR")
            position = top.get("POS")
            ref = top.get("REF")
            alt = top.get("ALT")
            p_value = top.get("P")
            beta = top.get("BETA")
            se = top.get("SE")

            answer += (
                " The top returned variant is "
                f"{chromosome}:{position} "
                f"{ref}>{alt}"
            )

            if p_value is not None:
                answer += (
                    f" with P={p_value:.3e}"
                )

            if beta is not None:
                answer += (
                    f", BETA={beta}"
                )

            if se is not None:
                answer += (
                    f", SE={se}"
                )

            answer += "."

        return answer

    def run(self, query: str):
        parsed = self.parser.parse(query)

        lower_query = query.lower()

        significance_terms = [
            "significant",
            "significance",
            "significative",
            "significativi",
            "significativa",
            "significativo",
            "signifikant",
        ]

        significance_requested = any(
            term in lower_query
            for term in significance_terms
        )

        if significance_requested and parsed.p_max is None:
            return {
                "status": "missing_parameters",
                "query": query,
                "parsed": parsed.to_dict(),
                "missing": ["p_max"],
                "message": (
                    "The query asks for significant variants, "
                    "but no significance threshold was specified. "
                    "Specify a threshold such as p < 5e-8, "
                    "or explicitly ask for genome-wide significant variants."
                ),
            }

        missing = []

        if parsed.trait is None:
            missing.append("trait")

        if parsed.sex is None:
            missing.append("sex")

        if parsed.build is None:
            missing.append("build")

        if missing:
            return {
                "status": "missing_parameters",
                "query": query,
                "parsed": parsed.to_dict(),
                "missing": missing,
                "message": (
                    "Missing required GWAS parameters: "
                    + ", ".join(missing)
                ),
            }

        result = self.service.query_variants(
            trait=parsed.trait,
            sex=parsed.sex,
            build=parsed.build,
            p_max=parsed.p_max,
            chromosome=parsed.chromosome,
            pos_min=parsed.pos_min,
            pos_max=parsed.pos_max,
            ref=parsed.ref,
            alt=parsed.alt,
            af_min=parsed.af_min,
            af_max=parsed.af_max,
            limit=parsed.limit,
        )

        return {
            "status": "ok",
            "query": query,
            "parsed": parsed.to_dict(),
            "answer": self._build_answer(
                result=result,
            ),
            "result": result,
        }


def print_result(response):
    print("\n" + "=" * 80)
    print("GWAS NATURAL LANGUAGE QUERY")
    print("=" * 80)

    print(f"Query: {response['query']}")

    print("\nParsed parameters:")
    print(
        json.dumps(
            response["parsed"],
            indent=2,
        )
    )

    if response["status"] != "ok":
        print("\nCannot execute query.")
        print(response["message"])
        return

    result = response["result"]

    dataset = result["dataset"]

    print("\nDataset:")
    print(f"  Trait:        {dataset['trait']}")
    print(f"  Sex:          {dataset['sex']}")
    print(f"  Genome build: {dataset['build']}")
    print(f"  Source:       {dataset['file_name']}")

    print("\nFilters:")

    if result["filters"]:
        for key, value in result["filters"].items():
            print(f"  {key}: {value}")
    else:
        print("  none")

    print(
        f"\nMatching variants: "
        f"{result['count']:,}"
    )

    rows = result["rows"]

    if not rows:
        print("\nNo matching variants.")
        return

    print("\nTop results:")

    for i, row in enumerate(rows, start=1):

        print(
            f"{i:>3}. "
            f"{row['CHR']}:{row['POS']} "
            f"{row['REF']}>{row['ALT']} "
            f"P={row['P']:.3e} "
            f"BETA={row['BETA']} "
            f"SE={row['SE']} "
            f"AF={row['ALT_AF']} "
            f"DIRECTION={row['DIRECTION']}"
        )


def main():
    argparser = argparse.ArgumentParser(
        description=(
            "Run a natural-language query "
            "against CHRIS GWAS datasets."
        )
    )

    argparser.add_argument(
        "query",
        type=str,
        help="Natural-language GWAS question",
    )

    args = argparser.parse_args()

    service = GWASNaturalLanguageService()

    response = service.run(
        args.query
    )

    print_result(response)


if __name__ == "__main__":
    main()
