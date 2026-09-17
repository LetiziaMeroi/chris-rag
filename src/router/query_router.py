import re
from dataclasses import dataclass


@dataclass
class RouteResult:
    route: str
    confidence: float
    reason: str


GWAS_KEYWORDS = [
    "gwas",
    "variant",
    "variants",
    "p-value",
    "p value",
    "genome-wide",
    "genome wide",
    "chromosome",
    "allele",
    "beta",
    "effect size",
    "association",
    "associated variant",
    "significant variant",
    "significant variants",
    "snp",
    "snps",
    "ref allele",
    "alt allele",
    "minor allele",
    "allele frequency",
]


def route_query(query: str) -> RouteResult:
    """
    Simple deterministic router.

    Routes:
        - gwas
        - documents
    """

    text = query.lower().strip()

    matches = []

    for keyword in GWAS_KEYWORDS:
        if keyword in text:
            matches.append(keyword)

    # Genomic coordinate patterns such as:
    # chr1:70000000-72000000
    # chromosome X
    # chr 6
    genomic_pattern = re.search(
        r"\bchr(?:omosome)?\s*[0-9xy]+"
        r"|\bchr[0-9xy]+[:\s]"
        r"|\b[0-9xy]+:\d+[-:]\d+",
        text,
        flags=re.IGNORECASE,
    )

    if genomic_pattern:
        matches.append("genomic_coordinate")

    # P-value style expressions
    pvalue_pattern = re.search(
        r"\bp\s*[<=>]\s*[0-9.eE+-]+",
        text
    )

    if pvalue_pattern:
        matches.append("p_value_expression")

    if matches:
        confidence = min(
            0.60 + 0.10 * len(matches),
            0.99
        )

        return RouteResult(
            route="gwas",
            confidence=confidence,
            reason=(
                "GWAS indicators detected: "
                + ", ".join(sorted(set(matches)))
            ),
        )

    return RouteResult(
        route="documents",
        confidence=0.70,
        reason="No explicit GWAS indicators detected.",
    )


if __name__ == "__main__":
    examples = [
        "What did we learn about Parkinson's disease in CHRIS?",
        "Which variants are associated with peppermint?",
        "Show genome-wide significant variants for banana",
        "What was measured at baseline?",
        "Which variants on chromosome 1 have p < 5e-8?",
        "How was blood pressure measured?",
        "What is the strongest SNP association for SCORE?",
    ]

    for query in examples:
        result = route_query(query)

        print("\nQuery:")
        print(query)

        print("Route:")
        print(result.route)

        print("Confidence:")
        print(result.confidence)

        print("Reason:")
        print(result.reason)
