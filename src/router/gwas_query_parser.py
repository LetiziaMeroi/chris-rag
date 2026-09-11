import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pandas as pd


import os

GWAS_ROOT = Path(
    os.getenv(
        "CHRIS_GWAS_ROOT",
        "data/processed/gwas",
    )
)

DEFAULT_CATALOG = (
    GWAS_ROOT
    / "gwas_catalog.csv"
)


@dataclass
class GWASQuery:
    trait: Optional[str] = None
    sex: Optional[str] = None
    build: Optional[str] = None

    p_max: Optional[float] = None

    chromosome: Optional[str] = None
    pos_min: Optional[int] = None
    pos_max: Optional[int] = None

    ref: Optional[str] = None
    alt: Optional[str] = None

    af_min: Optional[float] = None
    af_max: Optional[float] = None

    limit: int = 20

    def to_dict(self):
        return asdict(self)


class GWASQueryParser:

    def __init__(
        self,
        catalog_path: Path = DEFAULT_CATALOG,
    ):
        self.catalog_path = Path(catalog_path)

        if not self.catalog_path.exists():
            raise FileNotFoundError(
                f"GWAS catalog not found: "
                f"{self.catalog_path}"
            )

        catalog = pd.read_csv(
            self.catalog_path
        )

        self.traits = sorted(
            catalog["trait"]
            .dropna()
            .astype(str)
            .unique(),
            key=len,
            reverse=True,
        )

    # =====================================================
    # Trait
    # =====================================================

    def parse_trait(self, text: str):
        lower = text.lower()

        for trait in self.traits:
            pattern = (
                r"(?<![a-zA-Z0-9])"
                + re.escape(trait.lower())
                + r"(?![a-zA-Z0-9])"
            )

            if re.search(pattern, lower):
                return trait

        return None

    # =====================================================
    # Sex
    # =====================================================

    def parse_sex(self, text: str):
        lower = text.lower()

        female_patterns = [
            r"\bfemale\b",
            r"\bfemales\b",
            r"\bwomen\b",
            r"\bwoman\b",
            r"\bdonne\b",
            r"\bfemminile\b",
        ]

        male_patterns = [
            r"\bmale\b",
            r"\bmales\b",
            r"\bmen\b",
            r"\bman\b",
            r"\buomini\b",
            r"\bmaschile\b",
        ]

        all_patterns = [
            r"\ball\b",
            r"\bboth sexes\b",
            r"\bcombined\b",
            r"\btutti\b",
            r"\bentrambi i sessi\b",
        ]

        for pattern in female_patterns:
            if re.search(pattern, lower):
                return "female"

        for pattern in male_patterns:
            if re.search(pattern, lower):
                return "male"

        for pattern in all_patterns:
            if re.search(pattern, lower):
                return "all"

        return None

    # =====================================================
    # Genome build
    # =====================================================

    def parse_build(self, text: str):
        match = re.search(
            r"\bGRCh\s*(37|38)\b",
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return f"GRCh{match.group(1)}"

        # Common aliases
        lower = text.lower()

        if re.search(r"\bhg19\b", lower):
            return "GRCh37"

        if re.search(r"\bhg38\b", lower):
            return "GRCh38"

        return None

    # =====================================================
    # P-value
    # =====================================================

    def parse_p_max(self, text: str):
        lower = text.lower()

        # Conventional genome-wide significance
        genome_wide_patterns = [
            "genome-wide significant",
            "genome wide significant",
            "genome-wide significance",
            "genome wide significance",
            "significatività genome-wide",
            "significativita genome-wide",
        ]

        if any(
            phrase in lower
            for phrase in genome_wide_patterns
        ):
            return 5e-8

        # Examples:
        # p < 5e-8
        # p <= 1e-6
        # p-value < 0.0001
        match = re.search(
            r"\bp(?:-?value)?\s*"
            r"(?:<=|<|=)\s*"
            r"([0-9]*\.?[0-9]+"
            r"(?:e[+-]?\d+)?)",
            lower,
        )

        if match:
            try:
                return float(
                    match.group(1)
                )
            except ValueError:
                pass

        return None

    # =====================================================
    # Chromosome
    # =====================================================

    def parse_chromosome(self, text: str):
        patterns = [
            r"\bchromosome\s+([0-9]{1,2}|x|y|mt)\b",
            r"\bcromosoma\s+([0-9]{1,2}|x|y|mt)\b",
            r"\bchr\s*([0-9]{1,2}|x|y|mt)\b",
            r"\bchr([0-9]{1,2}|x|y|mt)\b",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                return (
                    match.group(1)
                    .upper()
                )

        return None

    # =====================================================
    # Genomic region
    # =====================================================

    def parse_region(self, text: str):
        """
        Examples:

        chr1:70000000-72000000
        1:70000000-72000000
        chromosome 1 from 70000000 to 72000000
        """

        # chr1:70000000-72000000
        match = re.search(
            r"(?:chr)?"
            r"([0-9]{1,2}|x|y|mt)"
            r"\s*:\s*"
            r"(\d+)"
            r"\s*[-:]\s*"
            r"(\d+)",
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return {
                "chromosome": (
                    match.group(1).upper()
                ),
                "pos_min": int(
                    match.group(2)
                ),
                "pos_max": int(
                    match.group(3)
                ),
            }

        # chromosome 1 from ... to ...
        match = re.search(
            r"(?:chromosome|cromosoma|chr)"
            r"\s*"
            r"([0-9]{1,2}|x|y|mt)"
            r".*?"
            r"(?:from|da)"
            r"\s+(\d+)"
            r"\s+(?:to|a)\s+"
            r"(\d+)",
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return {
                "chromosome": (
                    match.group(1).upper()
                ),
                "pos_min": int(
                    match.group(2)
                ),
                "pos_max": int(
                    match.group(3)
                ),
            }

        return None

    # =====================================================
    # Specific position
    # =====================================================

    def parse_position(self, text: str):
        """
        Example:
        chr1:71355857
        """

        match = re.search(
            r"\b(?:chr)?"
            r"([0-9]{1,2}|x|y|mt)"
            r"\s*:\s*"
            r"(\d+)\b",
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return {
                "chromosome": (
                    match.group(1).upper()
                ),
                "position": int(
                    match.group(2)
                ),
            }

        return None

    # =====================================================
    # REF / ALT
    # =====================================================

    def parse_alleles(self, text: str):

        ref = None
        alt = None

        ref_match = re.search(
            r"\bref(?:erence)?"
            r"(?:\s+allele)?"
            r"\s*[=:]?\s*"
            r"([ACGT]+)\b",
            text,
            flags=re.IGNORECASE,
        )

        if ref_match:
            ref = (
                ref_match.group(1)
                .upper()
            )

        alt_match = re.search(
            r"\balt(?:ernate)?"
            r"(?:\s+allele)?"
            r"\s*[=:]?\s*"
            r"([ACGT]+)\b",
            text,
            flags=re.IGNORECASE,
        )

        if alt_match:
            alt = (
                alt_match.group(1)
                .upper()
            )

        return ref, alt

    # =====================================================
    # Limit
    # =====================================================

    def parse_limit(self, text: str):
        """
        Examples:
        top 10 variants
        first 50 variants
        show 100 variants
        """

        match = re.search(
            r"\b(?:top|first|show|mostra)"
            r"\s+(\d+)\b",
            text,
            flags=re.IGNORECASE,
        )

        if match:
            value = int(
                match.group(1)
            )

            # Safety cap
            return min(
                max(value, 1),
                1000,
            )

        return 20

    # =====================================================
    # Full parsing
    # =====================================================

    def parse(self, query: str):

        result = GWASQuery()

        result.trait = (
            self.parse_trait(query)
        )

        result.sex = (
            self.parse_sex(query)
        )

        result.build = (
            self.parse_build(query)
        )

        result.p_max = (
            self.parse_p_max(query)
        )

        result.chromosome = (
            self.parse_chromosome(query)
        )

        # Region has priority
        region = self.parse_region(
            query
        )

        if region:
            result.chromosome = (
                region["chromosome"]
            )

            result.pos_min = (
                region["pos_min"]
            )

            result.pos_max = (
                region["pos_max"]
            )

        else:
            position = (
                self.parse_position(
                    query
                )
            )

            if position:
                result.chromosome = (
                    position["chromosome"]
                )

                result.pos_min = (
                    position["position"]
                )

                result.pos_max = (
                    position["position"]
                )

        result.ref, result.alt = (
            self.parse_alleles(query)
        )

        result.limit = (
            self.parse_limit(query)
        )

        return result


if __name__ == "__main__":

    parser = GWASQueryParser()

    examples = [
        (
            "Show genome-wide significant variants "
            "for peppermint in females using GRCh38"
        ),

        (
            "Top 10 variants for banana in males "
            "on chromosome 1 with p < 1e-6"
        ),

        (
            "Quali varianti significative ci sono "
            "per SCORE nelle donne su GRCh37?"
        ),

        (
            "Show peppermint variants in "
            "chr1:70000000-72000000 "
            "for female GRCh38"
        ),

        (
            "Find SCORE variant chrX:125808669 "
            "for all using hg19"
        ),
    ]

    for example in examples:

        result = parser.parse(
            example
        )

        print("\n" + "=" * 80)
        print(example)
        print("-" * 80)

        print(
            result.to_dict()
        )
