import argparse
import re
from pathlib import Path

import pandas as pd


KNOWN_COHORTS = [
    "CHRIS",
    "LIFE",
    "Rhineland",
    "ARIC",
]


def strip_known_extension(filename: str) -> tuple[str, str]:
    """
    Remove the known GWAS extension and return:
        stem, normalized format
    """

    extensions = [
        (".DHN.gz", "DHN"),
        (".TBL.gz", "TBL"),
        (".TBL", "TBL"),
        (".txt", "TXT"),
    ]

    for suffix, file_format in extensions:
        if filename.endswith(suffix):
            return filename[:-len(suffix)], file_format

    return filename, "UNKNOWN"


def parse_build(stem: str):
    """
    Extract genome build such as GRCh37 or GRCh38.
    """
    match = re.search(r"_(GRCh\d+)$", stem)

    if match:
        return match.group(1)

    return None


def remove_build(stem: str):
    """
    Remove final _GRCh37 / _GRCh38 from filename stem.
    """
    return re.sub(r"_GRCh\d+$", "", stem)


def parse_sex(stem: str):
    """
    Extract sex from filenames such as:
        ..._all1_GRCh37
        ..._female1_GRCh38
        ..._male1_GRCh37
    """

    # Remove genome build before looking for sex suffix
    working = remove_build(stem)

    match = re.search(
        r"_(all|female|male)\d*$",
        working,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return match.group(1).lower()


def remove_sex(stem: str):
    """
    Remove sex suffix from filename stem.
    """
    return re.sub(
        r"_(all|female|male)\d*$",
        "",
        stem,
        flags=re.IGNORECASE,
    )


def parse_cohorts(stem: str):
    """
    Detect known cohorts from the filename.
    """
    cohorts = []

    for cohort in KNOWN_COHORTS:
        if re.search(
            rf"(^|_){re.escape(cohort)}(_|$)",
            stem,
            flags=re.IGNORECASE,
        ):
            cohorts.append(cohort)

    return cohorts


def parse_trait(stem: str):
    """
    Extract trait after the cohort/population prefix.

    Expected filename pattern example:

    CHRIS_EUR_LIFE_EUR_Rhineland_EUR_ARIC_EUR_banana_female1_GRCh37
                                            ^^^^^^
                                            trait

    Strategy:
      1. remove build
      2. remove sex suffix
      3. locate final cohort marker ARIC_EUR_
      4. everything after it is the trait
    """

    working = remove_build(stem)
    working = remove_sex(working)

    marker = "_ARIC_EUR_"

    if marker in working:
        trait = working.split(marker, 1)[1]

        if trait:
            return trait

    return None


def make_dataset_id(trait, sex, build, file_format):
    """
    Create stable human-readable identifier.
    """

    parts = [
        value
        for value in [
            trait,
            sex,
            build,
            file_format,
        ]
        if value
    ]

    return "__".join(parts)


def normalize_inventory(input_path: Path, output_path: Path):

    print(f"Reading inventory: {input_path}")

    df = pd.read_csv(input_path)

    required_columns = {
        "file_name",
        "file_path",
        "extension",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Inventory is missing required columns: {sorted(missing)}"
        )

    normalized_rows = []

    for _, row in df.iterrows():

        filename = str(row["file_name"])

        stem, file_format = strip_known_extension(filename)

        build = parse_build(stem)
        sex = parse_sex(stem)
        trait = parse_trait(stem)
        cohorts = parse_cohorts(stem)

        normalized = {
            # -------------------------
            # Identity
            # -------------------------
            "dataset_id": make_dataset_id(
                trait,
                sex,
                build,
                file_format,
            ),

            "file_name": filename,
            "file_path": row.get("file_path"),
            "relative_path": row.get("relative_path"),

            # -------------------------
            # Normalized metadata
            # -------------------------
            "trait": trait,
            "sex": sex,
            "genome_build": build,
            "format": file_format,

            "cohorts": "|".join(cohorts),
            "num_cohorts": len(cohorts),

            # -------------------------
            # File information
            # -------------------------
            "extension": row.get("extension"),
            "size_mb": row.get("size_mb"),
            "is_compressed": row.get("is_compressed"),
            "readable": row.get("readable"),
            "num_rows_estimate": row.get("num_rows_estimate"),
            "num_columns": row.get("num_columns"),

            # -------------------------
            # Schema
            # -------------------------
            "columns": row.get("columns"),

            "col_chromosome": row.get("col_chromosome"),
            "col_position": row.get("col_position"),
            "col_variant_id": row.get("col_variant_id"),

            "col_effect_allele": row.get("col_effect_allele"),
            "col_other_allele": row.get("col_other_allele"),

            "col_beta": row.get("col_beta"),
            "col_standard_error": row.get("col_standard_error"),
            "col_p_value": row.get("col_p_value"),

            "col_sample_size": row.get("col_sample_size"),
            "col_frequency": row.get("col_frequency"),

            # -------------------------
            # Provenance / QA
            # -------------------------
            "md5": row.get("md5"),
            "read_error": row.get("read_error"),
        }

        normalized_rows.append(normalized)

    result = pd.DataFrame(normalized_rows)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print(f"\nSaved normalized catalog to:")
    print(output_path)

    # ------------------------------------------------------
    # QA summary
    # ------------------------------------------------------

    print("\n" + "=" * 80)
    print("GWAS CATALOG SUMMARY")
    print("=" * 80)

    print(f"Files: {len(result)}")

    print("\nFormats:")
    print(
        result["format"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nGenome builds:")
    print(
        result["genome_build"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nSex:")
    print(
        result["sex"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nTraits:")
    print(
        result["trait"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    # Rows where parsing failed
    parsing_issues = result[
        (
            result["format"].isin(["DHN", "TBL"])
        )
        &
        (
            result["trait"].isna()
            | result["sex"].isna()
        )
    ]

    if len(parsing_issues) > 0:

        print("\n" + "!" * 80)
        print("FILES WITH METADATA PARSING ISSUES")
        print("!" * 80)

        print(
            parsing_issues[
                [
                    "file_name",
                    "trait",
                    "sex",
                    "genome_build",
                    "format",
                ]
            ].to_string(index=False)
        )

    else:

        print(
            "\nNo metadata parsing problems "
            "detected for GWAS result files."
        )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Normalize CHRIS GWAS inventory metadata."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to gwas_inventory.csv",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/gwas/gwas_catalog.csv"
        ),
    )

    args = parser.parse_args()

    normalize_inventory(
        args.input,
        args.output,
    )


if __name__ == "__main__":
    main()
