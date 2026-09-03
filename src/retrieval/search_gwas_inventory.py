import argparse
import re
from pathlib import Path
from typing import List

import pandas as pd


DATA_ROOT = Path("/storage/data/chris-rag")
INVENTORY_PATH = DATA_ROOT / "processed" / "gwas_inventory.csv"


SEARCH_COLUMNS = [
    "file_name",
    "relative_path",
    "extension",
    "trait_from_filename",
    "columns",
    "col_variant_id",
    "col_chromosome",
    "col_position",
    "col_p_value",
    "col_beta",
    "col_standard_error",
]


def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[_\-.]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_query(query: str) -> List[str]:
    query = normalize_text(query)
    return [token for token in query.split() if token]


def score_row(row: pd.Series, query_terms: List[str]) -> int:
    searchable_text = " ".join(
        normalize_text(row.get(column, ""))
        for column in SEARCH_COLUMNS
    )

    score = 0

    for term in query_terms:
        if term in searchable_text:
            score += 1

    return score


def search_inventory(query: str, top_k: int = 20) -> pd.DataFrame:
    if not INVENTORY_PATH.exists():
        raise FileNotFoundError(f"Inventory not found: {INVENTORY_PATH}")

    df = pd.read_csv(INVENTORY_PATH)

    query_terms = tokenize_query(query)

    if not query_terms:
        return df.head(top_k)

    df = df.copy()
    df["score"] = df.apply(lambda row: score_row(row, query_terms), axis=1)

    results = df[df["score"] > 0].sort_values(
        by=["score", "size_mb"],
        ascending=[False, False],
    )

    return results.head(top_k)


def print_results(results: pd.DataFrame) -> None:
    if results.empty:
        print("No matching GWAS inventory records found.")
        return

    display_columns = [
        "score",
        "file_name",
        "extension",
        "size_mb",
        "num_rows_estimate",
        "num_columns",
        "trait_from_filename",
        "col_variant_id",
        "col_chromosome",
        "col_position",
        "col_p_value",
        "col_beta",
        "col_standard_error",
    ]

    available_columns = [
        column for column in display_columns
        if column in results.columns
    ]

    print(results[available_columns].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Search the GWAS inventory."
    )
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=20)

    args = parser.parse_args()

    results = search_inventory(args.query, top_k=args.top_k)
    print_results(results)


if __name__ == "__main__":
    main()
