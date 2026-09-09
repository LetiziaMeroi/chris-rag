import argparse
import ast
import gzip
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


DATA_ROOT = Path("/storage/data/chris-rag")
INVENTORY_PATH = DATA_ROOT / "processed" / "gwas_inventory.csv"


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[_\-.]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_inventory() -> pd.DataFrame:
    if not INVENTORY_PATH.exists():
        raise FileNotFoundError(f"GWAS inventory not found: {INVENTORY_PATH}")

    return pd.read_csv(INVENTORY_PATH)


def resolve_file(file_query: str, inventory: pd.DataFrame) -> pd.Series:
    """
    Resolve a file by exact file_name first, then by substring search.
    """
    exact = inventory[inventory["file_name"] == file_query]

    if len(exact) == 1:
        return exact.iloc[0]

    normalized_query = normalize_text(file_query)

    matches = inventory[
        inventory["file_name"].apply(lambda x: normalized_query in normalize_text(x))
        | inventory["trait_from_filename"].apply(lambda x: normalized_query in normalize_text(x))
    ]

    if len(matches) == 0:
        raise ValueError(f"No GWAS file found for query: {file_query}")

    if len(matches) > 1:
        print("Multiple files matched your query. Please use a more specific file name.")
        print()
        print(
            matches[
                [
                    "file_name",
                    "size_mb",
                    "num_rows_estimate",
                    "trait_from_filename",
                ]
            ].head(20).to_string(index=False)
        )
        raise ValueError(f"Ambiguous file query: {file_query}")

    return matches.iloc[0]


def parse_inventory_delimiter(value: str) -> Optional[str]:
    """
    The inventory stores delimiters using repr(), e.g. "'\\t'" or "','".
    Convert them back to real delimiters.
    """
    if pd.isna(value):
        return None

    value = str(value)

    try:
        delimiter = ast.literal_eval(value)
    except Exception:
        delimiter = value

    if delimiter == "":
        return None

    return delimiter


def get_sep_from_inventory(row: pd.Series):
    delimiter = parse_inventory_delimiter(row.get("delimiter", ""))

    if delimiter == "\t":
        return "\t"

    if delimiter == ",":
        return ","

    if delimiter == ";":
        return ";"

    if delimiter == " ":
        return r"\s+"

    return "\t"


def open_maybe_gzip(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")

    return path.open("r", encoding="utf-8", errors="replace")


def read_header(path: Path, sep) -> list:
    if sep == r"\s+":
        df = pd.read_csv(path, sep=sep, nrows=0, engine="python")
    else:
        df = pd.read_csv(path, sep=sep, nrows=0)

    return list(df.columns)


def show_columns(row: pd.Series) -> None:
    print("=" * 80)
    print("GWAS FILE")
    print("=" * 80)
    print(f"File: {row['file_name']}")
    print(f"Path: {row['file_path']}")
    print(f"Trait/name: {row.get('trait_from_filename', '')}")
    print(f"Rows estimate: {row.get('num_rows_estimate', '')}")
    print(f"Columns: {row.get('num_columns', '')}")
    print(f"Size MB: {row.get('size_mb', '')}")
    print()

    columns = str(row.get("columns", "")).split("|")

    print("Detected columns:")
    for column in columns:
        if column.strip():
            print(f"- {column}")

    print()
    print("Semantic GWAS column mapping:")
    semantic_cols = [
        "col_variant_id",
        "col_chromosome",
        "col_position",
        "col_p_value",
        "col_beta",
        "col_standard_error",
        "col_effect_allele",
        "col_other_allele",
        "col_sample_size",
        "col_frequency",
    ]

    for col in semantic_cols:
        if col in row.index:
            value = row.get(col, "")
            if pd.notna(value) and str(value).strip():
                print(f"- {col}: {value}")


def show_head(row: pd.Series, n: int) -> None:
    path = Path(row["file_path"])
    sep = get_sep_from_inventory(row)

    print(f"Reading first {n} rows from: {path.name}")

    if sep == r"\s+":
        df = pd.read_csv(path, sep=sep, nrows=n, engine="python")
    else:
        df = pd.read_csv(path, sep=sep, nrows=n)

    print(df.to_string(index=False))


def get_column(row: pd.Series, semantic_column: str) -> str:
    value = row.get(semantic_column, "")

    if pd.isna(value) or not str(value).strip():
        raise ValueError(f"Missing required semantic column in inventory: {semantic_column}")

    return str(value)


def top_p_values(
    row: pd.Series,
    n: int,
    chunksize: int = 500_000,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    path = Path(row["file_path"])
    sep = get_sep_from_inventory(row)

    p_col = get_column(row, "col_p_value")

    useful_cols = []

    for semantic_col in [
        "col_chromosome",
        "col_position",
        "col_variant_id",
        "col_effect_allele",
        "col_other_allele",
        "col_beta",
        "col_standard_error",
        "col_p_value",
        "col_sample_size",
        "col_frequency",
    ]:
        value = row.get(semantic_col, "")

        if pd.notna(value) and str(value).strip():
            useful_cols.append(str(value))

    # Remove duplicates while preserving order.
    useful_cols = list(dict.fromkeys(useful_cols))

    print(f"Scanning file for top {n} variants by smallest p-value...")
    print(f"File: {path.name}")
    print(f"P-value column: {p_col}")
    print(f"Chunk size: {chunksize}")
    if max_rows is not None:
        print(f"Max rows for this run: {max_rows}")

    best = pd.DataFrame()
    rows_seen = 0

    read_csv_kwargs = {
        "sep": sep,
        "chunksize": chunksize,
        "usecols": useful_cols,
    }

    if sep == r"\s+":
        read_csv_kwargs["engine"] = "python"

    for chunk_idx, chunk in enumerate(pd.read_csv(path, **read_csv_kwargs), start=1):
        chunk[p_col] = pd.to_numeric(chunk[p_col], errors="coerce")
        chunk = chunk.dropna(subset=[p_col])

        best = pd.concat([best, chunk], ignore_index=True)
        best = best.nsmallest(n, p_col)

        rows_seen += len(chunk)

        if chunk_idx % 10 == 0:
            print(f"Processed chunks: {chunk_idx}, valid rows seen: {rows_seen}")

        if max_rows is not None and rows_seen >= max_rows:
            break

    return best.sort_values(p_col)


def find_variant(
    row: pd.Series,
    variant: str,
    chunksize: int = 500_000,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    path = Path(row["file_path"])
    sep = get_sep_from_inventory(row)

    variant_col = get_column(row, "col_variant_id")

    print(f"Searching for variant {variant} in {path.name}")
    print(f"Variant column: {variant_col}")

    matches = []
    rows_seen = 0

    read_csv_kwargs = {
        "sep": sep,
        "chunksize": chunksize,
    }

    if sep == r"\s+":
        read_csv_kwargs["engine"] = "python"

    for chunk_idx, chunk in enumerate(pd.read_csv(path, **read_csv_kwargs), start=1):
        mask = chunk[variant_col].astype(str) == str(variant)
        found = chunk[mask]

        if not found.empty:
            matches.append(found)

        rows_seen += len(chunk)

        if chunk_idx % 10 == 0:
            print(f"Processed chunks: {chunk_idx}, rows seen: {rows_seen}")

        if max_rows is not None and rows_seen >= max_rows:
            break

    if not matches:
        return pd.DataFrame()

    return pd.concat(matches, ignore_index=True)


def find_chr_pos(
    row: pd.Series,
    chromosome: str,
    position: int,
    chunksize: int = 500_000,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    path = Path(row["file_path"])
    sep = get_sep_from_inventory(row)

    chr_col = get_column(row, "col_chromosome")
    pos_col = get_column(row, "col_position")

    print(f"Searching for CHR={chromosome}, POS={position} in {path.name}")
    print(f"CHR column: {chr_col}")
    print(f"POS column: {pos_col}")

    matches = []
    rows_seen = 0

    read_csv_kwargs = {
        "sep": sep,
        "chunksize": chunksize,
    }

    if sep == r"\s+":
        read_csv_kwargs["engine"] = "python"

    for chunk_idx, chunk in enumerate(pd.read_csv(path, **read_csv_kwargs), start=1):
        pos_numeric = pd.to_numeric(chunk[pos_col], errors="coerce")

        mask = (
            chunk[chr_col].astype(str).str.replace("chr", "", case=False, regex=False)
            == str(chromosome).replace("chr", "")
        ) & (pos_numeric == int(position))

        found = chunk[mask]

        if not found.empty:
            matches.append(found)

        rows_seen += len(chunk)

        if chunk_idx % 10 == 0:
            print(f"Processed chunks: {chunk_idx}, rows seen: {rows_seen}")

        if max_rows is not None and rows_seen >= max_rows:
            break

    if not matches:
        return pd.DataFrame()

    return pd.concat(matches, ignore_index=True)


def maybe_save_output(df: pd.DataFrame, output_path: Optional[str]) -> None:
    if output_path is None:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    print()
    print(f"Results written to: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Query a single GWAS summary statistics file."
    )

    parser.add_argument(
        "--file",
        required=True,
        help="Exact file name or unambiguous substring from the GWAS inventory.",
    )

    parser.add_argument("--show-columns", action="store_true")
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--top-p", type=int, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--chr", type=str, default=None)
    parser.add_argument("--pos", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional limit for testing on a subset of rows.",
    )
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    inventory = load_inventory()
    row = resolve_file(args.file, inventory)

    if args.show_columns:
        show_columns(row)

    if args.head is not None:
        show_head(row, args.head)

    if args.top_p is not None:
        results = top_p_values(
            row=row,
            n=args.top_p,
            chunksize=args.chunksize,
            max_rows=args.max_rows,
        )
        print()
        print(results.to_string(index=False))
        maybe_save_output(results, args.output)

    if args.variant is not None:
        results = find_variant(
            row=row,
            variant=args.variant,
            chunksize=args.chunksize,
            max_rows=args.max_rows,
        )
        print()
        if results.empty:
            print("No matching variant found.")
        else:
            print(results.to_string(index=False))
            maybe_save_output(results, args.output)

    if args.chr is not None or args.pos is not None:
        if args.chr is None or args.pos is None:
            raise ValueError("Both --chr and --pos are required for genomic position search.")

        results = find_chr_pos(
            row=row,
            chromosome=args.chr,
            position=args.pos,
            chunksize=args.chunksize,
            max_rows=args.max_rows,
        )
        print()
        if results.empty:
            print("No matching CHR/POS found.")
        else:
            print(results.to_string(index=False))
            maybe_save_output(results, args.output)

    if not any([
        args.show_columns,
        args.head is not None,
        args.top_p is not None,
        args.variant is not None,
        args.chr is not None,
        args.pos is not None,
    ]):
        show_columns(row)


if __name__ == "__main__":
    main()
