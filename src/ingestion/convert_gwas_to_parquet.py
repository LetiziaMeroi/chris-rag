import argparse
import time
from pathlib import Path

import duckdb


def main():
    parser = argparse.ArgumentParser(
        description="Convert a GWAS DHN/TBL file to Parquet using DuckDB."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input GWAS file (.DHN.gz, .TBL.gz, .TBL)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path"
    )

    args = parser.parse_args()

    input_path = args.input

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    if args.output is None:
        output_dir = Path("data/processed/gwas/parquet")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = (
            output_dir /
            f"{input_path.name}.parquet"
        )
    else:
        output_path = args.output
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    con = duckdb.connect()

    start = time.time()

    # DuckDB can read compressed delimited text directly.
    # auto_detect=True lets it infer delimiter and column types.
    query = f"""
        COPY (
            SELECT *
            FROM read_csv(
                '{input_path}',
                auto_detect=true,
                header=true,
                delim='\t',
                types={{
                    'CHR': 'VARCHAR',
                    'POS': 'BIGINT',
                    'REF': 'VARCHAR',
                    'ALT': 'VARCHAR',
                    'ALT_AF': 'DOUBLE',
                    'SE': 'DOUBLE',
                    'BETA': 'DOUBLE',
                    'DIRECTION': 'VARCHAR',
                    'P': 'DOUBLE'
                }}
            )
        )
        TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
    """

    print("\nConverting...")

    con.execute(query)

    elapsed = time.time() - start

    input_size_mb = input_path.stat().st_size / (1024 ** 2)
    output_size_mb = output_path.stat().st_size / (1024 ** 2)

    print("\nDone.")
    print(f"Time:        {elapsed:.2f} s")
    print(f"Input size:  {input_size_mb:.2f} MB")
    print(f"Output size: {output_size_mb:.2f} MB")

    # Quick validation
    row_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{output_path}')
        """
    ).fetchone()[0]

    print(f"Rows:        {row_count:,}")

    print("\nColumns:")
    columns = con.execute(
        f"""
        DESCRIBE
        SELECT *
        FROM read_parquet('{output_path}')
        """
    ).fetchall()

    for col in columns:
        print(f"  {col[0]}: {col[1]}")

    con.close()


if __name__ == "__main__":
    main()
