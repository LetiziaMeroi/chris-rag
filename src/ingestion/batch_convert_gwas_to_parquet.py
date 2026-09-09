import time
from pathlib import Path

import duckdb
import pandas as pd


CATALOG = Path("data/processed/gwas/gwas_catalog.csv")
OUTPUT_DIR = Path("data/processed/gwas/parquet")


def main():
    if not CATALOG.exists():
        raise FileNotFoundError(
            f"Catalog not found: {CATALOG}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(CATALOG)

    dhn = df[
        df["format"] == "DHN"
    ].copy()

    print(f"DHN files found: {len(dhn)}")

    con = duckdb.connect()

    results = []

    for i, row in dhn.iterrows():
        source = Path(row["file_path"])

        if not source.exists():
            print(f"[MISSING] {source}")

            results.append({
                "file_name": row["file_name"],
                "status": "missing",
                "seconds": None,
                "input_mb": None,
                "output_mb": None,
                "rows": None,
            })

            continue

        output = (
            OUTPUT_DIR /
            f"{source.name}.parquet"
        )

        if output.exists():
            print(f"[SKIP] {output.name}")

            results.append({
                "file_name": row["file_name"],
                "status": "already_exists",
                "seconds": 0,
                "input_mb": source.stat().st_size / (1024 ** 2),
                "output_mb": output.stat().st_size / (1024 ** 2),
                "rows": None,
            })

            continue

        print("\n" + "=" * 80)
        print(f"Converting: {source.name}")
        print("=" * 80)

        start = time.time()

        try:
            con.execute(f"""
                COPY (
                    SELECT *
                    FROM read_csv(
                        '{source}',
                        auto_detect=true,
                        header=true,
                        delim='\\t',
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
                TO '{output}'
                (
                    FORMAT PARQUET,
                    COMPRESSION ZSTD
                )
            """)

            elapsed = time.time() - start

            row_count = con.execute(f"""
                SELECT COUNT(*)
                FROM read_parquet('{output}')
            """).fetchone()[0]

            input_mb = source.stat().st_size / (1024 ** 2)
            output_mb = output.stat().st_size / (1024 ** 2)

            print(f"Done in {elapsed:.2f}s")
            print(f"Rows: {row_count:,}")
            print(f"Input: {input_mb:.2f} MB")
            print(f"Output: {output_mb:.2f} MB")

            results.append({
                "file_name": row["file_name"],
                "status": "ok",
                "seconds": round(elapsed, 2),
                "input_mb": round(input_mb, 2),
                "output_mb": round(output_mb, 2),
                "rows": row_count,
            })

        except Exception as exc:
            print(f"[ERROR] {source.name}")
            print(exc)

            results.append({
                "file_name": row["file_name"],
                "status": "error",
                "seconds": None,
                "input_mb": None,
                "output_mb": None,
                "rows": None,
            })

    con.close()

    report = pd.DataFrame(results)

    report_path = (
        Path("data/processed/gwas")
        / "parquet_conversion_report.csv"
    )

    report.to_csv(
        report_path,
        index=False
    )

    print("\n" + "=" * 80)
    print("BATCH SUMMARY")
    print("=" * 80)

    print(
        report["status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
