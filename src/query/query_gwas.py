import argparse
from pathlib import Path

import duckdb
import pandas as pd


DEFAULT_CATALOG = Path("data/processed/gwas/gwas_catalog.csv")
DEFAULT_PARQUET_DIR = Path("data/processed/gwas/parquet")


def find_dataset(catalog, trait, sex, build):
    matches = catalog[
        (catalog["trait"].str.lower() == trait.lower())
        & (catalog["sex"].str.lower() == sex.lower())
        & (catalog["genome_build"].str.lower() == build.lower())
        & (catalog["format"] == "DHN")
    ]

    if matches.empty:
        raise ValueError(
            f"No DHN dataset found for "
            f"trait={trait}, sex={sex}, build={build}"
        )

    if len(matches) > 1:
        print("WARNING: multiple matching datasets found:")
        print(
            matches[
                [
                    "dataset_id",
                    "file_name",
                    "file_path",
                ]
            ].to_string(index=False)
        )

    return matches.iloc[0]


def find_parquet(dataset_row, parquet_dir):
    """
    Find the Parquet file corresponding to the DHN source file.
    """

    source_name = dataset_row["file_name"]

    expected = parquet_dir / f"{source_name}.parquet"

    if expected.exists():
        return expected

    # fallback: search by source filename
    candidates = list(
        parquet_dir.glob(f"*{source_name}*.parquet")
    )

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            f"No Parquet file found for:\n"
            f"{source_name}\n\n"
            f"Expected:\n{expected}"
        )

    raise RuntimeError(
        f"Multiple Parquet candidates found for {source_name}: "
        f"{candidates}"
    )


def build_where_clause(args):
    conditions = []

    if args.p_max is not None:
        conditions.append(f"P <= {args.p_max}")

    if args.chr is not None:
        conditions.append(f"CHR = '{args.chr}'")

    if args.pos_min is not None:
        conditions.append(f"POS >= {args.pos_min}")

    if args.pos_max is not None:
        conditions.append(f"POS <= {args.pos_max}")

    if args.ref is not None:
        ref = args.ref.replace("'", "''")
        conditions.append(f"REF = '{ref}'")

    if args.alt is not None:
        alt = args.alt.replace("'", "''")
        conditions.append(f"ALT = '{alt}'")

    if args.af_min is not None:
        conditions.append(f"ALT_AF >= {args.af_min}")

    if args.af_max is not None:
        conditions.append(f"ALT_AF <= {args.af_max}")

    if not conditions:
        return ""

    return "WHERE " + " AND ".join(conditions)


def main():
    parser = argparse.ArgumentParser(
        description="Query CHRIS GWAS Parquet datasets."
    )

    parser.add_argument(
        "--trait",
        required=True,
        help="Trait name, e.g. banana"
    )

    parser.add_argument(
        "--sex",
        required=True,
        choices=["all", "female", "male"],
        help="Sex stratum"
    )

    parser.add_argument(
        "--build",
        required=True,
        choices=["GRCh37", "GRCh38"],
        help="Genome build"
    )

    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help="GWAS catalog CSV"
    )

    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=DEFAULT_PARQUET_DIR,
        help="Directory containing Parquet GWAS files"
    )

    parser.add_argument(
        "--p-max",
        type=float,
        default=None,
        help="Maximum P-value"
    )

    parser.add_argument(
        "--chr",
        type=str,
        default=None,
        help="Chromosome, e.g. 1, 22, X"
    )

    parser.add_argument(
        "--pos-min",
        type=int,
        default=None,
        help="Minimum genomic position"
    )

    parser.add_argument(
        "--pos-max",
        type=int,
        default=None,
        help="Maximum genomic position"
    )

    parser.add_argument(
        "--ref",
        type=str,
        default=None,
        help="Reference allele"
    )

    parser.add_argument(
        "--alt",
        type=str,
        default=None,
        help="Alternate allele"
    )

    parser.add_argument(
        "--af-min",
        type=float,
        default=None,
        help="Minimum ALT_AF"
    )

    parser.add_argument(
        "--af-max",
        type=float,
        default=None,
        help="Maximum ALT_AF"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of rows to display"
    )

    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Return only the count of matching variants"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path"
    )

    args = parser.parse_args()

    # -------------------------------------------------------
    # Load catalog
    # -------------------------------------------------------

    if not args.catalog.exists():
        raise FileNotFoundError(
            f"Catalog not found: {args.catalog}"
        )

    catalog = pd.read_csv(args.catalog)

    dataset = find_dataset(
        catalog,
        trait=args.trait,
        sex=args.sex,
        build=args.build,
    )

    parquet_path = find_parquet(
        dataset,
        args.parquet_dir,
    )

    print("\n" + "=" * 80)
    print("GWAS DATASET")
    print("=" * 80)

    print(f"Trait:        {dataset['trait']}")
    print(f"Sex:          {dataset['sex']}")
    print(f"Genome build: {dataset['genome_build']}")
    print(f"Source file:  {dataset['file_name']}")
    print(f"Parquet:      {parquet_path}")

    where_clause = build_where_clause(args)

    con = duckdb.connect()

    # -------------------------------------------------------
    # Count matching variants
    # -------------------------------------------------------

    count_query = f"""
        SELECT COUNT(*)
        FROM read_parquet('{parquet_path}')
        {where_clause}
    """

    count = con.execute(count_query).fetchone()[0]

    print("\n" + "=" * 80)
    print("QUERY")
    print("=" * 80)

    if where_clause:
        print(where_clause)
    else:
        print("No filters")

    print(f"\nMatching variants: {count:,}")

    if args.count_only:
        con.close()
        return

    # -------------------------------------------------------
    # Retrieve variants
    # -------------------------------------------------------

    query = f"""
        SELECT
            CHR,
            POS,
            REF,
            ALT,
            ALT_AF,
            BETA,
            SE,
            DIRECTION,
            P
        FROM read_parquet('{parquet_path}')
        {where_clause}
        ORDER BY P ASC
        LIMIT {args.limit}
    """

    df = con.execute(query).df()

    print("\n" + "=" * 80)
    print(f"TOP {min(args.limit, len(df))} RESULTS")
    print("=" * 80)

    if df.empty:
        print("No variants found.")
    else:
        print(df.to_string(index=False))

    # -------------------------------------------------------
    # Optional export
    # -------------------------------------------------------

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            args.output,
            index=False
        )

        print(f"\nSaved results to: {args.output}")

    con.close()


if __name__ == "__main__":
    main()
