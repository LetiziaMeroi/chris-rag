from pathlib import Path
from typing import Optional

import duckdb
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

DEFAULT_PARQUET_DIR = (
    GWAS_ROOT
    / "parquet"
)


class GWASService:
    def __init__(
        self,
        catalog_path: Path = DEFAULT_CATALOG,
        parquet_dir: Path = DEFAULT_PARQUET_DIR,
    ):
        self.catalog_path = Path(catalog_path)
        self.parquet_dir = Path(parquet_dir)

        if not self.catalog_path.exists():
            raise FileNotFoundError(
                f"GWAS catalog not found: {self.catalog_path}"
            )

        self.catalog = pd.read_csv(self.catalog_path)

    def list_traits(self):
        return sorted(
            self.catalog["trait"]
            .dropna()
            .unique()
            .tolist()
        )

    def find_dataset(
        self,
        trait: str,
        sex: str,
        build: str,
    ):
        matches = self.catalog[
            (self.catalog["format"] == "DHN")
            & (
                self.catalog["trait"]
                .astype(str)
                .str.lower()
                == trait.lower()
            )
            & (
                self.catalog["sex"]
                .astype(str)
                .str.lower()
                == sex.lower()
            )
            & (
                self.catalog["genome_build"]
                .astype(str)
                .str.lower()
                == build.lower()
            )
        ]

        if matches.empty:
            raise ValueError(
                f"No GWAS dataset found for "
                f"trait={trait}, sex={sex}, build={build}"
            )

        if len(matches) > 1:
            raise ValueError(
                f"Multiple GWAS datasets found for "
                f"trait={trait}, sex={sex}, build={build}"
            )

        row = matches.iloc[0]

        parquet_path = (
            self.parquet_dir
            / f"{row['file_name']}.parquet"
        )

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Parquet file not found: {parquet_path}"
            )

        return {
            "dataset_id": row.get("dataset_id"),
            "trait": row["trait"],
            "sex": row["sex"],
            "build": row["genome_build"],
            "file_name": row["file_name"],
            "source_path": row["file_path"],
            "parquet_path": str(parquet_path),
            "cohorts": row.get("cohorts"),
            "num_rows_estimate": (
                int(row["num_rows_estimate"])
                if pd.notna(row.get("num_rows_estimate"))
                else None
            ),
        }

    def query_variants(
        self,
        trait: str,
        sex: str,
        build: str,
        p_max: Optional[float] = None,
        chromosome: Optional[str] = None,
        pos_min: Optional[int] = None,
        pos_max: Optional[int] = None,
        ref: Optional[str] = None,
        alt: Optional[str] = None,
        af_min: Optional[float] = None,
        af_max: Optional[float] = None,
        limit: int = 20,
    ):
        dataset = self.find_dataset(
            trait=trait,
            sex=sex,
            build=build,
        )

        parquet_path = dataset["parquet_path"]

        conditions = []

        if p_max is not None:
            conditions.append(f"P <= {float(p_max)}")

        if chromosome is not None:
            chromosome = str(chromosome).replace("'", "''")
            conditions.append(
                f"CHR = '{chromosome}'"
            )

        if pos_min is not None:
            conditions.append(
                f"POS >= {int(pos_min)}"
            )

        if pos_max is not None:
            conditions.append(
                f"POS <= {int(pos_max)}"
            )

        if ref is not None:
            ref = str(ref).replace("'", "''")
            conditions.append(
                f"REF = '{ref}'"
            )

        if alt is not None:
            alt = str(alt).replace("'", "''")
            conditions.append(
                f"ALT = '{alt}'"
            )

        if af_min is not None:
            conditions.append(
                f"ALT_AF >= {float(af_min)}"
            )

        if af_max is not None:
            conditions.append(
                f"ALT_AF <= {float(af_max)}"
            )

        where_clause = ""

        if conditions:
            where_clause = (
                "WHERE " + " AND ".join(conditions)
            )

        con = duckdb.connect()

        count_query = f"""
            SELECT COUNT(*)
            FROM read_parquet('{parquet_path}')
            {where_clause}
        """

        total_matches = con.execute(
            count_query
        ).fetchone()[0]

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
            LIMIT {int(limit)}
        """

        df = con.execute(query).df()

        con.close()

        filters = {
            "p_max": p_max,
            "chromosome": chromosome,
            "pos_min": pos_min,
            "pos_max": pos_max,
            "ref": ref,
            "alt": alt,
            "af_min": af_min,
            "af_max": af_max,
        }

        filters = {
            key: value
            for key, value in filters.items()
            if value is not None
        }

        return {
            "dataset": dataset,
            "filters": filters,
            "count": int(total_matches),
            "rows": df.to_dict(
                orient="records"
            ),
        }

    def get_variant(
        self,
        trait: str,
        sex: str,
        build: str,
        chromosome: str,
        position: int,
        ref: Optional[str] = None,
        alt: Optional[str] = None,
    ):
        return self.query_variants(
            trait=trait,
            sex=sex,
            build=build,
            chromosome=chromosome,
            pos_min=position,
            pos_max=position,
            ref=ref,
            alt=alt,
            limit=100,
        )
