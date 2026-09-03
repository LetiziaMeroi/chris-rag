from pathlib import Path

import pandas as pd


DATA_ROOT = Path("/storage/data/chris-rag")
INVENTORY_PATH = DATA_ROOT / "processed" / "gwas_inventory.csv"
OUTPUT_PATH = Path("docs/gwas_inventory_summary.md")


def yes_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0

    return int(df[column].fillna("").astype(str).str.strip().ne("").sum())


def main():
    df = pd.read_csv(INVENTORY_PATH)

    total_files = len(df)
    total_size_gb = df["size_mb"].sum() / 1024

    extension_counts = df["extension"].value_counts()
    readable_counts = df["readable"].value_counts()

    lines = []

    lines.append("# GWAS Inventory Summary")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Files indexed: {total_files}")
    lines.append(f"- Total size: {total_size_gb:.2f} GB")
    lines.append(f"- Readable files: {int(df['readable'].sum())}/{total_files}")
    lines.append("")

    lines.append("## File extensions")
    lines.append("")
    lines.append("| Extension | Count |")
    lines.append("|---|---:|")
    for extension, count in extension_counts.items():
        lines.append(f"| `{extension}` | {count} |")
    lines.append("")

    lines.append("## Detected GWAS columns")
    lines.append("")
    lines.append("| Semantic column | Files detected |")
    lines.append("|---|---:|")

    semantic_columns = [
        "col_variant_id",
        "col_chromosome",
        "col_position",
        "col_p_value",
        "col_beta",
        "col_standard_error",
        "col_sample_size",
        "col_frequency",
    ]

    for column in semantic_columns:
        lines.append(f"| `{column}` | {yes_count(df, column)} |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- `.DHN.gz` files appear to contain more standardized GWAS-style columns "
        "such as `CHR`, `POS`, `P`, `BETA`, and `SE`."
    )
    lines.append(
        "- `.TBL` and `.TBL.gz` files contain related GWAS result columns such as "
        "`MarkerName`, `Effect`, and `StdErr`, but their p-value and genomic position "
        "columns may require additional column-name normalization."
    )
    lines.append(
        "- The files are large, with many containing tens of millions of rows. "
        "Therefore, the first tabular component focuses on metadata extraction and "
        "inventory search rather than loading full tables into memory."
    )
    lines.append("")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Summary written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
