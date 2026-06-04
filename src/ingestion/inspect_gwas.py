from pathlib import Path
import pandas as pd
import re


DATA_ROOT = Path("/storage/data/chris-rag")
MANIFEST_PATH = DATA_ROOT / "processed" / "manifest.csv"
OUTPUT_PATH = DATA_ROOT / "processed" / "gwas_inventory.csv"


def parse_gwas_filename(file_name: str):
    name = file_name

    phenotype = None
    sex_group = None
    genome_build = None
    file_role = None

    if "GRCh37" in name:
        genome_build = "GRCh37"
    elif "GRCh38" in name:
        genome_build = "GRCh38"

    if "_female" in name:
        sex_group = "female"
    elif "_male" in name:
        sex_group = "male"
    elif "_all" in name:
        sex_group = "all"

    # Extract phenotype between EUR_ and _all/_male/_female
    match = re.search(r"EUR_([A-Za-z0-9]+)_(all|male|female)", name)
    if match:
        phenotype = match.group(1)

    if ".TBL" in name:
        file_role = "association_results"
    elif ".DHN" in name:
        file_role = "harmonized_or_lifted_results"
    elif name.endswith(".info"):
        file_role = "info"
    elif name.endswith(".log") or ".log" in name:
        file_role = "log"
    else:
        file_role = "other"

    return phenotype, sex_group, genome_build, file_role


def main():
    manifest = pd.read_csv(MANIFEST_PATH)

    gwas = manifest[manifest["collection"] == "gwas"].copy()

    parsed = gwas["file_name"].apply(parse_gwas_filename)

    gwas["phenotype"] = parsed.apply(lambda x: x[0])
    gwas["sex_group"] = parsed.apply(lambda x: x[1])
    gwas["genome_build"] = parsed.apply(lambda x: x[2])
    gwas["file_role"] = parsed.apply(lambda x: x[3])

    gwas.to_csv(OUTPUT_PATH, index=False)

    print(f"GWAS inventory written to: {OUTPUT_PATH}")
    print(f"Total GWAS files: {len(gwas)}")

    print("\nPhenotypes:")
    print(gwas["phenotype"].value_counts(dropna=False))

    print("\nSex groups:")
    print(gwas["sex_group"].value_counts(dropna=False))

    print("\nGenome builds:")
    print(gwas["genome_build"].value_counts(dropna=False))

    print("\nFile roles:")
    print(gwas["file_role"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
