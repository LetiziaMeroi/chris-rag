import csv
import gzip
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional


DATA_ROOT = Path("/storage/data/chris-rag")
GWAS_ROOT = DATA_ROOT / "tabular" / "GWAS"
OUTPUT_PATH = DATA_ROOT / "processed" / "gwas_inventory.csv"


TABULAR_SUFFIXES = [
    ".csv",
    ".tsv",
    ".tbl",
    ".txt",
    ".gz",
]


GWAS_COLUMN_SYNONYMS = {
    "chromosome": {"chr", "chrom", "chromosome", "#chrom", "marker_chr"},
    "position": {"pos", "position", "bp", "base_pair_location", "marker_pos"},
    "variant_id": {
        "snp", "rsid", "rs_id", "variant", "variant_id",
        "markername", "marker_name", "id", "marker"
    },
    "effect_allele": {
        "a1", "ea", "effect_allele", "allele1", "alt",
        "codedallele", "coded_allele"
    },
    "other_allele": {
        "a2", "nea", "other_allele", "non_effect_allele",
        "allele2", "ref", "noncodedallele", "noncoded_allele"
    },
    "beta": {"beta", "effect", "estimate", "b"},
    "standard_error": {"se", "stderr", "standard_error", "stderr", "StdErr".lower()},
    "p_value": {
        "p", "pval", "p_value", "pvalue", "p.value",
        "p-value", "pvalue_gc", "p_gc"
    },
    "sample_size": {"n", "samplesize", "sample_size", "n_total", "n_total"},
    "frequency": {
        "eaf", "maf", "af", "effect_allele_frequency",
        "codedallelefreq", "coded_allele_freq"
    },
}


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.md5()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)

    return hasher.hexdigest()


def is_tabular_file(path: Path) -> bool:
    name = path.name.lower()

    if name.endswith(".tar.gz"):
        return False

    if any(name.endswith(suffix) for suffix in TABULAR_SUFFIXES):
        return True

    return False


def open_text_maybe_gzip(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")

    return path.open("r", encoding="utf-8", errors="replace")


def detect_delimiter(sample_line: str) -> str:
    candidates = {
        "\t": sample_line.count("\t"),
        ",": sample_line.count(","),
        ";": sample_line.count(";"),
        " ": sample_line.count(" "),
    }

    delimiter = max(candidates, key=candidates.get)

    if candidates[delimiter] == 0:
        return ""

    return delimiter


def split_line(line: str, delimiter: str) -> List[str]:
    line = line.strip()

    if delimiter == "":
        return [line]

    if delimiter == " ":
        return re.split(r"\s+", line)

    return line.split(delimiter)


def read_header_and_sample(path: Path, max_sample_rows: int = 5) -> Dict:
    header = []
    sample_rows = []
    delimiter = ""
    readable = True
    error = ""

    try:
        with open_text_maybe_gzip(path) as f:
            non_empty_lines = []

            for line in f:
                if line.strip():
                    non_empty_lines.append(line.rstrip("\n"))
                if len(non_empty_lines) >= max_sample_rows + 1:
                    break

        if not non_empty_lines:
            return {
                "readable": True,
                "delimiter": "",
                "columns": [],
                "num_columns": 0,
                "sample_rows": [],
                "error": "",
            }

        delimiter = detect_delimiter(non_empty_lines[0])
        header = split_line(non_empty_lines[0], delimiter)
        sample_rows = [
            split_line(line, delimiter)
            for line in non_empty_lines[1 : max_sample_rows + 1]
        ]

    except Exception as e:
        readable = False
        error = str(e)

    return {
        "readable": readable,
        "delimiter": delimiter,
        "columns": header,
        "num_columns": len(header),
        "sample_rows": sample_rows,
        "error": error,
    }


def estimate_num_rows(path: Path) -> Optional[int]:
    """
    Counts lines. For gzipped files this can be slow, but acceptable as a baseline
    for the current GWAS inventory. Returns data rows excluding the header.
    """
    try:
        with open_text_maybe_gzip(path) as f:
            line_count = sum(1 for _ in f)

        if line_count == 0:
            return 0

        return max(line_count - 1, 0)

    except Exception:
        return None


def normalize_column_name(column: str) -> str:
    return column.strip().lower()


def detect_gwas_columns(columns: List[str]) -> Dict[str, str]:
    normalized_to_original = {
        normalize_column_name(column): column
        for column in columns
    }

    detected = {}

    for semantic_name, synonyms in GWAS_COLUMN_SYNONYMS.items():
        detected_value = ""

        for normalized, original in normalized_to_original.items():
            if normalized in synonyms:
                detected_value = original
                break

        detected[semantic_name] = detected_value

    return detected


def infer_trait_from_filename(path: Path) -> str:
    name = path.name

    name = re.sub(r"\.gz$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\.(csv|tsv|tbl|txt)$", "", name, flags=re.IGNORECASE)

    # Remove common GWAS/result words, but keep the rest as rough phenotype label.
    name = re.sub(r"(?i)gwas|sumstats|summary|statistics|results|meta|analysis", "", name)
    name = re.sub(r"[_\-\.]+", " ", name)
    name = re.sub(r"\s+", " ", name)

    return name.strip()


def build_inventory() -> List[Dict]:
    rows = []

    files = [
        path
        for path in GWAS_ROOT.rglob("*")
        if path.is_file() and is_tabular_file(path)
    ]

    for i, path in enumerate(sorted(files), start=1):
        print(f"[{i}/{len(files)}] {path}")

        rel_path = path.relative_to(DATA_ROOT)
        stat = path.stat()

        header_info = read_header_and_sample(path)
        columns = header_info["columns"]
        detected_columns = detect_gwas_columns(columns)

        num_rows = estimate_num_rows(path)

        row = {
            "file_name": path.name,
            "file_path": str(path),
            "relative_path": str(rel_path),
            "extension": "".join(path.suffixes),
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 3),
            "is_compressed": int(path.name.lower().endswith(".gz")),
            "readable": int(header_info["readable"]),
            "delimiter": repr(header_info["delimiter"]),
            "num_columns": header_info["num_columns"],
            "num_rows_estimate": "" if num_rows is None else num_rows,
            "trait_from_filename": infer_trait_from_filename(path),
            "columns": "|".join(columns),
            "sample_rows": repr(header_info["sample_rows"][:3]),
            "read_error": header_info["error"],
            "md5": md5_file(path),
        }

        for semantic_name, detected_column in detected_columns.items():
            row[f"col_{semantic_name}"] = detected_column

        rows.append(row)

    return rows


def save_inventory(rows: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_fieldnames = [
        "file_name",
        "file_path",
        "relative_path",
        "extension",
        "size_bytes",
        "size_mb",
        "is_compressed",
        "readable",
        "delimiter",
        "num_columns",
        "num_rows_estimate",
        "trait_from_filename",
        "columns",
        "sample_rows",
        "read_error",
        "md5",
    ]

    semantic_fieldnames = [
        f"col_{name}"
        for name in GWAS_COLUMN_SYNONYMS.keys()
    ]

    fieldnames = base_fieldnames + semantic_fieldnames

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not GWAS_ROOT.exists():
        raise FileNotFoundError(f"GWAS root does not exist: {GWAS_ROOT}")

    rows = build_inventory()
    save_inventory(rows, OUTPUT_PATH)

    print("=" * 80)
    print("GWAS INVENTORY COMPLETE")
    print("=" * 80)
    print(f"Files indexed: {len(rows)}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
