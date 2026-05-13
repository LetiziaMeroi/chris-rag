from pathlib import Path
import csv
import hashlib
from datetime import datetime


DATA_ROOT = Path("/storage/data/chris-rag")
OUTPUT_FILE = DATA_ROOT / "processed" / "manifest.csv"


def compute_md5(path: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 hash for file version tracking."""
    md5 = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def classify_file(path: Path) -> str:
    """Classify files into simple categories."""
    suffix = path.suffix.lower()
    name = path.name.lower()

    if suffix == ".pdf":
        return "document_pdf"

    if suffix in [".csv", ".tsv", ".tbl"]:
        return "tabular"

    if suffix == ".gz":
        if ".tbl" in name or ".tsv" in name or ".csv" in name:
            return "compressed_tabular"
        return "compressed_file"

    if suffix in [".log", ".info"]:
        return "metadata_or_log"

    return "other"


def infer_collection(path: Path) -> str:
    """Infer high-level collection from path."""
    parts = [p.lower() for p in path.parts]

    if "codebooks" in parts:
        return "codebooks"
    if "governance" in parts:
        return "governance"
    if "papers" in parts:
        return "papers"
    if "gwas" in parts:
        return "gwas"

    return "unknown"


def build_manifest():
    rows = []

    for path in DATA_ROOT.rglob("*"):
        if path.is_file():
            stat = path.stat()

            rows.append({
                "file_name": path.name,
                "file_path": str(path),
                "relative_path": str(path.relative_to(DATA_ROOT)),
                "extension": path.suffix.lower(),
                "file_type": classify_file(path),
                "collection": infer_collection(path),
                "size_bytes": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "md5": compute_md5(path),
            })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file_name",
                "file_path",
                "relative_path",
                "extension",
                "file_type",
                "collection",
                "size_bytes",
                "modified_time",
                "md5",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest written to: {OUTPUT_FILE}")
    print(f"Total files indexed: {len(rows)}")


if __name__ == "__main__":
    build_manifest()
