from pathlib import Path
import json
import pandas as pd
import fitz  # PyMuPDF


DATA_ROOT = Path("/storage/data/chris-rag")
MANIFEST_PATH = DATA_ROOT / "processed" / "manifest.csv"

OUTPUT_TEXT_DIR = DATA_ROOT / "processed" / "extracted_text"
OUTPUT_TABLE_DIR = DATA_ROOT / "processed" / "extracted_tables"


def extract_pdf_pages_and_tables(pdf_path: Path, table_output_dir: Path):
    """
    Extract page text and detected PDF tables.

    Important:
    - Text is saved inside the JSON output.
    - Tables are saved separately as CSV files.
    - This does not perform OCR on image-based tables.
    """
    pages = []
    tables_metadata = []

    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text")

            pages.append({
                "page": page_number,
                "text": text.strip()
            })

            try:
                table_finder = page.find_tables()
            except Exception as e:
                print(f"  Table detection failed on page {page_number}: {e}")
                continue

            for table_index, table in enumerate(table_finder.tables):
                try:
                    df = table.to_pandas()

                    if df.empty:
                        continue

                    table_file_name = (
                        f"{pdf_path.stem}_page{page_number}_table{table_index}.csv"
                    )
                    table_path = table_output_dir / table_file_name

                    df.to_csv(table_path, index=False)

                    tables_metadata.append({
                        "page": page_number,
                        "table_index": table_index,
                        "table_path": str(table_path),
                        "n_rows": int(df.shape[0]),
                        "n_columns": int(df.shape[1]),
                        "columns": [str(col) for col in df.columns],
                    })

                except Exception as e:
                    print(
                        f"  Could not extract table {table_index} "
                        f"on page {page_number}: {e}"
                    )

    return pages, tables_metadata


def main():
    OUTPUT_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)
    pdfs = manifest[manifest["file_type"] == "document_pdf"]

    print(f"Found {len(pdfs)} PDF files")

    for _, row in pdfs.iterrows():
        pdf_path = Path(row["file_path"])
        file_name = pdf_path.stem

        print(f"Extracting: {pdf_path}")

        pages, tables_metadata = extract_pdf_pages_and_tables(
            pdf_path=pdf_path,
            table_output_dir=OUTPUT_TABLE_DIR,
        )

        output = {
            "file_name": row["file_name"],
            "file_path": row["file_path"],
            "relative_path": row["relative_path"],
            "collection": row["collection"],
            "md5": row["md5"],
            "num_pages": len(pages),
            "num_tables": len(tables_metadata),
            "tables": tables_metadata,
            "pages": pages,
        }

        output_path = OUTPUT_TEXT_DIR / f"{file_name}.json"

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"  Pages extracted: {len(pages)}")
        print(f"  Tables extracted: {len(tables_metadata)}")
        print(f"  Written: {output_path}")

    print("PDF extraction completed.")


if __name__ == "__main__":
    main()