from pathlib import Path
import json
import pandas as pd
import fitz  # PyMuPDF


DATA_ROOT = Path("/storage/data/chris-rag")
MANIFEST_PATH = DATA_ROOT / "processed" / "manifest.csv"
OUTPUT_DIR = DATA_ROOT / "processed" / "extracted_text"


def extract_pdf_pages(pdf_path: Path):
    pages = []

    with fitz.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text")

            pages.append({
                "page": page_number,
                "text": text.strip()
            })

    return pages


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)
    pdfs = manifest[manifest["file_type"] == "document_pdf"]

    print(f"Found {len(pdfs)} PDF files")

    for _, row in pdfs.iterrows():
        pdf_path = Path(row["file_path"])
        file_name = pdf_path.stem

        print(f"Extracting: {pdf_path}")

        pages = extract_pdf_pages(pdf_path)

        output = {
            "file_name": row["file_name"],
            "file_path": row["file_path"],
            "relative_path": row["relative_path"],
            "collection": row["collection"],
            "md5": row["md5"],
            "num_pages": len(pages),
            "pages": pages,
        }

        output_path = OUTPUT_DIR / f"{file_name}.json"

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"Written: {output_path}")

    print("PDF extraction completed.")


if __name__ == "__main__":
    main()
