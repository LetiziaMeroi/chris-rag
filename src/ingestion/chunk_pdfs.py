from pathlib import Path
import json
import re
from typing import List, Dict


DATA_ROOT = Path("/storage/data/chris-rag")
EXTRACTED_TEXT_DIR = DATA_ROOT / "processed" / "extracted_text"
OUTPUT_DIR = DATA_ROOT / "processed" / "chunks"
OUTPUT_FILE = OUTPUT_DIR / "document_chunks.jsonl"


CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def clean_text(text: str) -> str:
    """
    Basic text cleaning.
    Keeps content simple and readable for retrieval.
    """
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks.
    This is a simple character-based chunker.
    Good enough for the first RAG prototype.
    """
    text = clean_text(text)

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to end at a sentence boundary if possible.
        candidate = text[start:end]
        sentence_end = max(
            candidate.rfind(". "),
            candidate.rfind("? "),
            candidate.rfind("! "),
            candidate.rfind("\n")
        )

        if sentence_end > chunk_size * 0.5:
            end = start + sentence_end + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = max(end - overlap, start + 1)

    return chunks


def build_chunks_from_document(doc: Dict) -> List[Dict]:
    """
    Create chunks from one extracted PDF JSON.
    Each chunk keeps source metadata for citation.
    """
    chunks = []

    file_name = doc.get("file_name")
    file_path = doc.get("file_path")
    relative_path = doc.get("relative_path")
    collection = doc.get("collection")
    md5 = doc.get("md5")

    for page_obj in doc.get("pages", []):
        page_number = page_obj.get("page")
        text = page_obj.get("text", "")

        page_chunks = split_text(text)

        for chunk_index, chunk_text in enumerate(page_chunks):
            chunk_id = f"{Path(file_name).stem}_p{page_number}_c{chunk_index}"

            chunks.append({
                "chunk_id": chunk_id,
                "file_name": file_name,
                "file_path": file_path,
                "relative_path": relative_path,
                "collection": collection,
                "page": page_number,
                "chunk_index": chunk_index,
                "text": chunk_text,
                "char_count": len(chunk_text),
                "source_md5": md5,
            })

    return chunks


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(EXTRACTED_TEXT_DIR.glob("*.json"))

    print(f"Found {len(json_files)} extracted PDF files")

    total_chunks = 0

    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        for json_path in json_files:
            print(f"Chunking: {json_path.name}")

            with json_path.open("r", encoding="utf-8") as f:
                doc = json.load(f)

            chunks = build_chunks_from_document(doc)

            for chunk in chunks:
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            print(f"  chunks: {len(chunks)}")
            total_chunks += len(chunks)

    print(f"\nDone.")
    print(f"Output written to: {OUTPUT_FILE}")
    print(f"Total chunks: {total_chunks}")


if __name__ == "__main__":
    main()
