from pathlib import Path
import json
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer


DATA_ROOT = Path("/storage/data/chris-rag")
CHUNKS_PATH = DATA_ROOT / "processed" / "chunks" / "document_chunks.jsonl"
OUTPUT_DIR = DATA_ROOT / "processed" / "embeddings"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_chunks(path: Path):
    chunks = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(CHUNKS_PATH)
    texts = [chunk["text"] for chunk in chunks]

    print(f"Loaded chunks: {len(chunks)}")
    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.save(OUTPUT_DIR / "document_embeddings.npy", embeddings)

    with (OUTPUT_DIR / "document_chunks.pkl").open("wb") as f:
        pickle.dump(chunks, f)

    with (OUTPUT_DIR / "embedding_model.txt").open("w", encoding="utf-8") as f:
        f.write(MODEL_NAME + "\n")

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
