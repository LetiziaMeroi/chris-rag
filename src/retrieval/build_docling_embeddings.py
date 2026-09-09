from pathlib import Path
import argparse
import json

import numpy as np
from sentence_transformers import SentenceTransformer


DATA_ROOT = Path(
    "/storage/data/chris-rag/processed"
)

CHUNK_DIR = (
    DATA_ROOT
    / "chunks_docling"
)

OUTPUT_DIR = (
    DATA_ROOT
    / "embeddings_docling_e5_base"
)

MODEL_NAME = "intfloat/multilingual-e5-base"


LAYER_CONFIG = {
    "text": {
        "chunks": CHUNK_DIR / "text_chunks.jsonl",
        "embeddings": OUTPUT_DIR / "text_embeddings.npy",
        "metadata": OUTPUT_DIR / "text_index_metadata.json",
    },
    "table": {
        "chunks": CHUNK_DIR / "table_chunks.jsonl",
        "embeddings": OUTPUT_DIR / "table_embeddings.npy",
        "metadata": OUTPUT_DIR / "table_index_metadata.json",
    },
}


def load_chunks(path: Path):

    chunks = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if line.strip():
                chunks.append(
                    json.loads(line)
                )

    return chunks


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build multilingual E5 embeddings "
            "for Docling chunks."
        )
    )

    parser.add_argument(
        "--layer",
        choices=[
            "text",
            "table",
        ],
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    config = LAYER_CONFIG[
        args.layer
    ]

    chunks_path = config[
        "chunks"
    ]

    embeddings_path = config[
        "embeddings"
    ]

    metadata_path = config[
        "metadata"
    ]

    if not chunks_path.exists():
        raise FileNotFoundError(
            chunks_path
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks = load_chunks(
        chunks_path
    )

    print("=" * 80)
    print("DOCLING E5 EMBEDDING BUILD")
    print("=" * 80)

    print(f"Layer:  {args.layer}")
    print(f"Chunks: {len(chunks)}")
    print(f"Model:  {MODEL_NAME}")

    passages = [
        "passage: " + chunk["text"]
        for chunk in chunks
    ]

    print("\nLoading model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print("Encoding...")

    embeddings = model.encode(
        passages,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = embeddings.astype(
        np.float32
    )

    np.save(
        embeddings_path,
        embeddings,
    )

    metadata = {
        "layer": args.layer,
        "model": MODEL_NAME,
        "chunk_file": str(
            chunks_path
        ),
        "embedding_file": str(
            embeddings_path
        ),
        "num_chunks": len(chunks),
        "embedding_shape": list(
            embeddings.shape
        ),
        "normalized": True,
        "document_prefix": "passage: ",
        "query_prefix": "query: ",
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\nDone.")
    print(
        f"Shape: {embeddings.shape}"
    )

    print(
        f"Embeddings: {embeddings_path}"
    )

    print(
        f"Metadata:   {metadata_path}"
    )


if __name__ == "__main__":
    main()
