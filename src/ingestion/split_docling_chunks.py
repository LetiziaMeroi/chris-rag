from pathlib import Path
import json


INPUT = Path(
    "/storage/data/chris-rag/processed/"
    "chunks_docling/document_chunks.jsonl"
)

TEXT_OUTPUT = Path(
    "/storage/data/chris-rag/processed/"
    "chunks_docling/text_chunks.jsonl"
)

TABLE_OUTPUT = Path(
    "/storage/data/chris-rag/processed/"
    "chunks_docling/table_chunks.jsonl"
)


def main():

    text_count = 0
    table_count = 0

    with (
        INPUT.open("r", encoding="utf-8") as src,
        TEXT_OUTPUT.open("w", encoding="utf-8") as text_out,
        TABLE_OUTPUT.open("w", encoding="utf-8") as table_out,
    ):

        for line in src:

            if not line.strip():
                continue

            chunk = json.loads(line)

            chunk_type = chunk.get("chunk_type")

            if chunk_type == "text":

                text_out.write(
                    json.dumps(
                        chunk,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                text_count += 1

            elif chunk_type == "table":

                table_out.write(
                    json.dumps(
                        chunk,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                table_count += 1

    print("=" * 80)
    print("DOCLING LAYER SPLIT")
    print("=" * 80)

    print(f"Text chunks:  {text_count}")
    print(f"Table chunks: {table_count}")

    print()
    print(f"Text output:  {TEXT_OUTPUT}")
    print(f"Table output: {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
