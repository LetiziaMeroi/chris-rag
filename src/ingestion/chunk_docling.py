from pathlib import Path
import argparse
import json
import re
from collections import defaultdict
from typing import Dict, List, Optional
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "docling_test"

DEFAULT_OUTPUT = Path(
    "/storage/data/chris-rag/processed/"
    "chunks_docling/document_chunks.jsonl"
)

# Old chunks are used only to recover metadata such as
# collection, original file_path, relative_path.
OLD_CHUNKS_PATH = Path(
    "/storage/data/chris-rag/processed/"
    "chunks/document_chunks.jsonl"
)

DEFAULT_MAX_CHARS = 1800


SKIP_LABELS = {
    "page_header",
    "page_footer",
}

METADATA_PATH = Path(
    "data/config/document_metadata.csv"
)

def clean_text(text: str) -> str:
    """
    Clean Docling text while preserving semantic content.
    """

    if not text:
        return ""

    # Remove soft hyphen
    text = text.replace("\u00ad", "")

    # Repair PDF line-break hyphenation:
    # partici-\npants -> participants
    text = re.sub(
        r"([A-Za-zÀ-ÖØ-öø-ÿ])-"
        r"\s*\n\s*"
        r"([a-zà-öø-ÿ])",
        r"\1\2",
        text,
    )

    # Normalize horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def load_document_metadata():

    metadata = {}

    with open(
        METADATA_PATH,
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            metadata[row["file_name"]] = {
                "year": (
                    int(row["year"])
                    if row["year"]
                    else None
                ),
                "document_type": row[
                    "document_type"
                ],
                "data_domain": row[
                    "data_domain"
                ],
            }

    return metadata

def load_old_metadata(path: Path) -> Dict[str, Dict]:
    """
    Recover metadata from the old chunk index.

    Maps:
        file_name -> metadata
    """

    metadata = {}

    if not path.exists():
        return metadata

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            chunk = json.loads(line)

            file_name = chunk.get(
                "file_name"
            )

            if (
                file_name
                and file_name not in metadata
            ):
                metadata[file_name] = {
                    "collection": chunk.get(
                        "collection"
                    ),
                    "file_path": chunk.get(
                        "file_path"
                    ),
                    "relative_path": chunk.get(
                        "relative_path"
                    ),
                    "source_md5": chunk.get(
                        "source_md5"
                    ),
                }

    return metadata


def get_source_filename(
    doc: Dict,
    document_dir: Path,
) -> str:
    """
    Try to recover the original PDF filename.
    """

    origin = doc.get("origin") or {}

    if isinstance(origin, dict):

        for key in [
            "filename",
            "file_name",
            "name",
        ]:
            value = origin.get(key)

            if value:
                return Path(
                    str(value)
                ).name

    name = doc.get("name")

    if name:
        name = str(name)

        if not name.lower().endswith(
            ".pdf"
        ):
            name += ".pdf"

        return name

    return document_dir.name + ".pdf"


def resolve_text_ref(
    doc: Dict,
    ref: Dict,
) -> Optional[str]:
    """
    Resolve refs such as:
        {'cref': '#/texts/88'}
    """

    if not isinstance(ref, dict):
        return None

    cref = ref.get("cref")

    if not cref:
        return None

    match = re.fullmatch(
        r"#/texts/(\d+)",
        cref,
    )

    if not match:
        return None

    index = int(
        match.group(1)
    )

    texts = doc.get(
        "texts",
        [],
    )

    if not (
        0 <= index < len(texts)
    ):
        return None

    return clean_text(
        texts[index].get(
            "text",
            ""
        )
    )


def get_table_linked_text_refs(
    doc: Dict,
) -> set:
    """
    Captions / footnotes already represented by table
    chunks should not also become normal text chunks.
    """

    refs = set()

    for table in doc.get(
        "tables",
        [],
    ):
        for field in [
            "captions",
            "footnotes",
            "children",
        ]:

            for item in table.get(
                field,
                [],
            ):
                if isinstance(
                    item,
                    dict,
                ):
                    cref = item.get(
                        "cref"
                    )

                    if cref:
                        refs.add(cref)

    return refs


def extract_page_fragments(
    doc: Dict,
) -> Dict[Optional[int], List[Dict]]:
    """
    Convert Docling TextItems into page-specific fragments.

    charspan is important for items that span two pages.
    """

    by_page = defaultdict(list)

    linked_refs = (
        get_table_linked_text_refs(
            doc
        )
    )

    for order, item in enumerate(
        doc.get("texts", [])
    ):

        self_ref = item.get(
            "self_ref"
        )

        if self_ref in linked_refs:
            continue

        label = item.get(
            "label"
        )

        if label in SKIP_LABELS:
            continue

        if (
            item.get("content_layer")
            == "furniture"
        ):
            continue

        full_text = item.get(
            "text",
            ""
        )

        if not full_text:
            continue

        provenance = (
            item.get("prov")
            or []
        )

        # No provenance: retain the text,
        # but page is unknown.
        if not provenance:

            text = clean_text(
                full_text
            )

            if text:
                by_page[None].append({
                    "text": text,
                    "label": label,
                    "self_ref": self_ref,
                    "order": order,
                })

            continue

        for prov in provenance:

            page = prov.get(
                "page_no"
            )

            charspan = prov.get(
                "charspan"
            )

            page_text = full_text

            # Important for Docling items
            # spanning multiple pages.
            if (
                isinstance(charspan, list)
                and len(charspan) == 2
            ):
                start, end = charspan

                if (
                    isinstance(start, int)
                    and isinstance(end, int)
                    and 0 <= start <= end
                    and end <= len(full_text)
                ):
                    page_text = (
                        full_text[start:end]
                    )

            page_text = clean_text(
                page_text
            )

            if not page_text:
                continue

            by_page[page].append({
                "text": page_text,
                "label": label,
                "self_ref": self_ref,
                "order": order,
            })

    return by_page


def split_long_text(
    text: str,
    max_chars: int,
) -> List[str]:
    """
    Sentence-aware fallback for an individual
    Docling item larger than max_chars.
    """

    if len(text) <= max_chars:
        return [text]

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    parts = []
    current = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        candidate = (
            sentence
            if not current
            else current + " " + sentence
        )

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            parts.append(current)
            current = ""

        # Very long sentence fallback:
        # split by words, never in the
        # middle of a word.
        if len(sentence) > max_chars:

            words = sentence.split()
            subpart = ""

            for word in words:

                candidate = (
                    word
                    if not subpart
                    else subpart + " " + word
                )

                if (
                    len(candidate)
                    <= max_chars
                ):
                    subpart = candidate
                else:
                    if subpart:
                        parts.append(
                            subpart
                        )

                    subpart = word

            if subpart:
                current = subpart

        else:
            current = sentence

    if current:
        parts.append(current)

    return parts


def build_text_chunks(
    doc: Dict,
    file_name: str,
    metadata: Dict,
    max_chars: int,
) -> List[Dict]:

    fragments_by_page = (
        extract_page_fragments(
            doc
        )
    )

    chunks = []

    stem = Path(
        file_name
    ).stem

    for page, fragments in (
        fragments_by_page.items()
    ):

        current = []
        current_length = 0
        page_chunk_index = 0

        def flush():
            nonlocal current
            nonlocal current_length
            nonlocal page_chunk_index

            if not current:
                return

            text = "\n\n".join(
                x["text"]
                for x in current
            )

            labels = sorted({
                x["label"]
                for x in current
                if x["label"]
            })

            refs = [
                x["self_ref"]
                for x in current
                if x["self_ref"]
            ]

            page_label = (
                str(page)
                if page is not None
                else "unknown"
            )

            chunk_id = (
                f"{stem}_p{page_label}"
                f"_text_{page_chunk_index:03d}"
            )

            chunks.append({
                "chunk_id": chunk_id,
                "chunk_type": "text",
                "file_name": file_name,
                "file_path": metadata.get(
                    "file_path"
                ),
                "relative_path": metadata.get(
                    "relative_path"
                ),
                "collection": metadata.get(
                    "collection",
                    "unknown",
                ),
                "page": page,
                "pages": (
                    [page]
                    if page is not None
                    else []
                ),
                "text": text,
                "char_count": len(text),
                "labels": labels,
                "docling_refs": refs,
                "source_md5": metadata.get(
                    "source_md5"
                ),
                "source_parser": "docling",
            })

            page_chunk_index += 1
            current = []
            current_length = 0

        for fragment in fragments:

            fragment_text = (
                fragment["text"]
            )

            # A single Docling item can
            # occasionally be very long.
            if (
                len(fragment_text)
                > max_chars
            ):

                flush()

                for part in split_long_text(
                    fragment_text,
                    max_chars,
                ):

                    temp = dict(
                        fragment
                    )
                    temp["text"] = part

                    current = [temp]
                    current_length = len(
                        part
                    )

                    flush()

                continue

            separator_length = (
                2 if current else 0
            )

            candidate_length = (
                current_length
                + separator_length
                + len(fragment_text)
            )

            if (
                current
                and candidate_length
                > max_chars
            ):
                flush()

            current.append(
                fragment
            )

            current_length += (
                len(fragment_text)
                + (
                    2
                    if len(current) > 1
                    else 0
                )
            )

        flush()

    return chunks


def table_caption(
    doc: Dict,
    table: Dict,
) -> str:

    captions = []

    for ref in table.get(
        "captions",
        [],
    ):
        text = resolve_text_ref(
            doc,
            ref,
        )

        if text:
            captions.append(text)

    return " ".join(
        captions
    ).strip()


def table_pages(
    table: Dict,
) -> List[int]:

    pages = []

    for prov in table.get(
        "prov",
        [],
    ):
        page = prov.get(
            "page_no"
        )

        if (
            page is not None
            and page not in pages
        ):
            pages.append(page)

    return pages


def split_markdown(
    markdown: str,
    max_chars: int,
) -> List[str]:
    """
    Split table Markdown at line boundaries where possible.
    """

    lines = markdown.splitlines()

    parts = []
    current = []

    current_length = 0

    for line in lines:

        line_length = len(line) + 1

        if (
            current
            and current_length
            + line_length
            > max_chars
        ):
            parts.append(
                "\n".join(current)
                .strip()
            )

            current = []
            current_length = 0

        # One extremely long line
        if (
            not current
            and line_length
            > max_chars
        ):
            pieces = split_long_text(
                line,
                max_chars,
            )

            parts.extend(
                pieces[:-1]
            )

            if pieces:
                current = [
                    pieces[-1]
                ]
                current_length = len(
                    pieces[-1]
                )

            continue

        current.append(line)
        current_length += line_length

    if current:
        parts.append(
            "\n".join(current)
            .strip()
        )

    return [
        part
        for part in parts
        if part
    ]


def build_table_chunks(
    doc: Dict,
    document_dir: Path,
    file_name: str,
    metadata: Dict,
    max_chars: int,
) -> List[Dict]:

    chunks = []

    stem = Path(
        file_name
    ).stem

    for table_index, table in enumerate(
        doc.get("tables", []),
        start=1,
    ):

        markdown_path = (
            document_dir
            / f"table_{table_index:03d}.md"
        )

        caption = table_caption(
            doc,
            table,
        )

        pages = table_pages(
            table
        )

        primary_page = (
            pages[0]
            if pages
            else None
        )

        if markdown_path.exists():

            markdown = (
                markdown_path.read_text(
                    encoding="utf-8"
                )
            )

            markdown = (
                markdown
                .replace("\u00ad", "")
                .strip()
            )

        else:
            # Fallback if custom Markdown
            # does not exist.
            cells = (
                table.get(
                    "data",
                    {}
                )
                .get(
                    "table_cells",
                    []
                )
            )

            markdown = "\n".join(
                cell.get(
                    "text",
                    ""
                )
                for cell in cells
                if cell.get("text")
            )

        parts = split_markdown(
            markdown,
            max_chars=max_chars,
        )

        for part_index, part in enumerate(
            parts,
            start=1,
        ):

            # Caption is useful semantic context,
            # especially if the table has been split.
            retrieval_text = part

            if (
                caption
                and caption.lower()
                not in part.lower()
            ):
                retrieval_text = (
                    f"Table caption: {caption}"
                    f"\n\n{part}"
                )

            chunk_id = (
                f"{stem}_table_"
                f"{table_index:03d}_"
                f"part_{part_index:03d}"
            )

            chunks.append({
                "chunk_id": chunk_id,
                "chunk_type": "table",
                "file_name": file_name,
                "file_path": metadata.get(
                    "file_path"
                ),
                "relative_path": metadata.get(
                    "relative_path"
                ),
                "collection": metadata.get(
                    "collection",
                    "unknown",
                ),
                "page": primary_page,
                "pages": pages,
                "table_index": table_index,
                "table_part": part_index,
                "caption": caption,
                "text": retrieval_text,
                "char_count": len(
                    retrieval_text
                ),
                "docling_ref": table.get(
                    "self_ref"
                ),
                "source_md5": metadata.get(
                    "source_md5"
                ),
                "source_parser": "docling",
            })

    return chunks


def process_document(
    document_json: Path,
    metadata_by_filename: Dict,
    max_chars: int,
) -> List[Dict]:

    document_dir = (
        document_json.parent
    )

    with document_json.open(
        "r",
        encoding="utf-8",
    ) as f:
        doc = json.load(f)

    file_name = get_source_filename(
        doc,
        document_dir,
    )

    metadata = (
        metadata_by_filename.get(
            file_name,
            {}
        )
    )

    text_chunks = build_text_chunks(
        doc=doc,
        file_name=file_name,
        metadata=metadata,
        max_chars=max_chars,
    )

    table_chunks = build_table_chunks(
        doc=doc,
        document_dir=document_dir,
        file_name=file_name,
        metadata=metadata,
        max_chars=max_chars,
    )

    return (
        text_chunks
        + table_chunks
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build structured RAG chunks "
            "from Docling outputs."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
    )

    args = parser.parse_args()

    document_files = sorted(
        args.input_dir.glob(
            "*/document.json"
        )
    )

    print(
        f"Docling documents found: "
        f"{len(document_files)}"
    )

    metadata_by_filename = (
        load_old_metadata(
            OLD_CHUNKS_PATH
        )
    )

    print(
        "Old metadata records: "
        f"{len(metadata_by_filename)}"
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total = 0
    text_total = 0
    table_total = 0

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as out:

        for document_json in (
            document_files
        ):

            chunks = process_document(
                document_json,
                metadata_by_filename,
                max_chars=args.max_chars,
            )

            text_count = sum(
                c["chunk_type"] == "text"
                for c in chunks
            )

            table_count = sum(
                c["chunk_type"] == "table"
                for c in chunks
            )

            print(
                f"{document_json.parent.name}: "
                f"{text_count} text + "
                f"{table_count} table"
            )

            for chunk in chunks:
                out.write(
                    json.dumps(
                        chunk,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            total += len(chunks)
            text_total += text_count
            table_total += table_count

    print()
    print("=" * 80)
    print("DOCLING CHUNKING SUMMARY")
    print("=" * 80)
    print(
        f"Documents:    "
        f"{len(document_files)}"
    )
    print(
        f"Text chunks:  "
        f"{text_total}"
    )
    print(
        f"Table chunks: "
        f"{table_total}"
    )
    print(
        f"Total chunks: "
        f"{total}"
    )
    print()
    print(
        f"Output: {args.output}"
    )


if __name__ == "__main__":
    main()
