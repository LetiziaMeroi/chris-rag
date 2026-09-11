import requests
import streamlit as st


import os

API_URL = os.getenv(
    "CHRIS_API_URL",
    "http://127.0.0.1:8000/query",
)


st.set_page_config(
    page_title="CHRIS RAG",
    page_icon="🧬",
    layout="wide",
)


st.title("CHRIS RAG")
st.caption(
    "Query CHRIS documents and GWAS datasets using natural language."
)

with st.sidebar:

    st.subheader("Example questions")

    if st.button("Blood pressure"):
        st.session_state["query"] = (
            "How many measurements are used for "
            "the mean systolic blood pressure variable?"
        )

    if st.button("GWAS example"):
        st.session_state["query"] = (
            "Show genome-wide significant variants "
            "for peppermint in females using GRCh38"
        )

if "query" not in st.session_state:
    st.session_state["query"] = ""

query = st.text_area(
    "Ask a question",
    placeholder=(
            "Example: How many measurements are used for "
            "the mean systolic blood pressure variable?"
        ),
    key="query",
    height=100,
)


if st.button(
    "Run query",
    type="primary",
):

    if not query.strip():
        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Processing query..."
        ):

            try:

                response = requests.post(
                    API_URL,
                    json={
                        "query": query
                    },
                    timeout=120,
                )

                response.raise_for_status()

                result = response.json()

            except Exception as exc:

                st.error(
                    f"API request failed: {exc}"
                )

                st.stop()

        route = result.get(
            "route",
            "unknown",
        )

        status = result.get(
            "status",
            "unknown",
        )

        answer = result.get(
            "answer",
        )

        evidence = result.get(
            "evidence",
            [],
        )

        metadata = result.get(
            "metadata",
            {},
        )


        # ==========================================
        # Main answer
        # ==========================================



        st.subheader(
            "Answer"
        )

        if answer:
            st.write(
                answer
            )
        else:
            st.info(
                "No natural-language answer returned."
            )


        # ==========================================
        # GWAS results
        # ==========================================

        if route == "gwas":

            count = metadata.get(
                "count"
            )

            dataset = metadata.get(
                "dataset",
                {},
            )

            filters = metadata.get(
                "filters",
                {},
            )

            if count is not None:

                st.subheader(
                    "GWAS summary"
                )

                st.metric(
                    "Matching variants",
                    f"{count:,}",
                )

            if dataset:

                with st.expander(
                    "Dataset information"
                ):

                    st.json(
                        dataset
                    )

            if filters:

                with st.expander(
                    "Applied filters"
                ):

                    st.json(
                        filters
                    )

            variants = []

            for item in evidence:

                if (
                    item.get("type")
                    == "gwas_variant"
                ):

                    variant = item.get(
                        "variant"
                    )

                    if variant:
                        variants.append(
                            variant
                        )

            if variants:

                st.subheader(
                    "Top variants"
                )

                st.dataframe(
                    variants,
                    use_container_width=True,
                    hide_index=True,
                    column_order=[
                        "CHR",
                        "POS",
                        "REF",
                        "ALT",
                        "P",
                        "BETA",
                        "SE",
                        "ALT_AF",
                        "DIRECTION",
                    ],
                    column_config={
                        "CHR": "Chromosome",
                        "POS": "Position",
                        "P": st.column_config.NumberColumn(
                            "P-value",
                            format="%.3e",
                        ),
                        "BETA": st.column_config.NumberColumn(
                            "Beta",
                            format="%.4f",
                        ),
                        "SE": st.column_config.NumberColumn(
                            "SE",
                            format="%.4f",
                        ),
                        "ALT_AF": st.column_config.NumberColumn(
                            "ALT AF",
                            format="%.4f",
                        ),
                    },
                )


        # ==========================================
        # Document evidence
        # ==========================================

        if route == "documents":

            st.subheader("Sources")

            for item in evidence:

                label = item.get(
                    "label",
                    "?",
                )

                source = item.get(
                    "source",
                    {},
                )

                file_name = source.get(
                    "file_name",
                    "Unknown source",
                )

                page = source.get(
                    "page",
                    "?",
                )

                evidence_type = item.get(
                    "type",
                    "unknown",
                )

                table_index = source.get(
                    "table_index"
                )

                if table_index is not None:
                    location = (
                        f"page {page}, "
                        f"table {table_index}"
                    )
                else:
                    location = (
                        f"page {page}"
                    )

                title = (
                    f"[{label}] "
                    f"{file_name} — "
                    f"{location}"
                )

                with st.expander(
                    title
                ):

                    st.caption(
                        f"Evidence type: "
                        f"{evidence_type}"
                    )

                    caption = item.get(
                        "caption"
                    )

                    if caption:
                        st.markdown(
                            f"**Table caption:** "
                            f"{caption}"
                        )

                    text = item.get(
                        "text"
                    )

                    if text:
                        st.markdown(
                            text
                        )

        # ==========================================
        # Raw metadata
        # ==========================================

        with st.expander(
            "Technical metadata"
        ):

            st.json(
                metadata
            )


        with st.expander("Route and status"):

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Route",
                    route,
                )

            with col2:
                st.metric(
                    "Status",
                    status,
                )
