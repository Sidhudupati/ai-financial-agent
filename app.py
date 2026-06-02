import streamlit as st

from financial_agent import financial_agent

from rag_utils import (
    extract_pdf_text,
    chunk_text,
    create_embeddings,
    create_faiss_index,
    retrieve_context
)

st.set_page_config(
    page_title="AI Financial Agent",
    page_icon="📈",
    layout="wide",
)

st.title("📈 AI Financial Agent")

st.markdown(
    """
Ask questions about:
- Stock prices
- Analyst recommendations
- Financial news
- Company insights
- Uploaded financial reports
"""
)

query = st.text_input(
    "Enter your financial query",
    placeholder="Example: Give latest NVDA news and analyst recommendations",
)

uploaded_file = st.file_uploader(
    "Upload Financial PDF",
    type=["pdf"]
)


@st.cache_resource
def process_pdf(pdf_text):

    chunks = chunk_text(pdf_text)

    embeddings = create_embeddings(chunks)

    index = create_faiss_index(embeddings)

    return chunks, embeddings, index


if uploaded_file:

    pdf_text = extract_pdf_text(uploaded_file)

    st.success("PDF Loaded Successfully")

    chunks, embeddings, index = process_pdf(pdf_text)

    st.session_state["chunks"] = chunks
    st.session_state["index"] = index

    with st.expander("Technical Details"):

        st.write("PDF Length:", len(pdf_text))

        st.write("Embedding Shape:", embeddings.shape)

        st.write("Number of Chunks:", len(chunks))

        st.text_area(
            "First Chunk",
            chunks[0],
            height=200
        )

    st.success(
        f"FAISS Index Created with {index.ntotal} vectors"
    )


if st.button("Analyze"):

    if query.strip():

        with st.spinner("Analyzing financial data..."):

            try:

                if (
                    "chunks" in st.session_state
                    and "index" in st.session_state
                ):

                    chunks = st.session_state["chunks"]
                    index = st.session_state["index"]

                    context = retrieve_context(
                        query,
                        chunks,
                        index
                    )

                    with st.expander("Retrieved Context"):

                        st.write(context)

                    enhanced_query = f"""
You are an AI Financial Analyst.

Financial Report Context:

{context}

User Question:

{query}

Instructions:

1. Use the financial report context whenever relevant.
2. Use Yahoo Finance tools for live market data.
3. Use latest_stock_news for recent news.
4. Never repeat information.
5. Separate factual data from analysis.
6. Mention when information comes from the uploaded report.
7. If information is unavailable in the report, use financial tools.
8. Keep responses concise and professional.
"""

                    response = financial_agent.run(
                        enhanced_query
                    )

                else:

                    response = financial_agent.run(
                        query
                    )

                st.markdown("## Response")

                st.markdown(response.content)

            except Exception as e:

                st.error(f"Error: {str(e)}")