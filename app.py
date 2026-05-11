import streamlit as st
from financial_agent import financial_agent

st.set_page_config(
    page_title="AI Financial Agent",
    page_icon="📈",
    layout="wide",
)

st.title("📈 AI Financial Agent")

st.markdown(
    """
Ask questions about:
- stock prices
- analyst recommendations
- financial news
- company insights
"""
)

query = st.text_input(
    "Enter your financial query",
    placeholder="Example: Give latest NVDA news and analyst recommendations",
)

if st.button("Analyze"):

    if query.strip():

        with st.spinner("Analyzing financial data..."):

            try:

                response = financial_agent.run(query)

                st.markdown("## Response")

                st.markdown(response.content)

            except Exception as e:

                st.error(f"Error: {str(e)}")