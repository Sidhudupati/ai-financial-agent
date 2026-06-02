import os
import json

from ddgs import DDGS
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools

load_dotenv()

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

groq_model = Groq(
    id="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)


def latest_stock_news(query: str) -> str:
    """Search the web for recent stock news."""

    results = DDGS().text(query, max_results=5)

    formatted_news = []

    for article in results:
        formatted_news.append(
            {
                "headline": article.get("title", ""),
                "summary": article.get("body", ""),
            }
        )

    return json.dumps(formatted_news, indent=2)


financial_agent = Agent(
    name="AI Finance RAG Agent",
    role="""
    Provide financial analysis using:
    - Yahoo Finance data
    - Recent financial news
    - Retrieved financial report context
    """,
    model=groq_model,
    tools=[
        latest_stock_news,
        #pdf_retriever,
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
        ),
    ],
    instructions=[
    "Always use Yahoo Finance tools for stock prices, company information, and analyst recommendations.",
    "Always use latest_stock_news when the user asks for recent news.",
    "Never fabricate financial data.",
    "If data is unavailable, explicitly state it.",
    "Present company information only once.",
    "Present analyst recommendations only once.",
    "Do not repeat information already shown.",
    "Separate factual data from analysis.",
    "Use markdown tables for financial metrics.",
    "Do not provide website URLs.",
    "Keep responses concise and professional.",
],
    markdown=True,
    show_tool_calls=False,
)