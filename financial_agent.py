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
    name="Financial Agent",
    role="Get stock data and recent financial news",
    model=groq_model,
    tools=[
        latest_stock_news,
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
        ),
    ],
    instructions=[
    "Always use Yahoo Finance tools for stock prices and analyst recommendations",
    "Never guess or fabricate financial data",
    "If data is unavailable, explicitly mention it",
    "If financial tools fail, use latest_stock_news to provide qualitative analysis",
    "Use latest_stock_news for financial news",
    "Use tables for financial data",
    "Separate factual data from analysis",
    "Do not provide website URLs",
    "Summarize clearly and concisely",
    ],
    markdown=True,
    show_tool_calls=False,
)