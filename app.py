import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize agents
web_agent = Agent(
   name="Web Agent",
   model=Groq(id="llama-3.3-70b-versatile"),
   tools=[DuckDuckGo()],
   show_tool_calls=True,
   markdown=True,
   instructions=["Always include sources"]
)

finance_agent = Agent(
   name="Finance Agent",
   role="Get Financial Data",
   model=Groq(id="llama-3.3-70b-versatile"),
   tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
   show_tool_calls=True,
   markdown=True,
   instructions=["Use tables to present data"]
)

agent_team = Agent(
   team=[web_agent, finance_agent],
   model=Groq(id="llama-3.3-70b-versatile"),
   instructions=["Always include sources", "Use tables to display data"],
   show_tool_calls=True,
   markdown=True,
)

# Streamlit app interface
st.title("Finance and Stock Analysis Assistant")
st.write("Ask any question about finance and stocks, and get insights from our AI agents.")

user_input = st.text_area("Enter your question here:")

if st.button("Get Answer"):
   if user_input.strip():
       with st.spinner("Processing..."):
           response = agent_team.run(user_input)
           st.markdown(response)
   else:
       st.warning("Please enter a question before submitting.")
