import streamlit as st
from phi.agent import Agent, RunResponse
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(override=True)

# Initialize agents
web_agent = Agent(
    name="Web Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGo()],
    markdown=True,
    instructions=["Always include sources"]
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get Financial Data",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    markdown=True,
    instructions=["Use tables to present data"]
)

agent_team = Agent(
    team=[web_agent, finance_agent],
    model=OpenAIChat(id="gpt-4o"),
    instructions=["Always include sources", "Use tables to display data"],
    markdown=True,
)

# Initialize conversation history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Finance and Stock Analysis Chatbot")
st.write("Ask any question about finance and stocks.")

# Display the chat history using st.chat_message (Streamlit v1.22+)
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# Get user input using st.chat_input
user_input = st.chat_input("Type your question here...")

if user_input:
    # Append user's new message to the conversation history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Build a conversation context by concatenating the history.
    conversation_context = ""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            conversation_context += f"User: {msg['content']}\n"
        else:
            conversation_context += f"Assistant: {msg['content']}\n"
    
    conversation_context += "\nBased on the above conversation, please provide your answer."
    
    with st.spinner("Processing..."):
        # Pass the entire conversation as the input prompt to the agent team.
        run: RunResponse = agent_team.run(conversation_context)
        ai_response = run.content

    # Append the AI's response to the conversation history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    
    # Use st.rerun to update the UI with the new messages
    st.rerun()
