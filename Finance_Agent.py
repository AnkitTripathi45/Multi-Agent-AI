from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.openai import OpenAIChat
from dotenv import load_dotenv

load_dotenv(override=True)



web_agent = Agent(
    name="Web Agent",
    model=OpenAIChat(id="gpt-4o"),
    # model = Groq(id = "llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    show_tool_calls=True,
    markdown=True,
    instructions=["Alwasy include Sources"]
) 

# Set up the agent with the specified model
finance_agent = Agent(
    name="Finance Agent",
    role="Get Financial Data",
    # model = Groq(id = "llama-3.3-70b-versatile"),
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price= True, analyst_recommendations=True, stock_fundamentals=True)],
    show_tool_calls=True,
    markdown=True,
    instructions=["Use Tables to Present data"],
)

agent_team= Agent(
    team=[web_agent,finance_agent],
    # model = Groq(id = "llama-3.3-70b-versatile"),
    model=OpenAIChat(id="gpt-4o"),
    instructions=["Always Include Sources","Use Tables to Display Data"],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
)
agent_team.print_response("Summarise and Compare Analyse and give Recommendations and fundamentals for and which Stocks to buy betweenTSLA and AAPL")