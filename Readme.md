# Multi Agentic AI - Finance and Stock Analysis System

## Overview
This project is an advanced financial analysis system that combines multiple AI agents to provide comprehensive stock market analysis and financial insights. It uses a team of specialized agents working together to gather and analyze financial data, market news, and stock information.

## Features
- **Multi-Agent System**: Utilizes multiple specialized AI agents:
  - Web Agent: Gathers real-time information from the internet
  - Finance Agent: Analyzes financial data and stock metrics
  - Team Agent: Coordinates between agents for comprehensive analysis

- **Real-time Stock Analysis**: 
  - Live stock price tracking
  - Company fundamentals analysis
  - Analyst recommendations
  - Market sentiment analysis

- **Interactive Web Interface**:
  - Built with Streamlit for user-friendly interaction
  - Chat-based interface for natural language queries
  - Dynamic data visualization
  - Conversation history tracking

## Technology Stack
- Python 3.x
- Streamlit for web interface
- OpenAI GPT-4 for AI processing
- YFinance for financial data
- DuckDuckGo for web searches
- Phidata framework for agent management

## Prerequisites
- Python 3.x
- OpenAI API key
- Internet connection for real-time data

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd Multi-Agentic-AI
```

2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install required packages:
```bash
pip install -r require.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory and add your API keys:
```
OPENAI_API_KEY=your_openai_api_key
```

## Usage

1. Start the application:
```bash
streamlit run run.py
```

2. Access the web interface through your browser (typically http://localhost:8501)

3. Enter your financial queries in natural language, such as:
   - "Compare Tesla and Apple stocks"
   - "What are the latest analyst recommendations for MSFT?"
   - "Analyze the fundamentals of NVDA"

## Project Structure
- `run.py`: Main application file with Streamlit interface
- `Finance_Agent.py`: Finance agent configuration and setup
- `app.py`: Application logic and agent coordination
- `agent.py`: Base agent configuration
- `require.txt`: Project dependencies

## Features in Detail

### Web Agent
- Real-time web scraping for latest news and information
- Source verification and citation
- Market sentiment analysis

### Finance Agent
- Stock price tracking and analysis
- Company fundamental analysis
- Technical indicators
- Analyst recommendations

### Team Agent
- Coordinates between specialized agents
- Combines information for comprehensive analysis
- Provides formatted, easy-to-understand responses

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.


## Disclaimer
This tool is for informational purposes only. Do not make financial decisions solely based on this system's recommendations. Always conduct your own research and consult with financial advisors.
