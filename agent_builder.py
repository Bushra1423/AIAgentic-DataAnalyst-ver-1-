# agent_builder.py
import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import tools

load_dotenv()

# Load OpenAI API key from .env
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

llm = ChatOpenAI(
    openai_api_key=api_key,
    model_name="gpt-4o-mini",
    temperature=0
)

# ------------------- TOOL DEFINITIONS -------------------
# With the @tool decorator, we can just list the functions.
tools_list = [
    tools.preview_head,
    tools.dataset_summary,
    tools.get_unique_values,
    tools.run_sql_query,
    tools.filter_rows,
    tools.plot_histogram,
    tools.plot_scatter,
    tools.plot_boxplot,
    tools.plot_bar,
    tools.plot_line,
    tools.plot_pie,
    tools.plot_heatmap,
    tools.plot_stacked_bar,
    tools.plot_area,
    # Newly integrated data cleaning tools
    tools.drop_duplicates,
    tools.fill_missing_values,
    tools.convert_column_types,
    tools.trim_whitespace,
    tools.detect_outliers,
]

# ------------------- SYSTEM MESSAGE / PROMPT -------------------
# This prompt is crucial for directing the agent's behavior.
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert AI Data Analyst. Your goal is to help users understand, clean, and visualize their data.

- You have access to a suite of tools for data analysis, cleaning, and plotting.
- When a user asks a question, first understand the user's intent.
- Then, decide which tool is most appropriate to answer the question, using the provided tool descriptions.
- You MUST use the tools' Pydantic argument schemas. Call the tool with the correct keyword arguments.
- When a user asks for a plot, ALWAYS use a plotting tool. Do not describe the plot in text or refuse.
- If the data is changed (e.g., by cleaning), inform the user about what happened in your response.
- Think step-by-step. If a task requires multiple steps (e.g., filter then plot), do one at a time.
- If you are unsure about column names or data types, use the `dataset_summary` tool to understand the data first.
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ------------------- AGENT INITIALIZATION -------------------
# Create the agent using the modern LangChain constructor
agent = create_openai_tools_agent(llm, tools_list, prompt)

# Create the Agent Executor, which runs the agent and its tools
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools_list,
    verbose=True,
    handle_parsing_errors=True # Handles errors if the LLM output is not parsable
)

def get_agent():
    return agent_executor