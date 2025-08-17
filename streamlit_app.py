# streamlit_app.py
import os
import json
import time
import streamlit as st
import plotly.io as pio
from dotenv import load_dotenv

# project-local modules
from utils import load_data
import tools
from agent_builder import get_agent

# Load .env file for local development
load_dotenv()

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Agentic Data Analyst", layout="wide")
st.title("🤖 Agentic Data Analyst")

# --- Session State Initialization ---
if "df" not in st.session_state:
    st.session_state.df = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent_executor" not in st.session_state:
    try:
        st.session_state.agent_executor = get_agent()
    except Exception as e:
        st.session_state.agent_executor = None
        st.error(f"Failed to initialize the agent: {e}")

# --- Helper Functions ---
def render_chat_message(message):
    """Render a single chat message."""
    role = message.get("role", "assistant")
    with st.chat_message(role):
        content = message.get("content", "")
        plot_json = message.get("plot_json")

        if plot_json:
            try:
                fig = pio.from_json(plot_json)
                st.plotly_chart(fig, use_container_width=True)
            except (ValueError, json.JSONDecodeError):
                st.warning("Could not render plot. Displaying raw content.")
                st.write(content)
        elif isinstance(content, (dict, list)):
            st.json(content)
        else:
            st.markdown(str(content)) # Use markdown for better text rendering

def looks_like_plotly_json(s: str) -> bool:
    """Check if a string is likely a Plotly JSON object."""
    return isinstance(s, str) and "data" in s and "layout" in s and s.startswith("{")

def handle_agent_response(response):
    """Process and store the agent's response."""
    output = response.get("output", "Sorry, I encountered an error.")
    
    # The agent might return a string that is actually a JSON object (e.g., from a plot tool)
    if looks_like_plotly_json(output):
        st.session_state.chat_history.append({"role": "assistant", "plot_json": output})
    else:
        # Check if the output is a JSON string from another tool (e.g., summary)
        try:
            parsed_output = json.loads(output)
            st.session_state.chat_history.append({"role": "assistant", "content": parsed_output})
        except (json.JSONDecodeError, TypeError):
            # Treat as plain text
            st.session_state.chat_history.append({"role": "assistant", "content": output})


# --- Sidebar UI ---
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xls", "xlsx"])
    
    st.markdown("---")
    st.header("2. Quick Prompts")
    st.write("- Show top 10 vendors by total purchase quantity")
    st.write("- What is the average sale price by category?")
    st.write("- Plot a pie chart of sales by region")
    st.write("- Show me a heatmap of correlations")
    st.write("- Remove duplicate rows")
    st.write("- Fill missing age values with the median")

    st.markdown("---")
    if os.getenv("OPENAI_API_KEY"):
        st.success("OpenAI API key loaded.")
    else:
        st.error("OPENAI_API_KEY not found in .env file.")

    if st.session_state.df is not None:
        csv_bytes = st.session_state.df.to_csv(index=False).encode("utf-8")
        st.download_button("Download current dataset (CSV)", csv_bytes, "dataset.csv", "text/csv")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# --- Main App Logic ---

# Load data if uploaded
if uploaded_file and st.session_state.df is None:
    try:
        df = load_data(uploaded_file)
        st.session_state.df = df
        msg = tools.set_active_df(df)
        st.success(f"Successfully loaded `{uploaded_file.name}`. {msg}")
        with st.expander("Data Preview"):
            st.dataframe(df.head(), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load or process the file: {e}")

# Display chat history
for message in st.session_state.chat_history:
    render_chat_message(message)

# Handle user input
if prompt := st.chat_input("Ask about your data..."):
    if st.session_state.df is None:
        st.warning("Please upload a dataset first.")
        st.stop()
    
    # Add user message to history and display it
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    render_chat_message({"role": "user", "content": prompt})

    # Call the agent
    agent_executor = st.session_state.agent_executor
    if agent_executor:
        with st.spinner("Thinking..."):
            try:
                # The new agent executor expects a dictionary input
                response = agent_executor.invoke({
                    "input": prompt,
                    "chat_history": [
                        (msg["role"], str(msg["content"])) for msg in st.session_state.chat_history
                    ]
                })
                handle_agent_response(response)
            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})
        
        st.rerun()
    else:
        st.error("Agent is not available. Please check your configuration.")