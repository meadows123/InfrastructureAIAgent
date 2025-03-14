import os
import logging
import streamlit as st
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain_community.chat_models import ChatOpenAI
#from langchain_community.llms import Ollama
import urllib3
from pydantic import BaseModel

# Import the tools and prompt templates from your agent scripts
from R1_agent import tools as r1_tools, prompt_template as r1_prompt
from R2_agent import tools as r2_tools, prompt_template as r2_prompt
from SW1_agent import tools as sw1_tools, prompt_template as sw1_prompt
from SW2_agent import tools as sw2_tools, prompt_template as sw2_prompt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
#Configure logging for better debugging
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

#llm = Ollama(model="command-r7b", temperature=0.3, base_url="http://ollama:11434")
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3)

# Initialize sub-agents for each device
r1_agent = initialize_agent(
    tools=r1_tools,
    llm=llm,
    agent='zero-shot-react-description',
    prompt=r1_prompt,
    verbose=True
)

r2_agent = initialize_agent(
    tools=r2_tools,
    llm=llm,
    agent='zero-shot-react-description',
    prompt=r2_prompt,
    verbose=True
)

sw1_agent = initialize_agent(
    tools=sw1_tools,
    llm=llm,
    agent='zero-shot-react-description',
    prompt=sw1_prompt,
    verbose=True
)

sw2_agent = initialize_agent(
    tools=sw2_tools,
    llm=llm,
    agent='zero-shot-react-description',
    prompt=sw2_prompt,
    verbose=True
)

def r1_agent_func(input_text: str) -> str:
    try:
        # Remove any device prefix and clean up the command
        if "R1:" in input_text:
            command = input_text.split("R1:", 1)[1].strip()
        else:
            command = input_text.strip()
            
        # Execute the command directly
        return r1_agent.run(command)
    except Exception as e:
        logging.error(f"R1 agent error: {str(e)}")
        return str(e)


def r2_agent_func(input_text: str) -> str:
    return r2_agent.run(f"R2: {input_text}")

def sw1_agent_func(input_text: str) -> str:
    return sw1_agent.run(f"SW1: {input_text}")

def sw2_agent_func(input_text: str) -> str:
    return sw2_agent.run(f"SW2: {input_text}")

# Define tools for each sub-agent
#r1_tool = Tool(name="R1 Agent", func=r1_agent_func, description="Use for Router R1 commands.")
r1_tool = Tool(
    name="R1 Agent",
    func=r1_agent_func,
    description="""Use this tool to execute commands on Router R1. 
    Input should be the command you want to execute (e.g., 'show running-config').
    Do not include any device prefix - just the command itself."""
)
r2_tool = Tool(name="R2 Agent", func=r2_agent_func, description="Use for Router R2 commands.")
sw1_tool = Tool(name="SW1 Agent", func=sw1_agent_func, description="Use for Switch SW1 commands.")
sw2_tool = Tool(name="SW2 Agent", func=sw2_agent_func, description="Use for Switch SW2 commands.")

# Create the master tool list
master_tools = [r1_tool, r2_tool, sw1_tool, sw2_tool]

# Update the master agent's prompt to be more specific
master_prompt = """You are a network automation assistant that helps execute commands on network devices.
When using device-specific tools:
1. Use the exact command without any device prefixes
2. For R1 commands, use the R1 Agent tool with just the command
3. Commands should be standard Cisco IOS commands

For example:
- To see R1's configuration, use: show running-config
- To check interfaces, use: show interfaces

Remember to use the exact command syntax as would be used on the device itself.
"""

# Update master agent initialization
master_agent = initialize_agent(
    tools=master_tools,
    llm=llm,
    agent="zero-shot-react-description",
    agent_kwargs={"prefix": master_prompt},
    verbose=True,
    handle_parsing_errors=True
)
logging.info(f"Master agent initialized with tools: {[tool.name for tool in master_tools]}")

# ============================================================
# Streamlit UI
# ============================================================

# Set up Streamlit UI
st.title("Network Engineer as Agents - Cisco IOS")
st.write("Operate your network with natural language")

user_input = st.text_input("Enter your question:")

# Initialize chat state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ""

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Button to send input
if st.button("Send"):
    if user_input:
        st.session_state.conversation.append({"role": "user", "content": user_input})

        try:
            # 🚀 Invoke the master agent
            response = master_agent.run(user_input)

            # Display results
            st.write(f"**Question:** {user_input}")
            st.write(f"**Answer:** {response}")

            # Update conversation history
            st.session_state.conversation.append({"role": "assistant", "content": response})
            st.session_state.chat_history = "\n".join(
                [f"{entry['role'].capitalize()}: {entry['content']}" for entry in st.session_state.conversation]
            )

        except Exception as e:
            st.write(f"An error occurred: {str(e)}")

# Display conversation history
if st.session_state.conversation:
    st.write("## Conversation History")
    for entry in st.session_state.conversation:
        st.write(f"**{entry['role'].capitalize()}:** {entry['content']}")

def execute_show_command(device: str, command: str) -> str:
    """
    Execute show commands with proper error handling and raw output option
    """
    try:
        # Add logic to handle raw output for commands that don't need parsing
        if command.lower() == "show running-config":
            # Return raw output without parsing
            return your_device_connection.send_command(command, read_timeout=120)
        # Handle other commands with parsing as needed
        return your_device_connection.send_command(command)
    except Exception as e:
        logging.error(f"Error executing command {command} on {device}: {str(e)}")
        raise