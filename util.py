# Import necessary modules for LangChain integration
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import ChatMessage
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.runnables import RunnableConfig
import asyncio
from langchain_core.prompts import PromptTemplate
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import os

if os.environ.get('RUNNING_IN_DOCKER') == 'true':
    OLLAMA_BASE_URL = "http://host.docker.internal:11434/"
    python_path = "python"
    ddg_server_path = "/app/duckduckgo_mcp_server"
else:
    OLLAMA_BASE_URL = "http://localhost:11434/"
    python_path = "python"
    ddg_server_path = "duckduckgo_mcp_server"

# Define StreamHandler class for streaming agent responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# Define function to create LangChain agent
async def create_agent_aws():
    local_llm = "llama3.2:3b"
    
    try:
        client = MultiServerMCPClient(
            {
                "duck_duck_go": {
                    "command": python_path,
                    "args": [
                        f"{ddg_server_path}/server.py"
                    ],
                    "transport": "stdio"
                }
            }
        )

        tools = await client.get_tools()

        sys_prompt = f"""I am a helpful AI assistant specializing in providing recent and accurate information to user queries.
                         For the user enquiry, I can use duckduckgo tool to lookup the internet and fetch the latest information.
                         I will not generate fictional or hypothetical information.
                         The answer will be the factual data the user is looking for, not just the links.
                         Tools available:
                            {tools}
                        """

        llm = ChatOllama(model=local_llm, temperature=0, base_url=OLLAMA_BASE_URL)

        agent = create_agent(model=llm, tools=tools, system_prompt=sys_prompt)
        return agent
    except Exception as e:
        # Fallback to agent without tools if MCP server fails
        print(f"Warning: MCP server initialization failed: {e}")
        print("Creating agent without external tools...")
        
        llm = ChatOllama(model=local_llm, temperature=0, base_url=OLLAMA_BASE_URL)
        
        sys_prompt = """I am a helpful AI assistant specializing in providing stock market analysis and insights.
                         I can help answer questions about stock performance, market trends, and provide context for gains and losses.
                         I will provide thoughtful analysis based on the information provided in the query.
                        """
        
        agent = create_agent(model=llm, tools=[], system_prompt=sys_prompt)
        return agent

# Define helper function to run async tasks
def run_async(coro, loop=None):
    """Run an async function within the provided event loop."""
    if loop is None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# Define function to run agent with user input
async def run_agent(user_input, agent, thread_id):
    config = RunnableConfig(configurable={"thread_id": thread_id})
    response = await agent.ainvoke({"messages": [HumanMessage(content=user_input)]}, config=config)
    return response['messages'][-1].content

# Define function to handle AI-powered search in Streamlit
def ai_search_tab(st, st_session_state):
    st.subheader("🤖 AI-Powered Search")

    # Display previous messages
    for msg in st_session_state.messages:
        st.chat_message(msg.role).write(msg.content)

    # Handle new user input
    if prompt := st.text_input("Ask a question about gains/losses or stocks"):
        st_session_state.messages.append(ChatMessage(role="user", content=prompt))
        st.chat_message("user").write(prompt)

        with st.spinner("Agent thinking..."):
            st_callback = StreamlitCallbackHandler(st.container())
            agent_response = run_async(run_agent(prompt, st_session_state.agent, thread_id=12345678), st_session_state.loop)
            with st.chat_message("assistant"):
                st.markdown(agent_response)
                st_session_state.messages.append(ChatMessage(role="assistant", content=agent_response))