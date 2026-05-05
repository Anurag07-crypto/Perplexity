
from langchain_core.messages import  HumanMessage, ToolMessage
import json

from perplexitiy.deep_research_agent import deep_search
from perplexitiy.email_agent import mail_agent
from perplexitiy.webpages_extract_agent import extract

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
import os  
from dotenv import load_dotenv
from logger import get_logger
import sys 
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logger = get_logger(__name__)

def execute_tool(tool_name: str, tool_input: dict):
    """Execute a tool with the given parameters"""
    # Tool registry
    TOOL_REGISTRY = {
        "deep_search": deep_search,
        "mail_sender": mail_agent,
        "extract_pages": extract
    }
    
    if tool_name not in TOOL_REGISTRY:
        return f"Error: Tool '{tool_name}' not found"
    
    try:
        tool_func = TOOL_REGISTRY[tool_name]
        
        # Extract arguments from tool_input
        if isinstance(tool_input, str):
            tool_input = json.loads(tool_input)
        
        # Call the tool with unpacked arguments
        result = tool_func(**tool_input)
        return result
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
        return f"Error executing {tool_name}: {str(e)}"


def perplexitiy_agent(query: str, tools: str):
    """Execute the agent with tool support and return the final response"""
    load_dotenv()
    GROQ_API_KEY=os.getenv("GROQ_API_KEY")

    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
    
    try:
        # Get selected tools
        if tools == "deep_search":
            tools_list = [deep_search]
        elif tools == "mail_sender":
            tools_list = [mail_agent]
        elif tools == "extract_pages":
            tools_list = [extract]
        else:
            tools_list = []
        
        # Bind tools to LLM
        PROMPT = f"""You are a helpful assistant
        here is the query:
        {query}
        give me the full answer giving by the tool
        """
        agent = create_agent(
            model = llm,
            tools=tools_list,
            checkpointer=InMemorySaver()
        )
        
        config = {"configurable":{"thread_id":"test_1"}}
        # Initialize messages
        
        messages = [HumanMessage(content=query)]
        
        # Agentic loop - keep running until no more tool calls
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Get response from LLM
            response = agent.invoke({"messages": messages},
                                    config=config)
            ai_msg = response["messages"][-1]
            messages.append(ai_msg)
            
            # Check if there are tool calls
            if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
                for tool_call in ai_msg.tool_calls:
                    tool_name = tool_call["name"]
                    tool_input = tool_call["args"]
                    
                    logger.info(f"Executing tool: {tool_name} with input: {tool_input}")
                    
                    # Execute the tool
                    tool_result = execute_tool(tool_name, tool_input)
                    
                    # Add tool result to messages
                    messages.append(ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"]
                    ))
            else:
                # No tool calls, return the final response
                final_response = ai_msg.content 
                logger.info("Agent completed successfully")
                return final_response
        
        # If we hit max iterations, return the last response
        logger.warning("Max iterations reached")
        return response.content if hasattr(response, 'content') else str(response)
        
    except Exception as e:
        logger.error(f"Error in perplexitiy_agent: {e}", exc_info=True)
        return f"Error: {str(e)}"
