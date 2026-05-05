from tavily import TavilyClient
from dotenv import load_dotenv
import os
import sys 
from pathlib import Path
from logger import get_logger

sys.path.insert(0, str(Path(__file__).parent.parent))
logger = get_logger(__name__)

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY :
    logger.critical("Missing API keys")
    raise ValueError("Missing API keys")


def deep_search(query:str):
    """Search the web deeply, store results, and answer the query ."""
    client = TavilyClient(api_key=TAVILY_API_KEY)

    try:
        deep_search_results = client.search(query=query,
                                                  search_depth="advanced")
        
        
        deep_search_text = "\n\n".join([r["content"] for r in deep_search_results["results"]]) if deep_search_results.get("results") else ""
        
        dir_path = Path(__file__).parent.parent / "fetched_data" / "deep_Search_results"/f"deep_search.txt"
        
        # Fix: Specify encoding as UTF-8 to handle Unicode characters
        with open(dir_path, "w", encoding="utf-8") as f:
            f.write(deep_search_text)

        return deep_search_text
    
    except UnicodeEncodeError as e:
        return f"Error processing search results: {str(e)}. Please try a different query."
    except Exception as e:
        return f"Error during deep search: {str(e)}"
