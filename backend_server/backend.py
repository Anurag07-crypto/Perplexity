from fastapi import FastAPI, HTTPException
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from perplexitiy.main import perplexitiy_agent
from RAG.data_ingestion import data_ingest, splitter
from RAG.embedding_manager import EmbeddingManager
from RAG.vector_store import VectorStore
from RAG.retriever import Retriever
import uvicorn
from logger import get_logger
from pydantic import BaseModel

logger = get_logger(__name__)
app = FastAPI(title="server")

class Request(BaseModel):
    """Request Schema for /server end pipeline"""
    query: str
    tools: str

@app.post("/server")
def server(request: Request):
    """Handle incoming server queries through the RAG and perplexitiy pipeline.

    Args:
        request (QueryRequest): QueryRequest object containing the user query

    Raises:
        HTTPException: On runtime or unexpected errors

    Returns:
        Dict with 'response' key containing the agent's answer
    """
    
    try:
        response = perplexitiy_agent(request.query, tools=request.tools)
        logger.info("Request Accepted Successfully")        
   
        # Handle Unicode encoding properly
        return {"response": response}
    except RuntimeError as e:
        logger.error(f"Runtime error in /server: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except UnicodeEncodeError as e:
        logger.error(f"Unicode encoding error in /server: {e}")
        raise HTTPException(status_code=500, detail="Error processing response with special characters")
    except Exception as e:
        logger.error(f"Unexpected error in /server: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Something went wrong. please try again")

if __name__ == "__main__":
    
    uvicorn.run("backend:app", port=8000, host="127.0.0.1", reload=False)
