"""
Service layer for business logic separation.
Handles RAG operations, LLM interactions, and prompt construction.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_groq import ChatGroq
from RAG.retriever import Retriever
from RAG.embedding_manager import EmbeddingManager
from RAG.vector_store import VectorStore
from logger import get_logger

logger = get_logger(__name__)


class RAGService:
    """Service for Retrieval-Augmented Generation operations."""
    
    def __init__(self, embedding_manager: EmbeddingManager, vector_store: VectorStore):
        """Initialize RAG service with embedding manager and vector store."""
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
        self.retriever = Retriever(embedding_manager=embedding_manager, vector_store=vector_store)
    
    def retrieve_documents(self, query: str):
        """
        Retrieve relevant documents from vector store.
        
        Args:
            query: The search query
            
        Returns:
            List of relevant documents
            
        Raises:
            Exception: If retrieval fails
        """
        try:
            results = self.retriever.retrieve(query)
            logger.info(f"Retrieved {len(results)} documents for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise
    
    def extract_context(self, documents):
        """
        Extract text content from document objects.
        
        Args:
            documents: List of document objects (dict or LangChain Document)
            
        Returns:
            Concatenated context string
        """
        context_parts = []
        for doc in documents:
            if isinstance(doc, dict):
                # If retriever returns dict format
                content = doc.get('content') or doc.get('document', '')
            elif hasattr(doc, 'page_content'):
                # If it's a LangChain Document
                content = doc.page_content
            else:
                content = str(doc)
            
            if content:
                context_parts.append(content)
        
        return "\n\n".join(context_parts)


class LLMService:
    """Service for LLM interactions and prompt management."""
    
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        """Initialize LLM service with specified model."""
        self.llm = ChatGroq(model=model_name)
        self.model_name = model_name
    
    def build_rag_prompt(self, query: str, context: str) -> str:
        """
        Build a structured prompt for RAG-based question answering.
        
        Args:
            query: User's question
            context: Retrieved context from documents
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""### You are a 'Personal Assistant' with over '5+' years of experience

Whenever a person asks you any question, you must answer properly using the following context:

**Context:**
{context}

**User Query:**
{query}

**Constraints:**
- Don't give any false information
- Only use the provided context to answer
- This constraint is strictly followed

**Response Format:**
- Answer in Markdown format with clear, beautiful formatting
- Answer directly without greetings or repeated introductions
"""
        return prompt
    
    def invoke_llm(self, prompt: str) -> str:
        """
        Invoke the LLM with a prompt and return the response.
        
        Args:
            prompt: The prompt to send to LLM
            
        Returns:
            LLM response text
            
        Raises:
            Exception: If LLM invocation fails
        """
        try:
            response = self.llm.invoke(prompt)
            logger.info("LLM invocation successful")
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Error invoking LLM: {str(e)}")
            raise
    
    def answer_question(self, query: str, context: str) -> str:
        """
        Generate an answer to a question using provided context.
        
        Args:
            query: User's question
            context: Retrieved context
            
        Returns:
            LLM-generated answer
        """
        prompt = self.build_rag_prompt(query, context)
        return self.invoke_llm(prompt)
