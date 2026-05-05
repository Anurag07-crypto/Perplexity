from tavily import TavilyClient
from pathlib import Path
import uuid

from RAG.data_ingestion import data_ingest, splitter
from RAG.embedding_manager import EmbeddingManager
from RAG.vector_store import VectorStore
from RAG.retriever import Retriever

from langchain_groq import ChatGroq

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
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
if GROQ_API_KEY and TAVILY_API_KEY:
    logger.info("✅ API Imported Successfully")
    
else:
    logger.critical("❌ API not Imported Successfully")
    raise ValueError("Missing API keys")
            


class Extract_Pages:
    def __init__(self, client, llm, embedding_manager:EmbeddingManager, vector_store:VectorStore):
        self.client = client
        self.llm = llm 
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    def extract_pages(self, query:str):
        """Extract relevant web page contents and answer the query using RAG."""
        
        try:
            TAVILY_PROMPT = f'''
        Here is the query Extract the essential details according to the 
        Query:
        {query}
        '''
            
            search_results = self.client.search(TAVILY_PROMPT)
            urls = [res["url"] for res in search_results["results"]]
            extracted = self.client.extract(
                urls=urls,
                query=query
            )
            
            self.Web_page_extracted_text = "\n\n".join([r["content"] for r in extracted["results"]]) if extracted.get("results") else ""
            
            self.dir_path = Path(__file__).parent.parent / "fetched_data" / "text_docs"/f"webpage_{uuid.uuid4()}.txt"
            
            # Fix: Specify encoding as UTF-8 to handle Unicode characters
            with open(self.dir_path, "w", encoding="utf-8") as f:
                f.write(self.Web_page_extracted_text)

            docs = data_ingest()
            chunk = splitter(docs)
            texts = [doc.page_content for doc in chunk]
            embeddings = self.embedding_manager.generate_embeddings(texts)
            self.vector_store.add_documents(chunk, embeddings)
            
            retriever = Retriever(self.embedding_manager, self.vector_store)
            rag_response = retriever.retrieve(query)
            
            PROMPT = f"""You are a Helpful RAG assistant
        Answer me from this ***content***
        ***{rag_response}***
        and here is the ***query***
        {query}"""
            
            response = self.llm.invoke(PROMPT)
            
            # Return the response content properly
            return response.content
            
        except UnicodeEncodeError as e:
            return f"Error processing web pages: {str(e)}. Please try a different query."
        except Exception as e:
            return f"Error during webpage extraction: {str(e)}"
        
def extract(query):
    
    client = TavilyClient(api_key=TAVILY_API_KEY)
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
    
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()
    
    page_extraction_agent = Extract_Pages(client=client,
                                          llm=llm,
                                          embedding_manager=embedding_manager,
                                          vector_store=vector_store)
    
    extract_pages = page_extraction_agent.extract_pages(query)
    
    return extract_pages
    
    