from RAG.embedding_manager import EmbeddingManager
from RAG.vector_store import VectorStore
from typing import List, Any, Dict
from logger import get_logger

logger = get_logger(__name__)

class Retriever:
    """
    RAG Retriever for retrieving required data from the source
    """
    
    def __init__(self,
                 embedding_manager:EmbeddingManager,
                 vector_store:VectorStore):
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store
    
    def retrieve(self, query:str, top_k:int=5, threshold:float = 0.3)-> List[Dict[str, Any]]:
        """Retrieve Function for retrieving function

        Args:
            query (str): user query
            top_k (int, optional): number of results to show . Defaults to 5.
            threshold (float, optional): minimum score to classify the answer. Defaults to 0.3.

        Raises:
            f: File not Retrieved

        Returns:
            List[Dict[str, Any]]: List of response in Dictionary format
        """
        
        try:
            embeddings = self.embedding_manager.generate_embeddings([query])[0]
            result = self.vector_store.collection.query(
                query_embeddings=embeddings,
                n_results=top_k,
                include=["metadatas", "documents", "distances"]
            )
            
            retrieved_docs = []
            
            # Check if we got results - result structure is {"documents": [[docs]], "metadatas": [[metadata]], ...}
            if result["documents"] and len(result["documents"]) > 0 and len(result["documents"][0]) > 0:
                documents = result["documents"][0]  # Extract from nested list
                metadatas = result["metadatas"][0]  # Extract from nested list
                distances = result["distances"][0]  # Extract from nested list
                ids = result["ids"][0]  # Extract from nested list
                
                for i, (doc, metadata, distance, doc_id) in enumerate(zip(
                    documents, metadatas, distances, ids
                )):
                    similarity_score = 1 - distance

                    if similarity_score < threshold:
                        continue

                    retrieved_docs.append(
                        {
                            "id": doc_id,
                            "content": doc,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            "distance": distance,
                            "rank": i + 1,
                        }
                    )

            if not retrieved_docs:
                logger.warning("No relevant documents found for query")
                return []
 
            logger.info(
                f"Retrieved {len(retrieved_docs)} documents"
            )
            return retrieved_docs
        
        except Exception as e:
            logger.critical(F"File not Retrieved: {e}")
            raise RuntimeError(f"File not Retrieved: {e}") from e

                
