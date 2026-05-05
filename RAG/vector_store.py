import chromadb
import uuid
from typing import List, Any
import os
import numpy as np
from pathlib import Path
from logger import get_logger

logger = get_logger(__name__)

path = Path(__file__).parent.parent / "vector_store_data"

class VectorStore:
    """
    To store the info in the vector database
    """
    
    def __init__(self, 
                 persistant_dir:str=path,
                 collection_name:str="Text_Data"):
        
        self.persistant_dir = persistant_dir
        self.collection_name = collection_name
        self.collection = None
        self.client = None
        self.initialize_store()
        
    def initialize_store(self):
        """
        initialize store to store

        Raises:
            f: Store not initialized
        """
        
        try:
            os.makedirs(path, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(path))
            self.collection = self.client.get_or_create_collection(
                self.collection_name,
                metadata={"description":"Text Data For Rag"}
            )
            logger.info("Store initialized")
            
        except RuntimeError as e:
            logger.error(f"Store not initialized: {e}")
            raise RuntimeError(f"Store not initialized: {e}") from e 
        
    def add_documents(self, documents:List[Any], embeddings:np.ndarray):
        """
        adding documents in vector store

        Args:
            documents (List[Any]): Docs in list format to save in vector store
            embeddings (np.ndarray): embeddings

        Raises:
            ValueError: Documents and embeddings length should be same
            f: Document not added in Collection
        """
        
        if len(documents) != len(embeddings):
            logger.error("Documents and embeddings length should be same")
            raise ValueError("Documents and embeddings length should be same")
        
        ids = []
        metadatas = []
        document_texts = []
        embeddings_lists = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{i}_{uuid.uuid4().hex[:5]}"
            ids.append(doc_id)
            metadata = dict(doc.metadata) if hasattr(doc, 'metadata') else {}
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)
            document_texts.append(doc.page_content)
            # Convert embedding to list if it's numpy array
            if isinstance(embedding, np.ndarray):
                embeddings_lists.append(embedding.tolist())
            else:
                embeddings_lists.append(embedding)
            
        try:
            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=document_texts,
                embeddings=embeddings_lists
            )
            logger.info("Info Added in the collection")
            
        except RuntimeError as e:
            logger.critical(f"Document not added in Collection: {e}")
            raise RuntimeError(f"Document not added in Collection: {e}") from e
