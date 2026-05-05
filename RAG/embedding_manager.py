from sentence_transformers import SentenceTransformer
import numpy as np 
from typing import List
from logger import get_logger

logger = get_logger(__name__)

class EmbeddingManager:
    """ 
    EmbeddingManager:Manages system embeddings using SentenceTransformer
    """
    
    def __init__(self, model_name = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.model = None
        self.load_model()
    
    def load_model(self):
        """
        Loading the BAAI/bge-small-en-v1.5 Embedding Model

        Raises:
            f: Model Not Loaded
        """
        
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Model Loaded Successfully: {self.model_name}")
        except Exception as e:
            logger.error(f"Model Not Loaded: {e}")
            raise RuntimeError(f"Model Not Loaded: {e}") from e
    
    def generate_embeddings(self, text:List[str])->np.ndarray:
        """Generating Embeddings using list of texts

        Args:
            text (List[str]): list of strings

        Raises:
            f: Embeddings not generated 

        Returns:
            np.ndarray: a array of embeddings
        """
        if not text:
            logger.error("Cannot generate empty embeddings")
            raise ValueError("Cannot generate empty embeddings")
        try:
            embeddings = self.model.encode(text)
            logger.info(f"Embeddings Generated Successfully for {len(text)} texts")
            return embeddings
        except Exception as e:
            logger.info(f"Embeddings not generated: {e}")
            raise RuntimeError(f"Embeddings not generated: {e}") from e
