from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logger import get_logger
import os 
from pathlib import Path

dir_path = Path(__file__).parent.parent / "fetched_data" / "text_docs"
os.makedirs(dir_path, exist_ok=True)
logger = get_logger(__name__)

def data_ingest():
    Dir_loader = DirectoryLoader(
        dir_path, 
        glob="**/*.txt",
        loader_cls=UnstructuredFileLoader
    )
    
    file_load = Dir_loader.load()
    logger.info("File Has been Loaded")
    return file_load

def splitter(list_of_docs, chunk_size=1000, chunk_overlap=200):
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n","\n"," "]
    )
    
    spilted_doc = splitter.split_documents(list_of_docs)
    logger.info("File Splitted Successfully")
    return spilted_doc
    
    
