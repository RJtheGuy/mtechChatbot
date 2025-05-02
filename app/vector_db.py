from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import logging
from typing import Tuple, List, Dict

logger = logging.getLogger(__name__)

def build_vector_store(chunks: List[Dict], model_name: str = "all-MiniLM-L6-v2") -> FAISS:
    """Build FAISS vector store from chunks with proper error handling."""
    try:
        logger.info(f"Initializing embeddings with model: {model_name}")
        embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
        
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        if not texts or not metadatas:
            raise ValueError("No valid texts or metadata provided")

        logger.info(f"Creating vector store with {len(texts)} chunks...")
        faiss_db = FAISS.from_texts(
            texts=texts,
            embedding=embeddings_model,
            metadatas=metadatas
        )

        if not faiss_db:
            raise RuntimeError("FAISS vector store creation failed")

        logger.info("Vector store created successfully")
        return faiss_db

    except Exception as e:
        logger.error(f"Vector store creation failed: {str(e)}", exc_info=True)
        raise

def retrieve_relevant_text(query: str, db: FAISS, k: int = 3) -> Tuple[List[str], List[Dict]]:
    """Retrieve top-k relevant documents with proper typing and error handling."""
    try:
        if not db:
            raise ValueError("Vector database not initialized")
        
        results = db.similarity_search(query, k=k)
        
        chunks = [doc.page_content for doc in results]
        sources = [doc.metadata for doc in results]
        
        return chunks, sources

    except Exception as e:
        logger.error(f"Text retrieval failed: {str(e)}", exc_info=True)
        raise