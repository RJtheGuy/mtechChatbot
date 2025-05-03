from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import logging
from typing import Tuple, List, Dict, Optional

logger = logging.getLogger(__name__)

def build_vector_store(
    chunks: List[Dict], 
    model_name: str = "all-MiniLM-L6-v2", 
    index_size: Optional[int] = None
):
    try:
        from sentence_transformers import SentenceTransformer

        # Optionally truncate chunk list
        if index_size is not None:
            chunks = chunks[:index_size]

        # Load lightweight embeddings
        model = SentenceTransformer(
            model_name,
            device="cpu",
            cache_folder="/tmp/model_cache"
        )

        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        # Use smaller batch size
        embeddings = model.encode(texts, batch_size=8, show_progress_bar=False)

        # Create FAISS index directly
        import faiss
        import numpy as np
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))

        return index

    except Exception as e:
        logger.error(f"Vector store creation failed: {str(e)}")
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
