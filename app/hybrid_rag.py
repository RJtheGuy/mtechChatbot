import torch
import numpy as np
import pickle
import hashlib
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import time
import asyncio
from functools import lru_cache

from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM

from app.loader import load_and_chunk_readmes
from app.config import settings

logger = logging.getLogger(__name__)

class EfficientVectorStore:
    """Memory-efficient FAISS-based vector store with caching"""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.index = None
        self.documents = []
        self.metadata = []
        self.embeddings_model = None
        
    def _get_cache_path(self, cache_key: str) -> Path:
        """Generate cache file path"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _generate_cache_key(self, chunks: List[Dict]) -> str:
        """Generate cache key from document chunks"""
        content_hash = hashlib.md5(
            str([c["text"][:100] for c in chunks[:10]]).encode()
        ).hexdigest()
        return f"faiss_index_{content_hash}_{len(chunks)}"
    
    async def build_or_load_index(self, chunks: List[Dict], model_name: str = "distilbert-base-uncased"):
        """Build FAISS index or load from cache"""
        cache_key = self._generate_cache_key(chunks)
        cache_path = self._get_cache_path(cache_key)
        
        # Try loading from cache first
        if cache_path.exists():
            try:
                logger.info("Loading FAISS index from cache...")
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                self.index = cached_data['index']
                self.documents = cached_data['documents'] 
                self.metadata = cached_data['metadata']
                
                # Load embedding model
                self.embeddings_model = SentenceTransformer(
                    'all-MiniLM-L6-v2',  # Lightweight alternative to DistilBERT
                    device='cpu',
                    cache_folder=str(self.cache_dir / "models")
                )
                
                logger.info(f"Loaded index with {len(self.documents)} documents")
                return
                
            except Exception as e:
                logger.warning(f"Cache loading failed: {e}")
        
        # Build new index
        logger.info("Building new FAISS index...")
        await self._build_index(chunks, cache_key, cache_path)
    
    async def _build_index(self, chunks: List[Dict], cache_key: str, cache_path: Path):
        """Build FAISS index from scratch"""
        try:
            # Use lightweight sentence transformer
            self.embeddings_model = SentenceTransformer(
                'all-MiniLM-L6-v2',
                device='cpu',
                cache_folder=str(self.cache_dir / "models")
            )
            
            # Extract texts and metadata
            texts = [chunk["text"] for chunk in chunks]
            self.metadata = [chunk["metadata"] for chunk in chunks]
            self.documents = texts
            
            # Generate embeddings in batches
            logger.info(f"Generating embeddings for {len(texts)} documents...")
            embeddings = self.embeddings_model.encode(
                texts,
                batch_size=16,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype('float32'))
            
            # Cache the index
            cache_data = {
                'index': self.index,
                'documents': self.documents,
                'metadata': self.metadata
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"FAISS index built and cached with {len(texts)} documents")
            
        except Exception as e:
            logger.error(f"Index building failed: {e}")
            raise
    
    def search(self, query: str, k: int = 3) -> Tuple[List[str], List[Dict], List[float]]:
        """Search similar documents"""
        if not self.index or not self.embeddings_model:
            raise ValueError("Index not initialized")
        
        try:
            # Encode query
            query_embedding = self.embeddings_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding.astype('float32'), k)
            
            # Return results
            retrieved_docs = [self.documents[i] for i in indices[0]]
            retrieved_metadata = [self.metadata[i] for i in indices[0]]
            similarity_scores = scores[0].tolist()
            
            return retrieved_docs, retrieved_metadata, similarity_scores
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return [], [], []

class HybridRAG:
    """Hybrid RAG system with DistilBERT retrieval and TinyLlama generation"""
    
    def __init__(self):
        self.vector_store = EfficientVectorStore()
        self.llm_model = None
        self.llm_tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.query_cache = {}  # Simple query caching
        self.cache_hits = 0
        self.total_queries = 0
        
    async def initialize(self):
        """Initialize all components"""
        try:
            # Load and process documents
            logger.info("Loading documents...")
            chunks = load_and_chunk_readmes(
                readme_dir=str(settings.README_DIR),
                chunk_size=settings.CHUNK_SIZE,
                overlap=settings.CHUNK_OVERLAP
            )
            
            if not chunks:
                raise ValueError("No documents found to index")
            
            # Build vector store
            await self.vector_store.build_or_load_index(chunks)
            
            # Initialize LLM (lazy loading)
            self._load_llm()
            
            logger.info("Hybrid RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"RAG initialization failed: {e}")
            raise
    
    def _load_llm(self):
        """Load TinyLlama model with optimization"""
        try:
            logger.info("Loading TinyLlama model...")
            
            # Load tokenizer
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                settings.LLM_MODEL,
                padding_side="left",
                cache_dir="./cache/models"
            )
            
            if not self.llm_tokenizer.pad_token:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            # Load model with optimizations
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                settings.LLM_MODEL,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                cache_dir="./cache/models",
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.llm_model = self.llm_model.to(self.device)
            
            logger.info("TinyLlama model loaded successfully")
            
        except Exception as e:
            logger.error(f"LLM loading failed: {e}")
            raise
    
    def _get_query_hash(self, query: str, context: str) -> str:
        """Generate hash for query caching"""
        combined = f"{query}|{context[:200]}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def query(self, query: str, chat_history: List[Dict] = None, 
                   max_tokens: int = 150, temperature: float = 0.7) -> Tuple[str, List[Dict], float]:
        """Main RAG query method"""
        self.total_queries += 1
        
        try:
            # Retrieve relevant documents
            retrieved_docs, metadata, scores = self.vector_store.search(query, k=3)
            
            if not retrieved_docs:
                # Fallback to direct generation
                response = await self.generate_response(query, "", chat_history, max_tokens, temperature)
                return response, [], 0.0
            
            # Prepare context
            context = "\n\n".join(retrieved_docs[:2])  # Use top 2 most relevant
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            # Check cache
            query_hash = self._get_query_hash(query, context)
            if query_hash in self.query_cache:
                self.cache_hits += 1
                cached_response = self.query_cache[query_hash]
                return cached_response, metadata, avg_score
            
            # Generate response
            response = await self.generate_response(query, context, chat_history, max_tokens, temperature)
            
            # Cache response
            self.query_cache[query_hash] = response
            if len(self.query_cache) > 100:  # Limit cache size
                oldest_key = next(iter(self.query_cache))
                del self.query_cache[oldest_key]
            
            return response, metadata, avg_score
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return "I encountered an error processing your query. Please try again.", [], 0.0
    
    async def generate_response(self, query: str, context: str, chat_history: List[Dict] = None,
                              max_tokens: int = 150, temperature: float = 0.7) -> str:
        """Generate response using TinyLlama"""
        try:
            # Build conversation context
            history_text = ""
            if chat_history:
                recent_history = chat_history[-3:]  # Last 3 exchanges
                for exchange in recent_history:
                    if exchange.get("user"):
                        history_text += f"Human: {exchange['user']}\n"
                    if exchange.get("assistant"):
                        history_text += f"Assistant: {exchange['assistant']}\n"
            
            # Create prompt
            if context:
                prompt = f"""Based on the following information about my projects:

{context}

{history_text}Human: {query}
Assistant: """
            else:
                prompt = f"""{history_text}Human: {query}
Assistant: """
            
            # Generate response
            inputs = self.llm_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=40,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            # Clean up response
            if not response:
                response = "I'd be happy to help you with information about my projects. Could you please be more specific about what you'd like to know?"
            
            return response
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I'm having trouble generating a response right now. Please try rephrasing your question."
    
    def get_cache_stats(self) -> Dict:
        """Get caching statistics"""
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(self.total_queries, 1),
            "cached_responses": len(self.query_cache),
            "vector_store_size": len(self.vector_store.documents) if self.vector_store.documents else 0
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'llm_model') and self.llm_model:
            del self.llm_model
        if hasattr(self, 'llm_tokenizer'):
            del self.llm_tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None