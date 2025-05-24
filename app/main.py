from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
import logging
import time
import asyncio
from functools import lru_cache
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import re
import json
import resource
import pickle
import os
from contextlib import asynccontextmanager

# Import our modules
from app.hybrid_rag import HybridRAG
from app.loader import load_and_chunk_readmes
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model persistence
hybrid_rag = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - load models on startup"""
    global hybrid_rag
    try:
        logger.info("Initializing Hybrid RAG system...")
        hybrid_rag = HybridRAG()
        await hybrid_rag.initialize()
        logger.info("Hybrid RAG system initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise
    finally:
        if hybrid_rag:
            hybrid_rag.cleanup()

# FastAPI app with lifespan management
app = FastAPI(
    title="MTech Chatbot API", 
    docs_url=None, 
    redoc_url=None,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rjtheguy.github.io", "http://localhost:*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

class QueryRequest(BaseModel):
    query: str
    chat_history: Optional[List[Dict]] = None
    max_tokens: Optional[int] = 150
    temperature: Optional[float] = 0.7

class QueryResponse(BaseModel):
    response: str
    sources: List[Dict] = []
    processing_time: float
    retrieval_score: Optional[float] = None

# Cache for project summaries
@lru_cache(maxsize=1)
def get_project_summary():
    """Cached project summary parsing"""
    try:
        with open("project_summary.json", 'r') as f:
            return json.load(f)
    except:
        return parse_projects_fallback()

def parse_projects_fallback():
    """Fallback project parsing from markdown files"""
    projects = {}
    
    for md_file in Path("knowledge_base").glob("*.md"):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            project_name = md_file.stem
            description = re.split(r'\n\n', content.strip())[0]
            description = re.sub(r'^#+\s*', '', description)[:150]
            
            # Extract technologies
            tech_section = re.search(r'## Technologies?\s*Used\s*\n(.+?)(\n##|\Z)', content, re.DOTALL | re.I)
            techs = []
            if tech_section:
                tech_text = tech_section.group(1)
                techs = [t.strip('- ') for t in re.split(r'[\n,-]', tech_text) if t.strip()]
            
            projects[project_name] = {
                "name": project_name.title(),
                "description": description,
                "technologies": techs[:5]  # Limit to top 5
            }
        except Exception as e:
            logger.warning(f"Error parsing {md_file}: {e}")
    
    return projects

def should_use_retrieval(query: str) -> bool:
    """Determine if query needs RAG retrieval"""
    technical_keywords = [
        'project', 'technology', 'implementation', 'code', 'algorithm',
        'model', 'accuracy', 'feature', 'result', 'demo', 'github',
        'tensorflow', 'pytorch', 'python', 'streamlit', 'machine learning',
        'neural network', 'classification', 'prediction', 'sentiment', 'brain'
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in technical_keywords)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": hybrid_rag is not None,
        "timestamp": time.time()
    }

@app.get("/projects")
async def list_projects():
    """Get all projects summary"""
    projects = get_project_summary()
    return {
        "projects": list(projects.values()),
        "count": len(projects)
    }

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """Main query handling with hybrid RAG"""
    start_time = time.time()
    
    if not hybrid_rag:
        raise HTTPException(500, "RAG system not initialized")
    
    try:
        query = request.query.strip()
        
        # Handle simple project listing
        if re.search(r'\b(list|show|what)\s+(projects?|portfolio)\b', query, re.I):
            projects = get_project_summary()
            response = "Here are my current projects:\n\n" + "\n".join([
                f"• {p['name']}: {p['description']}\n  Technologies: {', '.join(p['technologies'][:3])}"
                for p in projects.values()
            ])
            
            return QueryResponse(
                response=response,
                sources=[],
                processing_time=time.time() - start_time
            )
        
        # Use hybrid RAG for technical queries
        if should_use_retrieval(query):
            response, sources, retrieval_score = await hybrid_rag.query(
                query=query,
                chat_history=request.chat_history or [],
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
        else:
            # Direct conversation without retrieval
            response = await hybrid_rag.generate_response(
                query=query,
                context="",
                chat_history=request.chat_history or [],
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            sources = []
            retrieval_score = None
        
        return QueryResponse(
            response=response,
            sources=sources,
            processing_time=time.time() - start_time,
            retrieval_score=retrieval_score
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(500, f"Query processing error: {str(e)}")

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    if not hybrid_rag:
        return {"error": "RAG system not initialized"}
    
    return hybrid_rag.get_cache_stats()

# Startup event
@app.on_event("startup")
async def startup():
    """Set resource limits"""
    try:
        # Limit memory usage to 1GB
        resource.setrlimit(resource.RLIMIT_AS, (1_000_000_000, 1_000_000_000))
        logger.info("Resource limits set successfully")
    except Exception as e:
        logger.warning(f"Could not set resource limits: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=1,
        access_log=False  # Reduce log overhead
    )