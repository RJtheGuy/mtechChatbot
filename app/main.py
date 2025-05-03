from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
import logging
import time
from functools import lru_cache
from fastapi.middleware.cors import CORSMiddleware
from app.loader import load_and_chunk_readmes
from app.vector_db import build_vector_store, retrieve_relevant_text
from app.llm import TinyLlama
from app.config import settings
import re
import os
from pathlib import Path
import json
import resource

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Project summary configuration
PROJECT_SUMMARY_FILE = Path(__file__).parent / "project_summary.json"
SUMMARY_CACHE_DURATION = 3600  # 1 hour cache

app = FastAPI(
    title="MTech.ai Chatbot API",
    description="Optimized for Render Free Tier",
    version="1.2.0",
    docs_url="/docs" if os.getenv('RENDER') else None,
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    max_age=600
)

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)
    chat_history: Optional[List[Dict]] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict]
    context_used: bool
    processing_time: float

class LLMState:
    __slots__ = ['llm', 'vector_db', 'initialized', 'initializing', 'error']
    
    def __init__(self):
        self.llm = None
        self.vector_db = None
        self.initialized = False
        self.initializing = False
        self.error = None

state = LLMState()

SYSTEM_PROMPT = """<|system|>
You are MTech.ai's assistant. Rules:
1. Respond in under 50 words
2. List projects like: "[Project] - [Techs]"
3. Never say "I don't have access"</s>"""

@lru_cache(maxsize=50)
def get_cached_response(query: str, context: str) -> str:
    prompt = f"""{SYSTEM_PROMPT}
<|context|>
{context or 'No context'}</s>
<|user|>
{query}</s>
<|assistant|>"""
    return state.llm.generate(prompt, max_tokens=100)

def generate_project_summary():
    """Dynamically creates project summary from knowledge base"""
    summary = {}
    readme_dir = Path(settings.README_DIR)
    
    for md_file in readme_dir.glob("*.md"):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                project = {
                    "name": md_file.stem,
                    "technologies": extract_technologies(content),
                    "description": extract_description(content)
                }
                summary[md_file.stem] = project
        except Exception as e:
            logger.warning(f"Error processing {md_file}: {str(e)}")
    
    with open(PROJECT_SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(summary, f)
    return summary

def extract_technologies(content: str) -> list:
    """Extracts technology stack from markdown"""
    tech_section = re.search(r'## Technologies?\s*\n(.+?)(\n##|\Z)', content, re.DOTALL | re.I)
    if tech_section:
        return [t.strip() for t in re.split(r'[\n,-]', tech_section.group(1)) if t.strip()]
    return []

def extract_description(content: str) -> str:
    """Extracts first paragraph as description"""
    first_para = re.split(r'\n\n', content.strip())[0]
    return re.sub(r'#+\s*', '', first_para)[:200]

def initialize_services():
    """Render-optimized initialization"""
    if state.initializing or state.initialized:
        return

    state.initializing = True
    logger.info("Starting Render-optimized initialization...")

    try:
        # Set memory limits (critical for free tier)
        resource.setrlimit(resource.RLIMIT_AS, (400 * 1024 * 1024, 400 * 1024 * 1024))

        # Load components sequentially
        generate_project_summary()
        
        chunks = load_and_chunk_readmes(
            readme_dir=settings.README_DIR,
            chunk_size=200,
            overlap=30
        )  
        
        state.vector_db = build_vector_store(chunks, index_size=20)
        
        # CPU-only initialization
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Force CPU
        state.llm = TinyLlama()  # No 8-bit loading
        
        state.initialized = True
        logger.info("Render services initialized")
    except Exception as e:
        state.error = str(e)
        logger.critical(f"Initialization failed: {str(e)}")
    finally:
        state.initializing = False

@app.on_event("startup")
async def startup_event():
    """Delayed startup for free tier cold starts"""
    import asyncio
    await asyncio.sleep(1)
    initialize_services()

@app.get("/list-projects", response_model=ChatResponse)
async def list_projects():
    """Dynamic project listing endpoint"""
    start_time = time.time()
    try:
        # Regenerate summary if needed
        if (not PROJECT_SUMMARY_FILE.exists() or 
            time.time() - PROJECT_SUMMARY_FILE.stat().st_mtime > SUMMARY_CACHE_DURATION):
            summary = generate_project_summary()
        else:
            with open(PROJECT_SUMMARY_FILE, 'r', encoding='utf-8') as f:
                summary = json.load(f)
        
        response = "Current Projects:\n\n" + "\n".join(
            f"{i+1}. {proj['name']} - {proj['description']}\n   Tech: {', '.join(proj['technologies'][:5])}"
            for i, proj in enumerate(summary.values())
        )
        
        return {
            "response": response,
            "sources": [],
            "context_used": False,
            "processing_time": time.time() - start_time
        }
        
    except Exception as e:
        logger.warning(f"Project summary failed: {str(e)}")
        # Fallback to LLM
        return await full_project_query()

async def full_project_query():
    """LLM-powered fallback for project listing"""
    chunks, sources = retrieve_relevant_text("List all projects", state.vector_db, k=5)
    return {
        "response": clean_response(get_cached_response("List all projects", "\n".join(chunks))),
        "sources": sources,
        "context_used": True,
        "processing_time": time.time() - start_time
    }

@app.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    """Main query endpoint"""
    start_time = time.time()
    
    if not state.initialized:
        raise HTTPException(503, "Service warming up...")

    try:
        # Route project listing requests
        if any(keyword in request.query.lower() 
               for keyword in ["list projects", "show projects", "what projects"]):
            return await list_projects()
            
        # Regular queries
        chunks, sources = retrieve_relevant_text(request.query, state.vector_db, k=3)
        context = format_context(chunks, sources)
        
        return {
            "response": clean_response(get_cached_response(request.query, context)),
            "sources": sources,
            "context_used": bool(chunks),
            "processing_time": time.time() - start_time
        }

    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(500, "Query processing error")

def format_context(chunks: List[str], sources: List[Dict]) -> str:
    return "\n".join(
        f"{src['project']}:\n{chunk[:200]}..." 
        for chunk, src in zip(chunks, sources)
    )

def clean_response(response: str) -> str:
    return (
        response.split("<|assistant|>")[-1]
        .replace("<|", "")
        .replace("|>", "")
        .strip()[:500]
    )

@app.get("/status")
def get_status():
    return {
        "ready": state.initialized,
        "project_count": len(json.load(open(PROJECT_SUMMARY_FILE))) if PROJECT_SUMMARY_FILE.exists() else 0
    }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 4000)),
        workers=1,
        timeout_keep_alive=30
    )
