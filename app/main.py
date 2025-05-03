from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import time
from functools import lru_cache
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import re
import json
import resource
import asyncio

# Configuration
app = FastAPI(docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://rjtheguy.github.io", "http://localhost:*"],
    allow_methods=["POST"]
)

class QueryRequest(BaseModel):
    query: str
    chat_history: Optional[List[Dict]] = None

class QueryResponse(BaseModel):
    response: str
    sources: List[Dict] = []
    processing_time: float

# Project Parser
def parse_projects():
    projects = []
    technologies = set()
    
    for md_file in Path("knowledge_base").glob("*.md"):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Extract project name from filename
                project_name = md_file.stem.replace("-", " ").title()
                
                # Extract description (first paragraph)
                description = re.split(r'\n\n', content.strip())[0]
                description = re.sub(r'^#+\s*', '', description)[:150]
                
                # Extract technologies
                tech_section = re.search(r'## Technologies?\s*\n(.+?)(\n##|\Z)', content, re.DOTALL | re.I)
                if tech_section:
                    techs = [t.strip() for t in re.split(r'[\n,-]', tech_section.group(1)) if t.strip()]
                    technologies.update(techs)
                else:
                    techs = []
                
                projects.append({
                    "name": project_name,
                    "description": description,
                    "technologies": techs
                })
        except Exception as e:
            logging.error(f"Error parsing {md_file}: {e}")
    
    return projects, sorted(technologies)

# Unified Query Handler
@app.post("/query")
async def handle_query(request: QueryRequest):
    start_time = time.time()
    
    # Detect project listing requests
    if re.search(r'\b(list|show|what)\s+projects?\b', request.query, re.I):
        projects, techs = parse_projects()
        response = "Current Projects:\n\n" + "\n".join(
            f"{i+1}. {p['name']} - {p['description']}\n   Tech: {', '.join(p['technologies'][:3])}"
            for i, p in enumerate(projects)
        )
        return QueryResponse(
            response=response,
            sources=[],
            processing_time=time.time() - start_time
        )
    
    # Normal queries
    try:
        # Your existing query logic here
        # For demo, using simple response
        return QueryResponse(
            response=f"I received your query: {request.query}",
            sources=[{"source": "demo"}],
            processing_time=time.time() - start_time
        )
    except Exception as e:
        logging.error(f"Query failed: {e}")
        raise HTTPException(500, "Query processing error")

# Startup
@app.on_event("startup")
async def startup():
    resource.setrlimit(resource.RLIMIT_AS, (800_000_000, 800_000_000))  

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)