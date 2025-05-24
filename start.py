#!/usr/bin/env python3
"""
Optimized startup script for MTech Chatbot
Handles resource management and graceful startup
"""

import os
import sys
import logging
import resource
import torch
from pathlib import Path

def setup_environment():
    """Configure environment for optimal performance"""
    
    # Set threading limits
    torch.set_num_threads(2)
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Reduce memory usage
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # Create necessary directories
    Path("cache").mkdir(exist_ok=True)
    Path("cache/models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def set_resource_limits():
    """Set system resource limits"""
    try:
        # Memory limit: 1GB
        memory_limit = 1024 * 1024 * 1024  # 1GB in bytes
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        
        # File descriptor limit
        resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 1024))
        
        print(f"✓ Resource limits set: Memory={memory_limit//1024//1024}MB")
        
    except Exception as e:
        print(f"⚠ Could not set resource limits: {e}")

def check_dependencies():
    """Verify all dependencies are available"""
    required_packages = [
        'fastapi', 'uvicorn', 'transformers', 
        'sentence_transformers', 'torch', 'faiss'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"✗ {package}")
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)

def main():
    """Main startup function"""
    print("🚀 Starting MTech Chatbot...")
    
    # Setup
    setup_environment()
    setup_logging()
    set_resource_limits()
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    check_dependencies()
    
    # Start server
    print("\n🌐 Starting server...")
    
    try:
        import uvicorn
        from app.main import app
        from app.config import settings
        
        uvicorn.run(
            app,
            host=settings.HOST,
            port=settings.PORT,
            workers=1,  # Single worker for resource efficiency
            access_log=False,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"💥 Startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()