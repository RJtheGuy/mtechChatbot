FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set environment variables
ENV README_DIR=/app/knowledge_base
ENV PORT=8000
ENV HF_SPACE=true

# Memory optimization for Hugging Face
RUN echo "vm.overcommit_memory = 1" >> /etc/sysctl.conf

# Expose port
EXPOSE 8000

# Run with 1 worker to stay within memory limits
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]