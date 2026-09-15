# ==============================================================================
# Astra RAG + AI Agent - Production Gunicorn Configuration
# ==============================================================================
import multiprocessing
import os

# Server Socket
bind = os.getenv("GUNICORN_BIND", "127.0.0.1:8000")
backlog = 2048

# Worker Processes
# For memory efficiency with RAG models, use 2-4 workers on standard VPS instances
cpu_count = multiprocessing.cpu_count()
default_workers = max(2, min(cpu_count * 2 + 1, 8))
workers = int(os.getenv("GUNICORN_WORKERS", default_workers))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000

# Timeouts
# Set to 300s to allow large file chunking, embeddings, and complex LLM reasoning
timeout = int(os.getenv("GUNICORN_TIMEOUT", "300"))
keepalive = 65
graceful_timeout = 30

# Memory Management & Worker Recycling
# Periodically restarts workers to prevent memory leaks from heavy embedding runs
max_requests = 2000
max_requests_jitter = 200

# Logging
# Log to standard output/error so systemd journal catches everything cleanly
loglevel = os.getenv("LOG_LEVEL", "info").lower()
accesslog = "-"
errorlog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" (%(D)s µs)'

# Process Naming
proc_name = "astra_ai_production"

# Preloading
# Preloading disabled for Uvicorn async workers to avoid shared event loop issues
preload_app = False
