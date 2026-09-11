# Aster RAG

Production-oriented document Q&A with semantic chunking, BGE/E5 embeddings, Qdrant, cross-encoder reranking, grounded Grok responses, authentication, and admin query telemetry.

## Run locally

1. Create a `.env` based on `.env.example` and set a newly issued `XAI_API_KEY`. Do not reuse a credential shared in chat.
2. Install packages: `python -m pip install -r requirements.txt`
3. Start: `python -m uvicorn app:app --host 127.0.0.1 --port 8000`
4. Visit `http://127.0.0.1:8000`. The first registered user is the local administrator.

Without `XAI_API_KEY`, the app intentionally provides cited extractive previews instead of fabricating LLM answers. SentenceTransformer downloads the embedding/reranker models on first use.

## Production

Use `docker compose up --build` only for a local integration environment. Production requires managed PostgreSQL, Qdrant, Redis/rate limiting, object storage, a secret manager, TLS behind a WAF/CDN, background ingestion workers, monitoring, and load testing against provider-approved xAI quotas. Never expose the development SQLite/Qdrant paths to the public internet.
