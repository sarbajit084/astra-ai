import asyncio
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from rag_engine import ProductionRAGService

def run_tests():
    print("=== Testing Production RAG Engine ===")
    rag = ProductionRAGService()

    # 1. Health check
    health = rag.health()
    print(f"[PASS] Health check: {health}")
    assert health["vector_store"] in ["local-persistent", "remote-cluster"]

    # 2. Semantic chunking test
    sample_text = (
        "QuantumPulse is an enterprise low-latency distributed streaming broker.\n\n"
        "It achieves ultra-reliable message delivery with a p99 latency of 120 milliseconds. "
        "The architecture is designed to handle up to 200,000 concurrent streaming subscribers without degraded throughput.\n\n"
        "Security and Compliance:\n"
        "The system complies with GDPR, HIPAA, and SOC2 Type II standards, implementing end-to-end AES-256 encryption."
    )
    chunks = rag._semantic_chunks(sample_text)
    print(f"[PASS] Semantic Chunks created: {len(chunks)}")
    assert len(chunks) >= 1

    # 3. Document Ingestion
    sample_file = Path(__file__).parent / "sample_knowledge.txt"
    with open(sample_file, "rb") as f:
        bytes_data = f.read()

    doc_id = "test-doc-001"
    user_id = "test-user-001"
    ingest_result = rag.ingest(doc_id, user_id, "sample_knowledge.txt", bytes_data)
    print(f"[PASS] Ingested document: chunks={ingest_result['chunks_count']}, chars={ingest_result['characters']}")
    assert ingest_result["chunks_count"] > 0

    # 4. Dense Retrieval
    async def test_retrieve():
        query = "What is the latency performance of QuantumPulse?"
        rewritten = await rag._rewrite(query)
        print(f"[PASS] Rewritten query: '{rewritten}'")
        candidates = await rag._retrieve_async(rewritten, user_id, doc_id)
        print(f"[PASS] Retrieved {len(candidates)} candidates from Qdrant")
        assert len(candidates) > 0

        # 5. Cross-Encoder Reranking
        ranked = await rag.reranker.rerank_async(rewritten, candidates)
        top_text = ranked[0]["text"]
        print(f"[PASS] Top ranked chunk contains latency: {'120 milliseconds' in top_text}")
        assert "120 milliseconds" in top_text

        # 6. End-to-end Answer with Groq/Grok
        res = await rag.answer(query, user_id, doc_id)
        print(f"[PASS] Answer model: {res['model_used']}")
        print(f"[PASS] Citations count: {len(res['sources'])}")
        print(f"[PASS] Timings (ms): {res['timings_ms']}")
        print(f"[PASS] Answer sample:\n{res['answer'][:180]}...")
        assert res["has_context"] is True
        assert len(res["sources"]) > 0
        assert res["timings_ms"]["pipeline"] > 0.0

        # 7. Multi-part Query Handling
        res2 = await rag.answer("What is the latency performance and which security standards are supported?", user_id, doc_id)
        print(f"\n[PASS] Multi-part query answer:\n{res2['answer'][:180]}...")
        assert res2["timings_ms"]["pipeline"] > 0.0

    asyncio.run(test_retrieve())
    print("\n=== All Production RAG Engine Tests Passed! ===")

if __name__ == "__main__":
    run_tests()
