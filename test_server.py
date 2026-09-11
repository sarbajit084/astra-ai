import sys
import uuid
from starlette.testclient import TestClient

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from app import app

def test_api():
    print("=== Testing FastAPI Endpoints ===")
    with TestClient(app) as client:
        # 1. Health check
        res = client.get("/api/health")
        print(f"[PASS] Health check: {res.status_code}, provider: {res.json().get('provider')}")
        assert res.status_code == 200

        # 2. Prometheus Metrics
        res = client.get("/metrics")
        print(f"[PASS] Metrics endpoint: {res.status_code}, length: {len(res.text)}")
        assert res.status_code == 200
        assert "aster_query_requests_total" in res.text

        # 3. Calligraphy Welcome / Landing HTML
        res = client.get("/")
        print(f"[PASS] Root HTML: {res.status_code}")
        assert res.status_code == 200
        assert "Aster" in res.text
        assert "calligraphy" in res.text

        # 4. Authentication (Signup admin)
        test_email = f"test_{uuid.uuid4().hex[:6]}@example.com"
        res = client.post("/api/auth/signup", json={"email": test_email, "password": "SecurePassword123!"})
        print(f"[PASS] Signup status: {res.status_code}")
        assert res.status_code == 201
        auth_data = res.json()
        token = auth_data["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 5. Documents list
        res = client.get("/api/documents", headers=headers)
        print(f"[PASS] Documents list status: {res.status_code}, count: {res.json()['total_documents']}")
        assert res.status_code == 200

        # 6. Upload sample document
        sample_content = b"Aster Enterprise RAG Architecture supports 200k simultaneous users with sub-second p99 latency."
        res = client.post(
            "/api/upload",
            headers=headers,
            files={"file": ("architecture_spec.txt", sample_content, "text/plain")},
        )
        print(f"[PASS] Upload document status: {res.status_code}")
        assert res.status_code == 200
        uploaded_doc = res.json()["document"]
        assert uploaded_doc["status"] == "ready"

        # 7. Chat Query with grounded response
        res = client.post(
            "/api/chat",
            headers=headers,
            json={"message": "How many simultaneous users are supported by Aster?"},
        )
        print(f"[PASS] Chat status: {res.status_code}")
        assert res.status_code == 200
        chat_data = res.json()
        print(f"[PASS] Latency (ms): {chat_data['latency_ms']}")
        assert chat_data["latency_ms"] > 0.0, "Latency must never be 0.00s!"
        assert "answer" in chat_data
        assert len(chat_data["sources"]) > 0
        print(f"[PASS] Answer snippet: {chat_data['answer'][:120]}...")

        # 8. Admin Dashboard (Promote user to admin for test)
        from database import SessionLocal, User
        with SessionLocal() as db:
            u = db.get(User, auth_data["user"]["id"])
            if u:
                u.role = "admin"
                db.commit()
        from auth import create_access_token
        admin_token = create_access_token(auth_data["user"]["id"], "admin")
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        res = client.get("/api/admin/dashboard", headers=admin_headers)
        print(f"[PASS] Admin dashboard status: {res.status_code}")
        assert res.status_code == 200
        dash_data = res.json()
        assert dash_data["totals"]["users"] >= 1
        assert dash_data["totals"]["queries"] >= 1
        assert dash_data["totals"]["average_latency_ms"] > 0.0

        # 9. Admin Usage Patterns
        res = client.get("/api/admin/usage", headers=admin_headers)
        print(f"[PASS] Admin usage patterns status: {res.status_code}")
        assert res.status_code == 200
        usage_data = res.json()
        assert "daily_queries" in usage_data
        assert "percentiles" in usage_data

        print("\n=== All FastAPI Integration Tests Passed Successfully! ===")

if __name__ == "__main__":
    test_api()
