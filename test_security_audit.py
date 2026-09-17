import io
import os
import random
import unittest
import uuid
from starlette.testclient import TestClient
from database import get_db, initialize_database, User, Base, engine
from auth import hash_password, verify_password, create_access_token, revoke_user_sessions
from rag_engine import clean_agent_response
import app as app_module

class SecurityAuditTestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        initialize_database()
        cls.client = TestClient(app_module.app, raise_server_exceptions=False)

    def get_valid_challenge(self):
        r = self.client.get("/api/auth/challenge")
        data = r.json()
        token = data["challenge_token"]
        num1 = data["num1"]
        num2 = data["num2"]
        op = data["operation"]
        ans = num1 + num2 if op == "+" else num1 - num2
        return token, ans

    def test_01_password_hashing(self):
        plain = "SecureP@ssw0rd2026!"
        hashed = hash_password(plain)
        self.assertTrue(hashed.startswith("$argon2id$"), "Must use Argon2id")
        self.assertTrue(verify_password(plain, hashed), "Valid password verification failed")
        self.assertFalse(verify_password("WrongP@ssword!", hashed), "Invalid password was accepted")

    def test_02_session_revocation_on_logout(self):
        uid = uuid.uuid4().hex[:8]
        uname = f"revoc_{uid}"
        uemail = f"revoc_{uid}@example.com"
        t_c, ans = self.get_valid_challenge()
        reg_payload = {
            "username": uname,
            "email": uemail,
            "phone": f"+9198{random.randint(10000000, 99999999)}",
            "password": "TestPassword123!",
            "password_confirm": "TestPassword123!",
            "challenge_token": t_c,
            "calculation_result": ans,
            "agreed_to_terms": True,
        }
        r = self.client.post("/api/auth/register", json=reg_payload)
        self.assertIn(r.status_code, [200, 201])
        data = r.json()
        token = data.get("access_token")
        self.assertTrue(token, "Must return token on register")

        headers = {"Authorization": f"Bearer {token}"}
        me_resp = self.client.get("/api/auth/me", headers=headers)
        self.assertEqual(me_resp.status_code, 200)
        self.assertEqual(me_resp.json().get("username"), uname)

        # Logout user
        logout_resp = self.client.post("/api/auth/logout", headers=headers)
        self.assertEqual(logout_resp.status_code, 200)

        # Attempt to reuse old token -> must be 401 Unauthorized
        reused_resp = self.client.get("/api/auth/me", headers=headers)
        self.assertEqual(reused_resp.status_code, 401, "Reused token after logout must return 401")

    def test_03_forgot_password_no_enumeration(self):
        uid = uuid.uuid4().hex[:8]
        uname = f"forgot_{uid}"
        uemail = f"forgot_{uid}@example.com"
        t_c0, ans0 = self.get_valid_challenge()
        self.client.post("/api/auth/register", json={
            "username": uname,
            "email": uemail,
            "password": "TestPassword123!",
            "password_confirm": "TestPassword123!",
            "challenge_token": t_c0,
            "calculation_result": ans0,
            "agreed_to_terms": True,
        })

        t_c1, ans1 = self.get_valid_challenge()
        resp_existing = self.client.post("/api/auth/forgot-password", json={
            "identifier": uemail,
            "challenge_token": t_c1,
            "calculation_result": ans1,
        })
        self.assertEqual(resp_existing.status_code, 200)
        data_existing = resp_existing.json()
        self.assertNotIn("reset_token", data_existing, "Reset token must not be leaked in API response")
        self.assertIn("message", data_existing)

        t_c2, ans2 = self.get_valid_challenge()
        resp_nonexisting = self.client.post("/api/auth/forgot-password", json={
            "identifier": f"ghost_{uuid.uuid4().hex[:8]}@example.com",
            "challenge_token": t_c2,
            "calculation_result": ans2,
        })
        self.assertEqual(resp_nonexisting.status_code, 200)
        data_nonexisting = resp_nonexisting.json()
        self.assertNotIn("reset_token", data_nonexisting)
        self.assertEqual(data_existing["message"], data_nonexisting["message"], "Messages must match to prevent account enumeration")

    def test_04_idor_bola_defense_404(self):
        uid_a = uuid.uuid4().hex[:8]
        uname_a = f"idora_{uid_a}"
        uemail_a = f"idora_{uid_a}@example.com"
        t_ca, ans_a = self.get_valid_challenge()
        self.client.post("/api/auth/register", json={
            "username": uname_a,
            "email": uemail_a,
            "password": "Password123!",
            "password_confirm": "Password123!",
            "challenge_token": t_ca,
            "calculation_result": ans_a,
            "agreed_to_terms": True,
        })
        t_la, ans_la = self.get_valid_challenge()
        login_a = self.client.post("/api/auth/login", json={
            "identifier": uname_a,
            "password": "Password123!",
            "challenge_token": t_la,
            "calculation_result": ans_la,
        })
        token_a = login_a.json()["access_token"]

        uid_b = uuid.uuid4().hex[:8]
        uname_b = f"idorb_{uid_b}"
        uemail_b = f"idorb_{uid_b}@example.com"
        t_cb, ans_b = self.get_valid_challenge()
        self.client.post("/api/auth/register", json={
            "username": uname_b,
            "email": uemail_b,
            "password": "Password123!",
            "password_confirm": "Password123!",
            "challenge_token": t_cb,
            "calculation_result": ans_b,
            "agreed_to_terms": True,
        })
        t_lb, ans_lb = self.get_valid_challenge()
        login_b = self.client.post("/api/auth/login", json={
            "identifier": uname_b,
            "password": "Password123!",
            "challenge_token": t_lb,
            "calculation_result": ans_lb,
        })
        token_b = login_b.json()["access_token"]

        # User A creates a conversation
        conv_resp = self.client.post(
            "/api/conversations",
            headers={"Authorization": f"Bearer {token_a}"},
            json={"title": "User A Private Chat"}
        )
        self.assertEqual(conv_resp.status_code, 200)
        conv_id = conv_resp.json()["id"]

        # User B attempts to access User A conversation via /api/chat
        chat_req = {
            "message": "Hello secret chat",
            "conversation_id": conv_id,
            "stream": False
        }
        cross_resp = self.client.post(
            "/api/chat",
            headers={"Authorization": f"Bearer {token_b}"},
            json=chat_req
        )
        self.assertEqual(cross_resp.status_code, 404, "Must return 404 on unowned resource, not 403")

        # User B attempts to read User A conversation details
        cross_msg_resp = self.client.get(
            f"/api/conversations/{conv_id}",
            headers={"Authorization": f"Bearer {token_b}"}
        )
        self.assertEqual(cross_msg_resp.status_code, 404, "Must return 404 on unowned conversation details, not 403")

    def test_05_file_upload_safety(self):
        uid_u = uuid.uuid4().hex[:8]
        uname_u = f"upload_{uid_u}"
        uemail_u = f"upload_{uid_u}@example.com"
        t_cu, ans_cu = self.get_valid_challenge()
        self.client.post("/api/auth/register", json={
            "username": uname_u,
            "email": uemail_u,
            "password": "Password123!",
            "password_confirm": "Password123!",
            "challenge_token": t_cu,
            "calculation_result": ans_cu,
            "agreed_to_terms": True,
        })
        t_lu, ans_lu = self.get_valid_challenge()
        login_u = self.client.post("/api/auth/login", json={
            "identifier": uname_u,
            "password": "Password123!",
            "challenge_token": t_lu,
            "calculation_result": ans_lu,
        })
        token_u = login_u.json()["access_token"]
        headers = {"Authorization": f"Bearer {token_u}"}

        # 1. MZ Executable disguised as PDF
        fake_pdf = b"MZ\x90\x00\x03\x00\x00\x00This is a Windows executable binary!"
        files = {"file": ("malware.pdf", io.BytesIO(fake_pdf), "application/pdf")}
        resp_mz = self.client.post("/api/upload", headers=headers, files=files)
        self.assertEqual(resp_mz.status_code, 400)
        self.assertIn("Executable binary", resp_mz.json().get("detail", ""))

        # 2. Text file containing binary null bytes
        bad_text = b"Safe text header\x00\x00malicious binary content"
        files = {"file": ("notes.txt", io.BytesIO(bad_text), "text/plain")}
        resp_null = self.client.post("/api/upload", headers=headers, files=files)
        self.assertEqual(resp_null.status_code, 400)
        self.assertIn("binary control characters", resp_null.json().get("detail", "").lower())

        # 3. Legitimate PNG file
        png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        files = {"file": ("pixel.png", io.BytesIO(png_bytes), "image/png")}
        resp_png = self.client.post("/api/upload", headers=headers, files=files)
        self.assertEqual(resp_png.status_code, 200, f"Legitimate PNG upload failed: {resp_png.text}")

    def test_06_path_traversal_image_endpoint(self):
        resp = self.client.get("/api/images/..%2f..%2fconfig.py")
        self.assertIn(resp.status_code, [400, 404])

        resp2 = self.client.get("/api/images/....//....//config.py")
        self.assertIn(resp2.status_code, [400, 404])

    def test_07_excised_endpoints_return_404(self):
        # Admin endpoints
        self.assertEqual(self.client.get("/admin").status_code, 404)
        self.assertEqual(self.client.get("/api/admin/dashboard").status_code, 404)
        self.assertEqual(self.client.get("/api/admin/usage").status_code, 404)
        self.assertEqual(self.client.get("/api/admin/queries").status_code, 404)

        # Delete account endpoints
        self.assertEqual(self.client.get("/delete-account").status_code, 404)
        self.assertEqual(self.client.post("/api/account/delete").status_code, 404)

    def test_08_security_headers_and_csp(self):
        resp = self.client.get("/")
        self.assertEqual(resp.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(resp.headers.get("X-Frame-Options"), "SAMEORIGIN")
        csp = resp.headers.get("Content-Security-Policy", "")
        self.assertIn("connect-src 'self'", csp)
        self.assertIn("frame-ancestors 'self'", csp)

        resp_prev = self.client.get("/preview/test_artifact_123")
        prev_csp = resp_prev.headers.get("Content-Security-Policy", "")
        self.assertIn("sandbox", prev_csp)
        self.assertNotIn("allow-same-origin", prev_csp, "Must NEVER have allow-same-origin in sandboxed preview")

    def test_09_agent_response_secret_scrubbing(self):
        leaky_response = (
            "Here is the secret: gsk_abcdef1234567890abcdef1234567890 "
            "Also my JWT: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c "
            "And reset token: rst_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef "
            "And xai: xai-abcdefghijklmnopqrstuvwxyz012345"
        )
        cleaned = clean_agent_response(leaky_response)
        self.assertNotIn("gsk_", cleaned)
        self.assertNotIn("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", cleaned)
        self.assertNotIn("rst_0123456789abcdef", cleaned)
        self.assertNotIn("xai-abcdef", cleaned)
        self.assertIn("[REDACTED]", cleaned)

    def test_10_sanitized_health_check(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "healthy"})

        resp2 = self.client.get("/health")
        self.assertEqual(resp2.status_code, 200)
        self.assertEqual(resp2.json(), {"status": "ok"})

if __name__ == "__main__":
    unittest.main()