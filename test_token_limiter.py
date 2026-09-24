import unittest
import uuid
from datetime import datetime, timezone
from starlette.testclient import TestClient
from database import get_db, initialize_database, User, DailyUsage, SessionLocal
from auth import hash_password, create_access_token
import app as app_module

class TokenLimiterTestSuite(unittest.TestCase):
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

    def test_01_guest_token_status_and_limit_5(self):
        # Create a unique device session for guest
        dev_id = f"dev_test_{uuid.uuid4().hex[:8]}"
        headers = {"X-Device-Id": dev_id}

        # Check initial status: should be 5 tokens
        r_status = self.client.get("/api/tokens/status", headers=headers)
        self.assertEqual(r_status.status_code, 200)
        data = r_status.json()
        self.assertEqual(data["daily_limit"], 5, "Guest daily limit must be 5")
        self.assertEqual(data["is_logged_in"], False)
        self.assertFalse(data["out_of_tokens"])

        # Manually consume 5 tokens in DB for this guest device
        db = SessionLocal()
        try:
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            # Find the guest user
            user = db.query(User).filter(User.email.like(f"%{dev_id}%")).first()
            if not user:
                # Trigger user creation
                self.client.get("/api/tokens/status", headers=headers)
                user = db.query(User).filter(User.email.like(f"%{dev_id}%")).first()

            # Set usage to 5
            ident = f"guest_user:{user.id}"
            rec = db.query(DailyUsage).filter(DailyUsage.identifier == ident, DailyUsage.usage_date == today_str).first()
            if not rec:
                rec = DailyUsage(identifier=ident, usage_date=today_str, tokens_used=5)
                db.add(rec)
            else:
                rec.tokens_used = 5
            db.commit()
        finally:
            db.close()

        # Check status after 5 tokens used: out_of_tokens must be True
        r_after = self.client.get("/api/tokens/status", headers=headers)
        self.assertEqual(r_after.status_code, 200)
        data_after = r_after.json()
        self.assertEqual(data_after["tokens_remaining"], 0)
        self.assertTrue(data_after["out_of_tokens"])

        # Attempt to chat when out of tokens: must return HTTP 429 with 'Out of tokens'
        chat_req = {
            "message": "Testing guest limit exhaustion",
            "stream": False
        }
        r_chat = self.client.post("/api/chat", headers=headers, json=chat_req)
        self.assertEqual(r_chat.status_code, 429)
        self.assertIn("Out of tokens", r_chat.json().get("detail", ""))

    def test_02_registered_user_limit_20(self):
        uid = uuid.uuid4().hex[:8]
        uname = f"tokuser_{uid}"
        uemail = f"tokuser_{uid}@example.com"
        t_c, ans = self.get_valid_challenge()
        r_reg = self.client.post("/api/auth/register", json={
            "username": uname,
            "email": uemail,
            "password": "Password123!",
            "password_confirm": "Password123!",
            "challenge_token": t_c,
            "calculation_result": ans,
            "agreed_to_terms": True,
        })
        self.assertIn(r_reg.status_code, [200, 201])
        token = r_reg.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Check initial status for registered user: daily_limit must be 20
        r_status = self.client.get("/api/tokens/status", headers=headers)
        self.assertEqual(r_status.status_code, 200)
        data = r_status.json()
        self.assertEqual(data["daily_limit"], 20, "Registered user daily limit must be 20")
        self.assertEqual(data["is_logged_in"], True)
        self.assertEqual(data["tokens_remaining"], 20)
        self.assertFalse(data["out_of_tokens"])

        # Check /api/auth/me also contains tokens info
        r_me = self.client.get("/api/auth/me", headers=headers)
        self.assertEqual(r_me.status_code, 200)
        self.assertIn("tokens", r_me.json())
        self.assertEqual(r_me.json()["tokens"]["daily_limit"], 20)

        # Set usage to 20 in DB
        db = SessionLocal()
        try:
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            user = db.query(User).filter(User.username == uname).first()
            ident = f"user:{user.id}"
            rec = db.query(DailyUsage).filter(DailyUsage.identifier == ident, DailyUsage.usage_date == today_str).first()
            if not rec:
                rec = DailyUsage(identifier=ident, usage_date=today_str, tokens_used=20)
                db.add(rec)
            else:
                rec.tokens_used = 20
            db.commit()
        finally:
            db.close()

        # Check status after 20 used
        r_exhausted = self.client.get("/api/tokens/status", headers=headers)
        self.assertEqual(r_exhausted.status_code, 200)
        self.assertEqual(r_exhausted.json()["tokens_remaining"], 0)
        self.assertTrue(r_exhausted.json()["out_of_tokens"])

        # Attempt to chat when out of tokens: must return HTTP 429
        chat_req = {
            "message": "Testing registered limit exhaustion",
            "stream": False
        }
        r_chat = self.client.post("/api/chat", headers=headers, json=chat_req)
        self.assertEqual(r_chat.status_code, 429)
        self.assertIn("Out of tokens", r_chat.json().get("detail", ""))

if __name__ == "__main__":
    unittest.main()

