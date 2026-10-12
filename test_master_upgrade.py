import asyncio
import os
import re
import time
import unittest
import uuid
from pathlib import Path
from fastapi.testclient import TestClient

from app import app, rag as rag_service, rate_limiter
from config import settings
from database import initialize_database


class TestMasterUpgrade(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        initialize_database()
        cls.client = TestClient(app)

    def setUp(self):
        rate_limiter.requests.clear()

    def get_valid_challenge(self):
        r = self.client.get("/api/auth/challenge")
        data = r.json()
        token = data["challenge_token"]
        num1 = data["num1"]
        num2 = data["num2"]
        op = data["operation"]
        ans = num1 + num2 if op == "+" else num1 - num2
        return token, ans

    def test_01_persistent_authentication_and_cookie_security(self):
        """Requirement 1: 1-year persistent HttpOnly session cookie, session restoration without localStorage, and server-side revocation on logout."""
        self.assertGreaterEqual(
            settings.jwt_access_token_ttl_seconds,
            31536000,
            "Session TTL must be at least 1 year (31536000 seconds).",
        )

        uid = uuid.uuid4().hex[:8]
        unique_username = f"persist_{uid}"
        unique_email = f"persist_{uid}@example.com"
        password = "StrongPassword123!"
        t_c, ans = self.get_valid_challenge()

        # 1. Register a new user
        reg_res = self.client.post(
            "/api/auth/register",
            json={
                "username": unique_username,
                "email": unique_email,
                "password": password,
                "password_confirm": password,
                "challenge_token": t_c,
                "calculation_result": ans,
                "agreed_to_terms": True,
            },
        )
        self.assertIn(reg_res.status_code, (200, 201), reg_res.text)
        set_cookie_header = reg_res.headers.get("set-cookie", "")
        self.assertIn("astra_session=", set_cookie_header)
        self.assertIn("HttpOnly", set_cookie_header)
        self.assertIn("Max-Age=31536000", set_cookie_header)
        self.assertIn("expires=", set_cookie_header.lower())

        user_id = reg_res.json()["user"]["id"]

        # 2. Simulate browser restart / page refresh: call /api/auth/me using ONLY cookies (no Authorization header)
        me_res = self.client.get("/api/auth/me")
        self.assertEqual(me_res.status_code, 200)
        me_data = me_res.json()
        self.assertEqual(me_data["id"], user_id)
        self.assertEqual(me_data["email"], unique_email)
        self.assertEqual(me_data["role"], "user")

        # 3. Verify protected routes work across multiple refreshes via cookie alone
        for _ in range(3):
            conv_res = self.client.get("/api/conversations")
            self.assertEqual(conv_res.status_code, 200)

        # 4. Save cookie value to test replay after logout
        saved_session_cookie = self.client.cookies.get("astra_session")
        self.assertIsNotNone(saved_session_cookie)

        # 5. Explicit logout must invalidate token_version in DB and clear cookie
        logout_res = self.client.post("/api/auth/logout")
        self.assertEqual(logout_res.status_code, 200)

        # After logout, /api/auth/me should return 401
        me_after_logout = self.client.get("/api/auth/me")
        self.assertEqual(me_after_logout.status_code, 401)

        # Even if old cookie is replayed, server-side token_version bump rejects it with 401
        self.client.cookies.set("astra_session", saved_session_cookie)
        me_replayed = self.client.get("/api/auth/me")
        self.assertEqual(me_replayed.status_code, 401)
        self.client.cookies.clear()

        # 6. Verify static/app.js never writes tokens to localStorage
        app_js = Path("static/app.js").read_text(encoding="utf-8")
        self.assertNotIn("localStorage.setItem('aster_token'", app_js)
        self.assertNotIn('localStorage.setItem("aster_token"', app_js)
        self.assertNotIn("localStorage.setItem('astra_guest_token'", app_js)
        self.assertIn("restoreSessionOnStartup", app_js)

    def test_02_website_generation_completely_disabled(self):
        """Requirement 2a: Website generation endpoints removed and website prompts blocked in all modes."""
        # Preview routes must be 404
        res_post = self.client.post("/api/preview", json={"html": "<h1>Test</h1>"})
        self.assertEqual(res_post.status_code, 404)
        res_get = self.client.get("/preview/12345678")
        self.assertEqual(res_get.status_code, 404)

        # Live Demo button removed from frontend
        app_js = Path("static/app.js").read_text(encoding="utf-8")
        self.assertNotIn("live-demo-btn", app_js)
        self.assertNotIn("/api/preview", app_js)

        website_prompts = [
            "Create a complete website for a coffee shop with HTML, CSS, and JS",
            "Build me a responsive landing page for my SaaS startup",
            "Generate a full web application dashboard in React",
            "Design a website template for an e-commerce store",
        ]
        for prompt in website_prompts:
            for mode in ("general", "code", "detailed"):
                res = asyncio.run(rag_service.answer(prompt, "test_user", None, mode=mode))
                ans = res["answer"]
                self.assertIn("disabled", ans.lower(), f"Failed to block website prompt '{prompt}' in mode '{mode}': {ans}")
                self.assertNotIn("<!doctype html>", ans.lower())

    def test_03_code_mode_restrictions_and_intent_classification(self):
        """Requirement 2b: Normal mode returns exact string on coding requests, allows conceptual questions, and Code Mode allows programming."""
        coding_prompts = [
            "Write a Python function to reverse a string",
            "Write a C++ program for binary search",
            "Give me a SQL query to find duplicate emails in a users table",
            "Implement bubble sort in Java",
        ]
        for prompt in coding_prompts:
            res = asyncio.run(rag_service.answer(prompt, "test_user", None, mode="general"))
            ans = res["answer"]
            self.assertEqual(
                ans,
                "Code can only be generated in Code Mode.",
                f"Expected exact Code Mode message for '{prompt}', got: {ans}",
            )

        # Ordinary conceptual questions mentioning 'website', 'Python', 'Java', or 'code' must NOT be blocked
        ordinary_questions = [
            "What is Python mainly used for in data science?",
            "How does a website work when you type a URL into a browser?",
            "What is the difference between Java and C++ conceptually?",
            "What is Morse code and who invented it?",
        ]
        for q in ordinary_questions:
            self.assertFalse(rag_service._is_website_request(q), f"False positive website filter on: {q}")
            self.assertFalse(rag_service._is_coding_request(q), f"False positive coding filter on: {q}")

        # In Code Mode (mode="code"), programming assistance works
        code_res = asyncio.run(
            rag_service.answer("Write a short Python function to check if a number is prime.", "test_user", None, mode="code")
        )
        code_ans = code_res["answer"]
        self.assertNotEqual(code_ans, "Code can only be generated in Code Mode.")
        self.assertTrue("def " in code_ans or "return" in code_ans, f"Expected Python function in Code Mode, got: {code_ans[:200]}")

    def test_04_font_changed_to_anthropic_serif(self):
        """Requirement 3: anthropicSerif configured across styling architecture while keeping monospace for code."""
        style_css = Path("static/style.css").read_text(encoding="utf-8")
        self.assertIn("font-family: 'anthropicSerif'", style_css)
        self.assertIn("--font-serif: 'anthropicSerif'", style_css)
        self.assertIn("--font-mono: 'JetBrains Mono'", style_css)

    def test_05_privacy_section_removed_and_settings_contrast_fixed(self):
        """Requirement 4: Privacy & Security Policy section removed from Settings modal, legal routes preserved, and contrast fixed."""
        index_html = Path("static/index.html").read_text(encoding="utf-8")
        self.assertNotIn("Privacy & Security Policies", index_html)
        self.assertNotIn("privacy-policy-list", index_html)
        self.assertIn('id="popoverModeCode"', index_html)

        # Legal routes must still work
        self.assertEqual(self.client.get("/privacy").status_code, 200)
        self.assertEqual(self.client.get("/terms").status_code, 200)

        # Settings contrast rules in style.css for both light and dark modes
        style_css = Path("static/style.css").read_text(encoding="utf-8")
        self.assertIn('[data-bg-theme="dark"] .settings-modal-card', style_css)
        self.assertIn(".settings-btn:focus-visible", style_css)

    def test_06_default_mode_speed_and_intelligence(self):
        """Requirements 5 & 6: Fast heuristic follow-up resolution, live query classification, and <4s default-mode latency."""
        # 1. Heuristic follow-up resolution under 5ms
        t0 = time.perf_counter()
        resolved = rag_service._heuristic_resolve(
            "What are its main applications?",
            [{"role": "user", "content": "Tell me about photosynthesis"}],
        )
        rewrite_ms = (time.perf_counter() - t0) * 1000
        self.assertLess(rewrite_ms, 5.0)
        self.assertIn("photosynthesis", resolved.lower())

        # 2. Live query detection
        self.assertTrue(rag_service._is_live_external_query("What is the weather in London today?"))
        self.assertTrue(rag_service._is_live_external_query("What happened in the news yesterday?"))
        self.assertFalse(rag_service._is_live_external_query("Explain how photosynthesis works."))
        self.assertFalse(rag_service._is_live_external_query("What is the capital of Japan?"))

        # 3. Default-mode response speed benchmark (< 4.0 seconds)
        queries = [
            "Explain what photosynthesis is in 2 sentences.",
            "What is the capital of Japan and one famous landmark there?",
        ]
        latencies = []
        for q in queries:
            start = time.perf_counter()
            res = asyncio.run(rag_service.answer(q, "test_user", None, mode="general"))
            ans = res["answer"]
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)
            self.assertTrue(len(ans) > 20, f"Empty/too short response for '{q}': {ans}")
            self.assertLess(elapsed_ms, 4000.0, f"Query '{q}' took {elapsed_ms:.0f}ms (> 4000ms target)")

        avg_ms = sum(latencies) / len(latencies)
        print(f"\n[BENCHMARK] Default-mode latencies: {[round(x, 1) for x in latencies]} ms (Avg: {avg_ms:.1f} ms)")
        self.assertLess(avg_ms, 4000.0)


if __name__ == "__main__":
    unittest.main()
