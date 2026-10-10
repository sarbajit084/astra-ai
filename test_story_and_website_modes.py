import unittest
import asyncio
from rag_engine import ProductionRAGService

class TestStoryAndWebsiteModes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag = ProductionRAGService()

    def test_01_creative_detection(self):
        creative_queries = [
            "write me a story",
            "tell me a story about a dragon",
            "write a short bedtime story",
            "write a script",
            "write me a script for a youtube video",
            "write a movie script about two detectives",
            "write a screenplay",
            "write a dialogue between two friends",
            "compose a poem about the ocean",
            "tell a tale of ancient kings",
        ]
        for q in creative_queries:
            self.assertTrue(
                self.rag._is_creative_request(q),
                f"Query '{q}' should be identified as creative request",
            )
            self.assertFalse(
                self.rag._is_coding_request(q),
                f"Query '{q}' must NOT be identified as coding request",
            )
            self.assertFalse(
                self.rag._is_website_request(q),
                f"Query '{q}' must NOT be identified as website request",
            )

    def test_02_code_script_not_creative(self):
        coding_queries = [
            "write a python script to download files",
            "bash script to backup database",
            "shell script for monitoring cpu",
            "write code in python to reverse a string",
            "write a binary search algorithm",
        ]
        for q in coding_queries:
            self.assertFalse(
                self.rag._is_creative_request(q),
                f"Query '{q}' should NOT be identified as creative request",
            )
            self.assertTrue(
                self.rag._is_coding_request(q),
                f"Query '{q}' should be identified as coding request",
            )

    def test_03_website_request_outside_code_mode(self):
        website_queries = [
            "make me a website",
            "create a website for my cafe",
            "build a landing page",
            "generate a portfolio website",
        ]
        for q in website_queries:
            self.assertTrue(
                self.rag._is_website_request(q),
                f"Query '{q}' should be identified as website request",
            )
            # When asked for a website in general or code mode:
            res = asyncio.run(self.rag.answer(q, owner_id="test_user", document_id=None, mode="general"))
            ans = res["answer"]
            # Must NOT contain code blocks
            self.assertNotIn("```html", ans.lower(), f"Website must NOT output ```html for '{q}'")
            self.assertNotIn("```css", ans.lower(), f"Website must NOT output ```css for '{q}'")
            self.assertNotIn("```javascript", ans.lower(), f"Website must NOT output ```javascript for '{q}'")
            # Must clearly state that websites are not generated and small code (Python, C++, C) is supported
            self.assertIn("do not generate websites", ans.lower(), f"Website request must state websites are not generated for '{q}'")
            self.assertIn("python", ans.lower(), f"Website request must mention Python for '{q}'")

    def test_04_story_request_never_outputs_website_or_code(self):
        story_queries = [
            "write me a story",
            "write a story about a brave knight",
            "write a script for a short film",
            "Write me a story of how blender became worl's most consulable 3d rendering software",
        ]
        for q in story_queries:
            # Test in general mode
            res_gen = asyncio.run(self.rag.answer(q, owner_id="test_user", document_id=None, mode="general"))
            ans_gen = res_gen["answer"]
            self.assertNotIn("```html", ans_gen.lower(), f"Story in general mode must NOT output ```html for '{q}'")
            self.assertNotIn("<!-- index.html -->", ans_gen, f"Story in general mode must NOT output html code for '{q}'")
            self.assertNotIn("<canvas id=\"webgl-canvas\">", ans_gen, f"Story must not contain canvas for '{q}'")

            # Test in code mode (even in code mode, a story should not generate a website)
            res_code = asyncio.run(self.rag.answer(q, owner_id="test_user", document_id=None, mode="code"))
            ans_code = res_code["answer"]
            self.assertNotIn("```html", ans_code.lower(), f"Story in code mode must NOT output ```html for '{q}'")
            self.assertNotIn("<!-- index.html -->", ans_code, f"Story in code mode must NOT output html code for '{q}'")

    def test_05_project_continuity_not_hijacked_by_story(self):
        # Simulate previous conversation history containing an active website project
        history = [
            {"role": "user", "content": "build me a supercar website in code mode"},
            {
                "role": "assistant",
                "content": "```html\n<div class='hero'>Supercar</div>\n```\n```css\nbody { background: black; }\n```\n```javascript\nconsole.log('init');\n```"
            }
        ]
        # User now asks for a story
        state = self.rag._detect_project_state(history, "write me a story")
        self.assertFalse(state["is_modification"], "Story query must NOT be considered a project modification")

        res = asyncio.run(self.rag.answer("write me a story", owner_id="test_user", document_id=None, history=history, mode="general"))
        ans = res["answer"]
        self.assertNotIn("```html", ans.lower(), "Follow-up story must NOT generate website code")
        self.assertNotIn("<!-- index.html -->", ans, "Follow-up story must NOT generate website HTML")

if __name__ == "__main__":
    unittest.main()
