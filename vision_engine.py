"""Multimodal Vision Engine for Astra RAG AI.

Provides high-speed, production-ready vision capabilities:
- Mathematical equation recognition, domain detection, and step-by-step LaTeX solution
- Plant, animal, flora, fauna, and object identification with confidence assessments
- Screenshot and error diagnosis with stack trace reading and precise resolution steps
- OCR and visual scene description for RAG document ingestion
- Support for Groq (qwen/qwen3.8-27b), xAI Grok Vision, and OpenAI-compatible vision models
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from config import settings

logger = logging.getLogger("astra_vision")


def load_image_as_base64(image_input: str | bytes) -> tuple[str, str]:
    """Loads image from raw bytes, base64 string, data URL, or file path.
    Returns (mime_type, base64_str)."""
    if isinstance(image_input, bytes):
        raw_bytes = image_input
    elif image_input.startswith("data:"):
        # Format: data:image/png;base64,....
        match = re.match(r"data:([^;]+);base64,(.*)", image_input, re.DOTALL)
        if match:
            mime = match.group(1)
            b64 = match.group(2).strip()
            return mime, b64
        raw_bytes = base64.b64decode(image_input.split(",")[-1])
    elif image_input.startswith("/api/images/"):
        filename = Path(image_input).name
        local_path = settings.image_dir / filename
        if not local_path.is_file():
            raise FileNotFoundError(f"Image not found at {local_path}")
        raw_bytes = local_path.read_bytes()
    else:
        p = Path(image_input)
        if p.is_file():
            raw_bytes = p.read_bytes()
        else:
            try:
                raw_bytes = base64.b64decode(image_input)
            except Exception:
                raise ValueError("Unsupported image input format")

    try:
        im = Image.open(io.BytesIO(raw_bytes))
        im_format = (im.format or "JPEG").upper()
        mime = f"image/{im_format.lower()}"
        if mime == "image/jpg":
            mime = "image/jpeg"

        max_dim = 1600
        if max(im.size) > max_dim:
            im.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            out = io.BytesIO()
            save_format = "JPEG" if im_format in ("JPEG", "JPG") else "PNG"
            im.save(out, format=save_format, quality=90)
            raw_bytes = out.getvalue()
            mime = "image/jpeg" if save_format == "JPEG" else "image/png"
    except Exception as e:
        logger.warning("image_preprocess_warning error=%s", e)
        mime = "image/jpeg"

    b64 = base64.b64encode(raw_bytes).decode("utf-8")
    return mime, b64


class MultimodalVisionEngine:
    """Core multimodal vision interface for Astra RAG AI."""

    def __init__(self) -> None:
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(50.0, connect=10.0), verify=False)

    async def close(self) -> None:
        await self.http.aclose()

    def get_vision_endpoint_and_model(self) -> tuple[str, str, str]:
        """Determines active vision endpoint, auth key, and model ID.
        Priority:
        1. Groq qwen/qwen3.8-27b (verified fast & accurate multimodal)
        2. xAI Grok Vision if xai key starts with xai-
        """
        groq_key = settings.groq_api_key or (settings.xai_api_key if settings.xai_api_key.startswith("gsk_") else "")
        xai_key = settings.xai_api_key if settings.xai_api_key.startswith("xai-") else ""

        if groq_key:
            return (
                "https://api.groq.com/openai/v1/chat/completions",
                groq_key,
                settings.vision_model or "qwen/qwen3.8-27b",
            )
        elif xai_key:
            return (
                "https://api.x.ai/v1/chat/completions",
                xai_key,
                "grok-2-vision-1212",
            )
        else:
            return (
                "https://api.groq.com/openai/v1/chat/completions",
                settings.active_llm_key,
                settings.vision_model or "qwen/qwen3.8-27b",
            )

    async def analyze_image(
        self,
        image_input: str | bytes,
        query: str,
        history: list[dict] | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Analyzes an image with the user's prompt using the multimodal vision pipeline."""
        mime, b64 = load_image_as_base64(image_input)
        endpoint, api_key, model = self.get_vision_endpoint_and_model()

        data_url = f"data:{mime};base64,{b64}"

        default_system = (
            "You are Astra AI Multimodal Intelligence — an expert visual scientist, mathematician, and computer vision polymath.\n\n"
            "CORE OPERATIONAL CAPABILITIES:\n"
            "1. MATHEMATICS & SCIENTIFIC EQUATIONS:\n"
            "   - Accurately read all mathematical formulas, symbols, indices, matrices, fractions, and Greek letters from the image.\n"
            "   - Identify the exact branch of mathematics: Algebra, Single/Multivariable Calculus, Differential Equations, Geometry, Trigonometry, Linear Algebra, Statistics, or Classical/Quantum Physics.\n"
            "   - Solve the problem with thorough, step-by-step deductive reasoning.\n"
            "   - Format all mathematical equations in pristine LaTeX: display blocks with $$ ... $$ and inline equations with $ ... $.\n\n"
            "2. PLANT, ANIMAL, AND NATURE IDENTIFICATION:\n"
            "   - Identify species name (common and binomial/scientific nomenclature where possible).\n"
            "   - Detail distinctive botanical/morphological characteristics, native habitat, and care/properties.\n"
            "   - Express honest probabilistic confidence if lighting, angle, or resolution prevents 100% conclusive identification.\n\n"
            "3. CODE, SCREENSHOT & ERROR DIAGNOSTICS:\n"
            "   - Read terminal outputs, IDE stack traces, log files, UI errors, or code snippets with pixel accuracy.\n"
            "   - Explain the exact root cause of the error or bug.\n"
            "   - Provide the complete, working code correction or terminal command to resolve the issue.\n\n"
            "4. OBJECT, PRODUCT, ARCHITECTURE & GENERAL SCENES:\n"
            "   - Describe objects, materials, intended purpose, historical/architectural context, and notable features.\n"
            "   - Directly address the user's specific question about the image with clarity and authoritative insight."
        )

        messages = [
            {"role": "system", "content": system_prompt or default_system}
        ]

        # Add recent conversation context if available
        if history:
            for turn in history[-4:]:
                r = turn.get("role")
                t = turn.get("text") or turn.get("content") or ""
                if r in ("user", "assistant") and t.strip():
                    messages.append({"role": r, "content": t.strip()})

        # User multimodal message
        user_content = [
            {"type": "text", "text": query or "Analyze this image in detail and explain what is shown."},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        messages.append({"role": "user", "content": user_content})

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 2048,
        }

        try:
            res = await self.http.post(endpoint, headers=headers, json=payload, timeout=50.0)
            if res.status_code != 200:
                logger.error("vision_api_error status=%d text=%s", res.status_code, res.text[:300])
                raise RuntimeError(f"Vision API returned error status {res.status_code}: {res.text[:200]}")

            result_json = res.json()
            answer_text = result_json["choices"][0]["message"]["content"].strip()

            return {
                "success": True,
                "answer": answer_text,
                "model": model,
                "image_data": data_url,
            }
        except Exception as e:
            logger.exception("vision_analysis_failed error=%s", e)
            raise

    async def extract_ocr_and_visual_description(self, image_bytes: bytes, filename: str) -> str:
        """Extracts OCR text and rich semantic visual description from an image for RAG vector indexing."""
        prompt = (
            f"Perform a comprehensive document and visual extraction of this uploaded file '{filename}'.\n"
            "1. OCR TEXT: Transcribe ALL readable text, numbers, headings, tables, labels, and code verbatim.\n"
            "2. VISUAL ELEMENTS: Describe any charts, diagrams, graphs (including axes, trends, values), screenshots, illustrations, or key visual objects.\n"
            "3. MATHEMATICS: If equations exist, transcribe them in LaTeX ($$...$$).\n"
            "Output the extracted text and structured descriptions clearly so this file can be indexed into a semantic knowledge base."
        )

        try:
            res = await self.analyze_image(
                image_input=image_bytes,
                query=prompt,
                system_prompt="You are an expert document OCR and visual indexing engine. Transcribe and describe all content faithfully.",
            )
            return res.get("answer", "")
        except Exception as exc:
            logger.warning("ocr_and_visual_extraction_failed file=%s error=%s", filename, exc)
            return f"[Image document: {filename}] (Visual extraction unavailable)"

    async def detect_object_bounding_box(self, image_input: str | bytes, target_object: str) -> dict[str, Any]:
        """Queries the multimodal model to locate target_object and return approximate normalized bounding box."""
        prompt = (
            f"Analyze this image to locate the object described as '{target_object}'.\n"
            "Identify its location in the image. Return a JSON object with the bounding box coordinates normalized from 0 to 1000:\n"
            '{\n  "found": true,\n  "object": "' + target_object + '",\n  "box_2d": [ymin, xmin, ymax, xmax],\n  "description": "Brief note on appearance and location"\n}\n'
            "If the object is not found in the image, return {\"found\": false}.\n"
            "Only return the raw JSON object, without markdown fences or other text."
        )

        try:
            res = await self.analyze_image(
                image_input=image_input,
                query=prompt,
                system_prompt="You are an expert computer vision object detector. Return only valid JSON with normalized [ymin, xmin, ymax, xmax] coordinates from 0 to 1000.",
            )
            raw = res.get("answer", "").strip()
            clean = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
            data = json.loads(clean)
            return data
        except Exception as exc:
            logger.warning("object_detection_failed target=%s error=%s", target_object, exc)
            return {"found": False, "error": str(exc)}


vision_engine = MultimodalVisionEngine()
