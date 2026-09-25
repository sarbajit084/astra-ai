from __future__ import annotations

import base64
import logging
import os
from typing import Any, Optional
import httpx

from config import settings

logger = logging.getLogger("astra_github_service")

# Disallowed file extensions and paths for security
RESTRICTED_EXTENSIONS = frozenset({
    ".env", ".key", ".pem", ".cert", ".crt", ".pfx", ".p12", ".db", ".sqlite", ".dat"
})
RESTRICTED_FILES = frozenset({
    ".env", ".env.local", ".env.production", ".gitignore", ".gitmodules", "id_rsa", "id_ed25519"
})


class GitHubService:
    """Secure server-side proxy service for private GitHub repository interaction.
    
    All GitHub credentials and repository coordinates are kept strictly on the backend.
    Raw GitHub API responses, headers, commit SHAs, clone URLs, and error traces
    are never leaked to clients.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        owner: Optional[str] = None,
        repository: Optional[str] = None,
        branch: Optional[str] = None,
    ) -> None:
        self.token = (token or getattr(settings, "github_token", "") or "").strip()
        self.owner = (owner or getattr(settings, "github_owner", "") or "").strip()
        self.repository = (repository or getattr(settings, "github_repository", "") or "").strip()
        self.branch = (branch or getattr(settings, "github_branch", "main") or "main").strip()

    @property
    def is_configured(self) -> bool:
        """Returns True if the required GitHub credentials and repository info are configured."""
        return bool(self.token and self.owner and self.repository)

    @staticmethod
    def sanitize_path(raw_path: str) -> str:
        """Sanitizes and validates a requested repository file path.
        Prevents directory traversal, absolute paths, and access to sensitive files.
        """
        if not raw_path or not isinstance(raw_path, str):
            raise ValueError("Invalid file path.")

        # Normalize slashes and strip whitespace
        normalized = raw_path.strip().replace("\\", "/")

        # Remove leading slashes
        while normalized.startswith("/"):
            normalized = normalized[1:]

        # Reject path traversal attempts
        parts = normalized.split("/")
        if any(part in ("..", ".", "") for part in parts):
            raise ValueError("Path traversal sequences are strictly prohibited.")

        filename = parts[-1].lower()

        # Reject restricted files and hidden sensitive directories
        if filename in RESTRICTED_FILES or any(p.startswith(".git") for p in parts):
            raise ValueError("Access to the requested file or directory is restricted.")

        # Check restricted extensions
        _, ext = os.path.splitext(filename)
        if ext in RESTRICTED_EXTENSIONS:
            raise ValueError("Access to this file type is restricted.")

        return normalized

    def fetch_file(self, raw_path: str) -> dict[str, Any]:
        """Fetches a file's text content from the private repository.
        Returns only sanitized content and size. Never leaks tokens or repository URLs.
        """
        if not self.is_configured:
            raise RuntimeError("Repository integration is not configured on this server.")

        clean_path = self.sanitize_path(raw_path)
        api_url = f"https://api.github.com/repos/{self.owner}/{self.repository}/contents/{clean_path}"

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Astra-Agent/3.0",
        }
        params = {"ref": self.branch}

        try:
            with httpx.Client(timeout=15.0) as client:
                response = client.get(api_url, headers=headers, params=params)
        except Exception:
            logger.error("github_network_error path=%s", clean_path)
            raise RuntimeError("Failed to communicate with repository service.") from None

        if response.status_code == 404:
            raise FileNotFoundError("Requested resource was not found in repository.")
        elif response.status_code in (401, 403):
            logger.error("github_auth_error status=%d", response.status_code)
            raise PermissionError("Repository access authentication failed.")
        elif response.status_code != 200:
            logger.error("github_api_error status=%d", response.status_code)
            raise RuntimeError("Repository service returned an unexpected response.")

        try:
            data = response.json()
        except Exception:
            raise RuntimeError("Invalid response received from repository service.")

        # If path points to a directory rather than a file
        if isinstance(data, list):
            # Return list of filenames only, no URLs or internal metadata
            entries = []
            for item in data:
                if isinstance(item, dict) and "name" in item and "type" in item:
                    entries.append({"name": item["name"], "type": item["type"]})
            return {
                "path": clean_path,
                "type": "directory",
                "entries": entries,
            }

        if data.get("type") != "file":
            raise ValueError("Requested resource is not a regular file.")

        content_encoded = data.get("content", "")
        encoding = data.get("encoding", "")
        size = data.get("size", 0)

        if encoding == "base64":
            try:
                decoded_bytes = base64.b64decode(content_encoded)
                text_content = decoded_bytes.decode("utf-8")
            except UnicodeDecodeError:
                raise ValueError("File content is binary and cannot be displayed as text.")
            except Exception:
                raise RuntimeError("Failed to decode file content.")
        else:
            text_content = content_encoded

        # Return strictly minimal, sanitized payload
        return {
            "path": clean_path,
            "type": "file",
            "size": size,
            "content": text_content,
        }


github_service = GitHubService()
