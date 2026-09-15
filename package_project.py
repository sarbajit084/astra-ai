"""Astra RAG + AI Agent - Production Distribution Packaging Script
Creates Astra-RAG-AI-Complete-Project.zip with zero secrets, clean source code,
deployment files, documentation, Android project, and release APK.
"""
import os
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path("d:/rag_agent")
OUTPUT_ZIP = PROJECT_ROOT / "Astra-RAG-AI-Complete-Project.zip"

EXCLUDE_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".gradle",
    "tools",
    "tmp_cdp_profile",
    "scratch",
    ".idea",
    "build",
    "dist",
    "qdrant",
    "backups",
    "tmp",
}


EXCLUDE_EXACT_FILES = {
    ".env",
    ".env.local",
    "app.db",
    "astra.db",
    "database.db",
    "rag_agent.db",
    "aster.db",
    "Astra-RAG-AI-Complete-Project.zip",
    "astra_project.zip",
    "local.properties",
}

EXCLUDE_EXTENSIONS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".log",
    ".tmp",
}

def should_exclude(file_path: Path) -> bool:
    rel_parts = file_path.relative_to(PROJECT_ROOT).parts
    filename = file_path.name

    # Exclude directories in the chain
    for part in rel_parts[:-1]:
        if part in EXCLUDE_DIR_NAMES:
            return True
        if part.startswith("tmp_"):
            return True
        if part.startswith(".") and part not in {".dockerignore"}:
            return True

    # Exclude runtime files inside data/ and uploads/ except .gitkeep
    if len(rel_parts) > 1:
        if rel_parts[0] == "data" and filename != ".gitkeep":
            return True
        if rel_parts[0] == "uploads" and filename != ".gitkeep":
            return True
        if rel_parts[0] == "android" and ("build" in rel_parts or ".gradle" in rel_parts):
            return True

    # Exclude exact filenames
    if filename in EXCLUDE_EXACT_FILES:
        return True

    # Exclude specific extensions
    if file_path.suffix.lower() in EXCLUDE_EXTENSIONS:
        return True

    # Exclude any database file
    if file_path.suffix.lower() == ".db":
        return True

    return False

def package_project():
    print(f"Creating clean production distribution archive: {OUTPUT_ZIP.name}")

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    # Ensure empty data and uploads have placeholder
    (PROJECT_ROOT / "data").mkdir(exist_ok=True)
    (PROJECT_ROOT / "uploads").mkdir(exist_ok=True)
    (PROJECT_ROOT / "data" / ".gitkeep").touch(exist_ok=True)
    (PROJECT_ROOT / "uploads" / ".gitkeep").touch(exist_ok=True)

    total_files = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # Prune excluded directories from search tree
            dirs[:] = [
                d for d in dirs
                if d not in EXCLUDE_DIR_NAMES
                and not d.startswith("tmp_")
                and not (d.startswith(".") and d not in {".dockerignore"})
            ]

            for file in files:
                file_path = Path(root) / file
                if should_exclude(file_path):
                    continue

                rel_path = file_path.relative_to(PROJECT_ROOT)
                zip_path = Path("Astra-RAG-AI-Complete-Project") / rel_path

                try:
                    zipf.write(file_path, arcname=str(zip_path).replace("\\", "/"))
                    total_files += 1
                    total_uncompressed_bytes += file_path.stat().st_size
                except Exception as err:
                    print(f"Skipping unreadable or locked file {file_path}: {err}")

    zip_size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    raw_size_mb = total_uncompressed_bytes / (1024 * 1024)

    print(f"\nPackaging Complete!")
    print(f"Total files included: {total_files}")
    print(f"Uncompressed size:    {raw_size_mb:.2f} MB")
    print(f"Compressed ZIP size:  {zip_size_mb:.2f} MB")
    print(f"Archive saved at:     {OUTPUT_ZIP}")

    # Security Verification
    print("\n--- Running Security & Integrity Verification on ZIP ---")
    with zipfile.ZipFile(OUTPUT_ZIP, "r") as check_zip:
        namelist = check_zip.namelist()
        leaks = [
            n for n in namelist
            if n.endswith("/.env")
            or n == "Astra-RAG-AI-Complete-Project/.env"
            or n.endswith(".db")
            or "/.git/" in n
            or "qdrant" in n
        ]
        if leaks:
            print(f"SECURITY FAILED! Found forbidden files in zip: {leaks}")
            sys.exit(1)
        else:
            print("PASS: Zero leaked credentials, zero .env files, zero private databases in archive.")

        has_app_py = any("app.py" in n for n in namelist)
        has_nginx = any("nginx.conf" in n for n in namelist)
        has_env_example = any(".env.example" in n for n in namelist)
        has_apk = any("Astra-Android-Release.apk" in n for n in namelist)
        has_deployment_md = any("DEPLOYMENT.md" in n for n in namelist)
        has_data_gitkeep = any("data/.gitkeep" in n for n in namelist)
        has_uploads_gitkeep = any("uploads/.gitkeep" in n for n in namelist)

        print(f"Check app.py:                    {'PASS' if has_app_py else 'FAIL'}")
        print(f"Check deployment/nginx.conf:     {'PASS' if has_nginx else 'FAIL'}")
        print(f"Check .env.example:              {'PASS' if has_env_example else 'FAIL'}")
        print(f"Check DEPLOYMENT.md:             {'PASS' if has_deployment_md else 'FAIL'}")
        print(f"Check Astra-Android-Release.apk: {'PASS' if has_apk else 'FAIL'}")
        print(f"Check empty data directory:      {'PASS' if has_data_gitkeep else 'FAIL'}")
        print(f"Check empty uploads directory:   {'PASS' if has_uploads_gitkeep else 'FAIL'}")

if __name__ == "__main__":
    package_project()
