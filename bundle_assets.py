import os
import shutil
import sys
from pathlib import Path

def bundle_assets(api_base_url="https://YOUR-DOMAIN.com"):
    src_static = Path("d:/rag_agent/static")
    dst_www = Path("d:/rag_agent/android/app/src/main/assets/www")

    print(f"Bundling assets to {dst_www} with API base: {api_base_url}")

    # 1. Clean destination directory
    if dst_www.exists():
        shutil.rmtree(dst_www)
    dst_www.mkdir(parents=True, exist_ok=True)

    # 2. Copy clean static folder
    dst_static = dst_www / "static"
    shutil.copytree(src_static, dst_static, dirs_exist_ok=True)

    # 3. Create config.js for configurable production endpoint
    config_js = f"""// Astra Android Mobile Client - Production Backend Endpoint
// Automatically configured for GoDaddy / Remote Cloud Production Deployment.
// Update this value or set 'astra_api_base_url' in localStorage.
window.ASTRA_API_BASE_URL = localStorage.getItem('astra_api_base_url') || '{api_base_url.rstrip("/")}';
"""
    (dst_www / "config.js").write_text(config_js, encoding="utf-8")

    # 4. Prepare index.html with relative paths and config.js included
    index_path = src_static / "index.html"
    index_content = index_path.read_text(encoding="utf-8")

    # Convert absolute paths to relative
    index_relative = (
        index_content
        .replace('href="/static/', 'href="static/')
        .replace('src="/static/', 'src="static/')
    )

    # Inject config.js script before app.js
    script_injection = '<script src="config.js"></script>\n    <script src="static/app.js'
    index_relative = index_relative.replace('<script src="static/app.js', script_injection)

    (dst_www / "index.html").write_text(index_relative, encoding="utf-8")

    print("SUCCESS: Frontend assets bundled into Android assets directory.")

if __name__ == "__main__":
    target_url = sys.argv[1] if len(sys.argv) > 1 else "https://YOUR-DOMAIN.com"
    bundle_assets(target_url)
