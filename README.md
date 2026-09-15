# Astra RAG + AI Agent

An enterprise-grade, full-stack **Retrieval-Augmented Generation (RAG)** platform and autonomous **AI Agent** featuring multi-modal document ingestion, hybrid semantic vector search, cryptographic calculation challenge security, dual Argon2id password authentication, per-device unauthenticated chat isolation, and a native Android mobile application.

---

## 🌟 Key Features

- **Autonomous AI Agent & Reasoning**: Multi-turn grounded responses powered by Groq (`openai/gpt-oss-120b`) or xAI Grok (`grok-2-latest`) with real-time streaming (SSE).
- **Multi-Format Document Ingestion**: Upload and index PDF, Word (`.docx`), plain text, markdown, CSV, JSON, and `.zip` archives up to **400 MB per file** with built-in path traversal and zip-bomb defenses.
- **Hybrid Semantic Retrieval**: Qdrant vector store integration, BAAI embeddings (`BAAI/bge-small-en-v1.5`), and cross-encoder reranking (`BAAI/bge-reranker-base`).
- **Security & Dual Authentication**:
  - **Dynamic Calculation Verification**: Zero-knowledge HMAC math challenge (anti-bot & anti-replay protection).
  - **Dual Password Hashing**: Argon2id (`time_cost=3, memory_cost=65536, parallelism=4`) with automatic legacy Bcrypt migration.
  - **Account Lockout & Rate Limiting**: 5 consecutive failed attempts trigger a 10-minute temporary lockout.
  - **Strict User & Device Isolation**: Zero IDOR vulnerabilities; unauthenticated guest sessions are bound cryptographically to physical device origin tokens (`X-Guest-Token`, `X-Device-Id`) ensuring zero cross-device chat leaks even on shared Wi-Fi networks.
- **Code Generation & Export**: Fenced syntax highlighting, one-click pure code copy, and direct language-specific file downloads (`.py`, `.html`, `.css`, `.js`, `.json`, etc.).
- **Interactive Visualizations**: Inline rendering of Mermaid diagrams, KaTeX mathematical equations, Chart.js interactive charts, and Three.js 3D canvas animations.
- **Native Android App**: Hardware-accelerated WebView container with document picker, custom dark theme status bar, offline reconnection handler, and configurable remote API endpoints.

---

## 🏗️ Technology Stack

- **Frontend**: Vanilla ES6+ HTML5/CSS3/JavaScript, Three.js, Mermaid.js, KaTeX, Highlight.js, Chart.js.
- **Backend API**: FastAPI (Python 3.10+), Uvicorn, Gunicorn (high-concurrency ASGI workers).
- **Database**: SQLAlchemy 2.0 ORM with SQLite (default) and PostgreSQL support (`psycopg3`).
- **Vector Database**: Qdrant (`qdrant-client`).
- **Deployment**: Nginx Reverse Proxy, Systemd service unit, automated bash scripts (`setup.sh`, `deploy.sh`).
- **Mobile Client**: Native Android application (Java, Android SDK 34, minSdk 24).

---

## 🚀 Quick Start (Local Development)

### 1. Clone & Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/your-username/astra-ai.git
cd astra-ai

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
cp .env.example .env
```
Edit `.env` and configure your API key:
- `GROQ_API_KEY`: Your Groq Cloud API Key (`gsk_...` from [console.groq.com](https://console.groq.com))
- `JWT_SECRET`: Any random secure string for local development

### 4. Run the Application
```bash
python app.py
```
Or run with Uvicorn directly:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser at [http://127.0.0.1:8000](http://127.0.0.1:8000). The first registered user automatically receives administrative privileges.

---

## 🌐 Production Deployment (GoDaddy Server)

Astra includes a production deployment package ready for **GoDaddy Linux VPS / Dedicated Servers** (Ubuntu/Debian) or **cPanel Web Hosting**.

### Automated 1-Command Setup on GoDaddy VPS
1. Upload the project zip archive to your server.
2. Unzip into `/opt/astra`.
3. Run the automated installer:
   ```bash
   chmod +x deployment/setup.sh deployment/deploy.sh
   sudo ./deployment/setup.sh
   ```
4. Enter your API key in `/opt/astra/.env` and restart:
   ```bash
   sudo systemctl restart astra
   ```
5. Configure free SSL with Certbot:
   ```bash
   sudo certbot --nginx -d your-domain.com -d www.your-domain.com
   ```

For detailed GoDaddy deployment instructions (including DNS setup, cPanel Python App alternative, and zero-downtime updates), see [DEPLOYMENT.md](DEPLOYMENT.md).

---

## 📱 Android Mobile Application

The project includes an Android application version configured for production HTTPS connectivity:

- **Pre-built APK**: `Astra-Android-Release.apk` (in the project root and `dist/`).
- **Source Code**: Located in the [`android/`](android/) directory.
- **Configuring Server Domain**:
  ```bash
  python bundle_assets.py https://your-godaddy-domain.com
  python build_astra_apk.py
  ```

For full Android build and installation instructions, see [android/README.md](android/README.md).

---

## 🔒 Security & Privacy Notice

- All passwords are encrypted with **Argon2id** password hashing.
- File uploads are verified for MIME types, maximum size (400 MB), path traversal (`../`), and compression expansion limits.
- Sensitive files (`.env`, private databases, logs, and build artifacts) are excluded via `.gitignore` and distribution packaging.
- Always keep your `.env` private and never share API keys publicly.

---

## 📄 License
This project is licensed under the MIT License.
