# Astra RAG + AI Agent - Production Deployment Guide for GoDaddy

This guide provides step-by-step instructions for deploying the **Astra RAG + AI Agent** application on **GoDaddy** servers. It covers both **GoDaddy Linux VPS / Dedicated Servers** (recommended for production) and **GoDaddy cPanel Shared / Business Hosting** with the Python Application selector.

---

## 1. Architecture Overview

| Component | Technology | Role |
|---|---|---|
| **Frontend** | Vanilla ES6+ HTML5/CSS3/JS, Three.js, Mermaid, KaTeX | Responsive SPA served directly or via Nginx |
| **Backend** | FastAPI (Python 3.10+), Uvicorn & Gunicorn | High-concurrency async REST & streaming API |
| **Vector DB** | Qdrant (`qdrant-client`) | Semantic vector search & document retrieval |
| **Database** | SQLAlchemy 2.0 (SQLite or PostgreSQL) | User credentials, sessions, history, metadata |
| **Web Server** | Nginx Reverse Proxy | SSL/HTTPS termination, 400MB uploads, SSE streaming |
| **Process Daemon**| Systemd (`astra.service`) | Background process management & auto-restart |

---

## 2. Server Requirements & Prerequisites

### Hardware Recommendations
- **Minimum**: 1 vCPU, 2 GB RAM, 20 GB SSD (handles embedded Qdrant + SQLite + Groq/Grok API).
- **Recommended**: 2-4 vCPUs, 4-8 GB RAM, 50 GB SSD (optimal for local neural models, large document indexing, and concurrent users).

### Operating System
- **Ubuntu 20.04 / 22.04 / 24.04 LTS** or **Debian 11 / 12** on GoDaddy VPS / Dedicated Server.
- Root or `sudo` access via SSH.

### External API Requirements
- At least one LLM API key:
  - **Groq Cloud API Key** (`gsk_...` from [console.groq.com](https://console.groq.com)) - Free, ultra-fast.
  - **xAI Grok API Key** (`xai-...` from [console.x.ai](https://console.x.ai)) - High intelligence.

---

## 3. Domain & DNS Configuration on GoDaddy

1. Log in to your **GoDaddy Account** and navigate to **My Products** > **DNS Management** for your domain.
2. Under **DNS Records**, add or update the **A Record**:
   - **Type**: `A`
   - **Name**: `@` (or subdomain such as `astra` for `astra.yourdomain.com`)
   - **Value**: Your GoDaddy Server Public IP Address (e.g. `123.45.67.89`)
   - **TTL**: `1/2 Hour` or `Default`
3. If using `www`, add a CNAME record:
   - **Type**: `CNAME`
   - **Name**: `www`
   - **Value**: `@`
4. Wait 5–15 minutes for global DNS propagation. You can verify propagation by running:
   ```bash
   ping your-domain.com
   ```

---

## 4. Method A: Automated Deployment on GoDaddy VPS (Recommended)

### Step 1: Upload the Project Package
From your local computer, copy `Astra-RAG-AI-Complete-Project.zip` to your GoDaddy server via SCP or SFTP:
```bash
scp Astra-RAG-AI-Complete-Project.zip root@YOUR_SERVER_IP:/root/
```

### Step 2: SSH into Your Server
```bash
ssh root@YOUR_SERVER_IP
```

### Step 3: Extract the Archive
```bash
cd /root
unzip Astra-RAG-AI-Complete-Project.zip -d /opt/astra
cd /opt/astra
```

### Step 4: Run the Automated Setup Script
The included setup script installs all Linux dependencies, sets up the virtual environment, configures Nginx, and starts the systemd service:
```bash
chmod +x deployment/setup.sh deployment/deploy.sh
sudo ./deployment/setup.sh
```

### Step 5: Configure Your API Keys in `.env`
Open the generated `.env` configuration file:
```bash
nano /opt/astra/.env
```
Update the following critical values:
- `GROQ_API_KEY`: Your Groq API key (`gsk_...`)
- `GROK_API_KEY`: Your xAI Grok API key (optional if using Groq)
- `CORS_ORIGINS`: Set to `https://your-domain.com,https://www.your-domain.com`

Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X`).

### Step 6: Restart the Astra Service
```bash
systemctl restart astra
```

### Step 7: Enable Free SSL/HTTPS via Certbot
Run Certbot to automatically configure Let's Encrypt HTTPS certificates for Nginx:
```bash
certbot --nginx -d your-domain.com -d www.your-domain.com
```
Follow the interactive prompts (enter your admin email and agree to terms). Certbot will automatically install the certificate and configure HTTPS redirection in Nginx!

### Step 8: Verify Application Status
Open your browser and visit `https://your-domain.com`. You will see the Astra interface live and connected!

---

## 5. Method B: Manual Deployment Step-by-Step

If you prefer to configure the server manually without running `setup.sh`:

### 1. Install System Packages
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv python3-dev build-essential nginx curl git certbot python3-certbot-nginx
```

### 2. Create Service User & Directories
```bash
sudo useradd -r -s /bin/false -d /opt/astra astra
sudo mkdir -p /opt/astra /opt/astra/data /opt/astra/uploads /var/log/astra
sudo chown -R astra:www-data /opt/astra /var/log/astra
```

### 3. Setup Python Virtual Environment
```bash
cd /opt/astra
sudo -u astra python3 -m venv venv
sudo -u astra ./venv/bin/pip install --upgrade pip setuptools wheel
sudo -u astra ./venv/bin/pip install -r requirements.txt
```

### 4. Create `.env`
```bash
cp .env.example .env
nano .env
# Fill in JWT_SECRET, GROQ_API_KEY, and CORS_ORIGINS
chmod 600 .env
chown astra:www-data .env
```

### 5. Install Systemd Service
```bash
sudo cp deployment/astra.service /etc/systemd/system/astra.service
sudo systemctl daemon-reload
sudo systemctl enable astra
sudo systemctl start astra
```

### 6. Install Nginx Configuration
```bash
sudo cp deployment/nginx.conf /etc/nginx/sites-available/astra
# Edit domain name:
sudo nano /etc/nginx/sites-available/astra # Replace your-domain.com with your actual domain
sudo ln -sf /etc/nginx/sites-available/astra /etc/nginx/sites-enabled/astra
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

### 7. Issue SSL Certificate
```bash
sudo certbot --nginx -d your-domain.com
```

---

## 6. Method C: GoDaddy cPanel Hosting (Python App Selector)

If your GoDaddy plan is cPanel Web Hosting with "Setup Python App" (CloudLinux / Passenger):

1. **Access cPanel**: Log in to GoDaddy cPanel and click **Setup Python App** under the **Software** section.
2. **Create Application**:
   - **Python Version**: Select `3.10`, `3.11`, or `3.12`.
   - **Application root**: `astra`
   - **Application URL**: `your-domain.com` or `astra.your-domain.com`
   - **Application startup file**: `passenger_wsgi.py`
   - Click **Create**.
3. **Upload Files**: Using cPanel File Manager or FTP, upload project files into `public_html` or `astra/`.
4. **Install Dependencies**:
   - Enter the virtualenv command displayed in cPanel via SSH or cPanel Terminal:
     ```bash
     source /home/username/virtualenv/astra/3.11/bin/activate
     pip install -r requirements.txt
     pip install asgiref
     ```
5. **Create `passenger_wsgi.py`**:
   In the application root directory, create `passenger_wsgi.py`:
   ```python
   import sys, os
   from asgiref.wsgi import WsgiToAsgi
   sys.path.insert(0, os.path.dirname(__file__))
   from app import app as asgi_app
   application = WsgiToAsgi(asgi_app)
   ```
6. **Configure Environment Variables**:
   In cPanel Python App settings, scroll to **Environment variables** and add:
   - `APP_ENV` = `production`
   - `JWT_SECRET` = `(your-random-64-character-secret)`
   - `GROQ_API_KEY` = `(your-groq-key)`
   - `CORS_ORIGINS` = `https://your-domain.com`
7. Click **Restart Application**.

---

## 7. Operational Management & Maintenance

### Managing the Service (Systemd)
```bash
# Check service status and health
sudo systemctl status astra

# Restart application (e.g. after modifying .env)
sudo systemctl restart astra

# View real-time service logs
sudo journalctl -u astra -f

# View Nginx access & error logs
sudo tail -f /var/log/nginx/astra_error.log
sudo tail -f /var/log/nginx/astra_access.log
```

### Applying Future Code Updates
To update your server without manual reconfigurations, run:
```bash
cd /opt/astra
sudo ./deployment/deploy.sh
```
This script backs up your SQLite database, pulls/updates python dependencies, gracefully reloads Gunicorn workers with zero downtime, and checks the health endpoint.

### Backing Up Data
Your application data is stored in `/opt/astra/data/`:
- `aster.db`: SQLite database containing user accounts, chat histories, preferences, and document metadata.
- `uploads/`: Uploaded documents.
To create a backup snapshot:
```bash
tar -czvf /root/astra_backup_$(date +%Y%m%d).tar.gz /opt/astra/data /opt/astra/.env
```

---

## 8. Troubleshooting & FAQ

### 1. `502 Bad Gateway` in Nginx
- **Cause**: The backend service `astra.service` is not running.
- **Fix**: Check `systemctl status astra` and inspect logs using `journalctl -u astra -n 50 --no-pager`. Verify your `.env` file exists and has valid syntax.

### 2. Large File Uploads (400 MB) Fail with `413 Payload Too Large`
- **Cause**: Nginx `client_max_body_size` is too low.
- **Fix**: Verify `client_max_body_size 400M;` is present in `/etc/nginx/sites-available/astra` and restart Nginx (`systemctl restart nginx`).

### 3. AI Responses Are Blank or Fail
- **Cause**: Missing or expired LLM API key.
- **Fix**: Check `/opt/astra/.env` and verify `GROQ_API_KEY` or `GROK_API_KEY` is valid. Test with `curl` or visit `/health` to verify status.

### 4. Memory Usage Optimization
- If running on a 1 GB or 2 GB RAM GoDaddy VPS, ensure `USE_NEURAL_MODELS=false` in `.env` so Astra uses lightweight API embeddings and saves RAM.
