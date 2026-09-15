#!/usr/bin/env bash
# ==============================================================================
# Astra RAG + AI Agent - Automated Production Setup Script
# Target: GoDaddy Linux VPS / Dedicated Server (Ubuntu 20.04/22.04/24.04, Debian 11/12)
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}======================================================${NC}"
echo -e "${CYAN}   Astra RAG + AI Agent - Production Setup Server     ${NC}"
echo -e "${CYAN}======================================================${NC}"

# Check for root / sudo privileges
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}[ERROR] This script must be run as root (use sudo ./setup.sh)${NC}" 1>&2
   exit 1
fi

APP_DIR="/opt/astra"
APP_USER="astra"
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1. System Package Installation
echo -e "\n${YELLOW}[1/7] Updating package index and installing dependencies...${NC}"
apt-get update -y
apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    nginx \
    curl \
    git \
    unzip \
    libpq-dev \
    certbot \
    python3-certbot-nginx

# 2. Service User Setup
echo -e "\n${YELLOW}[2/7] Configuring dedicated application user '${APP_USER}'...${NC}"
if ! id -u "${APP_USER}" >/dev/null 2>&1; then
    useradd -r -s /bin/false -d "${APP_DIR}" "${APP_USER}"
    echo -e "${GREEN}Created service user '${APP_USER}'${NC}"
else
    echo -e "${GREEN}Service user '${APP_USER}' already exists${NC}"
fi
usermod -aG www-data "${APP_USER}" || true

# 3. Application Directory Setup
echo -e "\n${YELLOW}[3/7] Setting up application directory at ${APP_DIR}...${NC}"
mkdir -p "${APP_DIR}"
mkdir -p "${APP_DIR}/data"
mkdir -p "${APP_DIR}/data/images"
mkdir -p "${APP_DIR}/uploads"
mkdir -p /var/log/astra

# If running directly from extracted archive outside /opt/astra, copy files
if [[ "${CURRENT_DIR}" != "${APP_DIR}" ]]; then
    echo "Copying files from ${CURRENT_DIR} to ${APP_DIR}..."
    rsync -av --exclude="venv" --exclude="__pycache__" --exclude=".git" --exclude="*.db" "${CURRENT_DIR}/" "${APP_DIR}/"
fi

# 4. Python Virtual Environment Setup
echo -e "\n${YELLOW}[4/7] Setting up Python virtual environment and packages...${NC}"
python3 -m venv "${APP_DIR}/venv"
"${APP_DIR}/venv/bin/pip" install --upgrade pip setuptools wheel
"${APP_DIR}/venv/bin/pip" install -r "${APP_DIR}/requirements.txt"
"${APP_DIR}/venv/bin/pip" install gunicorn

# 5. Environment File (.env) Configuration
echo -e "\n${YELLOW}[5/7] Configuring environment variables (.env)...${NC}"
if [[ ! -f "${APP_DIR}/.env" ]]; then
    if [[ -f "${APP_DIR}/.env.example" ]]; then
        cp "${APP_DIR}/.env.example" "${APP_DIR}/.env"
        # Generate a random cryptographically secure JWT secret
        RANDOM_JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(48))")
        sed -i "s/your-production-super-secret-jwt-key-change-this-now/${RANDOM_JWT_SECRET}/" "${APP_DIR}/.env"
        echo -e "${GREEN}Generated fresh .env file with secure JWT secret.${NC}"
        echo -e "${YELLOW}IMPORTANT: Edit '${APP_DIR}/.env' to add your GROQ_API_KEY or GROK_API_KEY!${NC}"
    else
        echo -e "${RED}Warning: .env.example not found. Please create ${APP_DIR}/.env manually.${NC}"
    fi
else
    echo -e "${GREEN}Existing .env file detected; preserving configuration.${NC}"
fi

# Set proper ownership and file permissions
chown -R "${APP_USER}:www-data" "${APP_DIR}"
chmod -R 750 "${APP_DIR}"
chmod 600 "${APP_DIR}/.env" || true
chown -R "${APP_USER}:www-data" /var/log/astra

# 6. Systemd Service Installation
echo -e "\n${YELLOW}[6/7] Installing and starting systemd service...${NC}"
cp "${APP_DIR}/deployment/astra.service" /etc/systemd/system/astra.service
systemctl daemon-reload
systemctl enable astra.service
systemctl restart astra.service
echo -e "${GREEN}Astra service enabled and started.${NC}"

# 7. Nginx Reverse Proxy Setup
echo -e "\n${YELLOW}[7/7] Configuring Nginx reverse proxy...${NC}"
cp "${APP_DIR}/deployment/nginx.conf" /etc/nginx/sites-available/astra
ln -sf /etc/nginx/sites-available/astra /etc/nginx/sites-enabled/astra
rm -f /etc/nginx/sites-enabled/default || true

if nginx -t; then
    systemctl restart nginx
    echo -e "${GREEN}Nginx configured and restarted successfully.${NC}"
else
    echo -e "${RED}[ERROR] Nginx configuration test failed. Please check /etc/nginx/sites-available/astra${NC}"
fi

echo -e "\n${CYAN}======================================================${NC}"
echo -e "${GREEN}   Astra RAG + AI Agent Setup Completed Successfully! ${NC}"
echo -e "${CYAN}======================================================${NC}"
echo -e "Next steps:"
echo -e "1. Edit API Keys:  ${YELLOW}nano ${APP_DIR}/.env${NC} (Add GROQ_API_KEY or GROK_API_KEY)"
echo -e "2. Restart Astra:  ${YELLOW}systemctl restart astra${NC}"
echo -e "3. Check Logs:     ${YELLOW}journalctl -u astra -f${NC}"
echo -e "4. Configure SSL:  ${YELLOW}certbot --nginx -d your-domain.com -d www.your-domain.com${NC}"
echo -e "${CYAN}======================================================${NC}"
