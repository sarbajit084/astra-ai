#!/usr/bin/env bash
# ==============================================================================
# Astra RAG + AI Agent - Fast Update & Deployment Script
# Use this script to apply updates to your live GoDaddy server with zero downtime.
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

APP_DIR="/opt/astra"
BACKUP_DIR="/opt/astra/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo -e "${CYAN}=== Deploying Astra Update (${TIMESTAMP}) ===${NC}"

# Check for root / sudo privileges
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}[ERROR] Please run this script with sudo (sudo ./deploy.sh)${NC}" 1>&2
   exit 1
fi

# 1. Automatic Backup of Database & Settings
echo -e "${YELLOW}[1/4] Creating pre-deploy database backup...${NC}"
mkdir -p "${BACKUP_DIR}"
if [[ -f "${APP_DIR}/data/aster.db" ]]; then
    cp "${APP_DIR}/data/aster.db" "${BACKUP_DIR}/aster_${TIMESTAMP}.db"
    echo -e "${GREEN}Database backed up to ${BACKUP_DIR}/aster_${TIMESTAMP}.db${NC}"
fi

# 2. Update Python Dependencies
echo -e "${YELLOW}[2/4] Verifying and updating Python virtual environment dependencies...${NC}"
"${APP_DIR}/venv/bin/pip" install --upgrade pip
"${APP_DIR}/venv/bin/pip" install -r "${APP_DIR}/requirements.txt"

# Ensure permissions
chown -R astra:www-data "${APP_DIR}"
chmod -R 750 "${APP_DIR}"
chmod 600 "${APP_DIR}/.env" || true

# 3. Graceful Reload / Restart
echo -e "${YELLOW}[3/4] Gracefully reloading application service...${NC}"
if systemctl is-active --quiet astra; then
    systemctl reload astra || systemctl restart astra
else
    systemctl start astra
fi

# 4. Service Verification & Health Check
echo -e "${YELLOW}[4/4] Performing health check verification...${NC}"
sleep 2

HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health || echo "failed")

if [[ "${HEALTH_STATUS}" == "200" ]]; then
    echo -e "${GREEN}SUCCESS: Astra service is live and healthy (HTTP 200 OK)!${NC}"
else
    echo -e "${RED}WARNING: Healthcheck returned status ${HEALTH_STATUS}. Checking logs:${NC}"
    journalctl -u astra -n 25 --no-pager
    exit 1
fi

echo -e "${CYAN}=== Deployment completed successfully! ===${NC}"
