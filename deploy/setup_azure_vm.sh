#!/bin/bash
# ============================================================================
# Longevity Quantum Platform — Azure VM Deployment Script
# ============================================================================
# Target: Ubuntu 24.04 LTS on Azure (vm-longevity-quantum @ 20.4.0.0)
# Purpose: Deploy QubitPage-OS + Longevity Research Platform
# ============================================================================

set -e

echo "========================================="
echo " LONGEVITY QUANTUM PLATFORM — DEPLOYER"
echo "========================================="

# --- Configuration ---
export DEBIAN_FRONTEND=noninteractive
INSTALL_DIR="/opt/longevity-quantum"
QUBITPAGE_DIR="/opt/qubitpage-os"

# --- System Update ---
echo "[1/8] Updating system packages..."
apt-get update -qq && apt-get upgrade -y -qq

# --- Python 3.12 ---
echo "[2/8] Installing Python 3.12 and build tools..."
apt-get install -y -qq python3.12 python3.12-venv python3.12-dev python3-pip \
    git build-essential libopenblas-dev liblapack-dev gfortran \
    nginx certbot python3-certbot-nginx curl wget

# --- Clone Repositories ---
echo "[3/8] Cloning QubitPage repositories..."
mkdir -p /opt
cd /opt

if [ ! -d "$QUBITPAGE_DIR" ]; then
    git clone https://github.com/qubitpage/QubitPage-OS.git "$QUBITPAGE_DIR"
fi

if [ ! -d "/opt/qubios" ]; then
    git clone https://github.com/qubitpage/QuBIOS.git /opt/qubios
fi

if [ ! -d "/opt/qlang" ]; then
    git clone https://github.com/qubitpage/QLang.git /opt/qlang
fi

# --- Deploy Longevity Platform ---
echo "[4/8] Deploying Longevity Quantum Platform..."
mkdir -p "$INSTALL_DIR"
# NOTE: This will be rsync'd from local machine. Placeholder for now.

# --- Python Virtual Environment ---
echo "[5/8] Creating Python virtual environment and installing dependencies..."
cd "$INSTALL_DIR"
python3.12 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip wheel setuptools
pip install -r requirements.txt 2>/dev/null || pip install \
    qiskit qiskit-algorithms qiskit-ibm-runtime \
    stim cirq-core openfermion openfermionpyscf \
    pyscf numpy scipy requests \
    google-generativeai flask flask-socketio python-dotenv

# Also install QuBIOS locally
pip install -e /opt/qubios 2>/dev/null || true

# --- Environment Variables ---
echo "[6/8] Configuring environment..."
cat > "$INSTALL_DIR/.env" << 'ENVFILE'
# Longevity Quantum Platform — Environment Configuration
# Fill these in after deployment

# IBM Quantum (required for real hardware VQE)
IBM_QUANTUM_TOKEN=

# Google Gemini (required for AI orchestration)
GEMINI_API_KEY=

# Groq (optional — fast LLM inference)
GROQ_API_KEY=

# HuggingFace (required for TxGemma ADMET prediction)
HUGGINGFACE_TOKEN=

# NCBI API Key (optional — faster PubMed queries)
NCBI_API_KEY=
ENVFILE

# --- Systemd Service ---
echo "[7/8] Creating systemd service..."
cat > /etc/systemd/system/longevity-quantum.service << 'SERVICE'
[Unit]
Description=Longevity Quantum Platform (QubitPage OS)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/longevity-quantum
Environment="PATH=/opt/longevity-quantum/.venv/bin:/usr/local/bin:/usr/bin"
EnvironmentFile=/opt/longevity-quantum/.env
ExecStart=/opt/longevity-quantum/.venv/bin/python src/app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
# Don't start yet — needs .env to be filled

# --- Nginx Reverse Proxy ---
echo "[8/9] Configuring Nginx for quantumqub.com..."
cat > /etc/nginx/sites-available/longevity << 'NGINX'
server {
    listen 80;
    server_name quantumqub.com www.quantumqub.com;

    location / {
        proxy_pass http://127.0.0.1:5050;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:5050/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
NGINX

ln -sf /etc/nginx/sites-available/longevity /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl restart nginx

# --- SSL with Let's Encrypt ---
echo "[9/9] Setting up SSL certificate for quantumqub.com..."
certbot --nginx -d quantumqub.com -d www.quantumqub.com --non-interactive --agree-tos -m msrusu87@outlook.com || echo "SSL cert may need DNS propagation — retry: certbot --nginx -d quantumqub.com"

# --- Firewall ---
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 5050/tcp
ufw --force enable

echo ""
echo "========================================="
echo " DEPLOYMENT COMPLETE"
echo "========================================="
echo ""
echo " Server: http://$(curl -s ifconfig.me):80"
echo " Install dir: $INSTALL_DIR"
echo " QubitPage-OS: $QUBITPAGE_DIR"
echo ""
echo " NEXT STEPS:"
echo " 1. Edit /opt/longevity-quantum/.env with API keys"
echo " 2. rsync your local platform code to $INSTALL_DIR"
echo " 3. systemctl start longevity-quantum"
echo " 4. Verify: curl http://localhost:5050/api/qubilogic/status"
echo ""
echo " To run data pipeline:"
echo "   cd $INSTALL_DIR && source .venv/bin/activate"
echo "   python src/longevity_data.py"
echo ""
echo " To run quantum simulations:"
echo "   python src/longevity_sim.py"
echo ""
