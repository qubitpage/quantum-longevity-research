#!/bin/bash
# ============================================================================
# Deploy Longevity Platform to Azure VM (run from Windows/local machine)
# ============================================================================
# Usage: bash deploy/push_to_azure.sh
# Requires: SSH key access to 20.4.0.0 as user 'qubitpage'
# ============================================================================

VM_IP="20.4.0.0"
VM_USER="qubitpage"
REMOTE_DIR="/opt/longevity-quantum"
LOCAL_DIR="$(dirname "$0")/.."

echo "=== Deploying Longevity Quantum Platform to Azure VM ==="
echo "Target: ${VM_USER}@${VM_IP}:${REMOTE_DIR}"

# 1. Sync platform code
echo "[1/4] Syncing platform code..."
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='data/*.pdb' \
    "$LOCAL_DIR/" "${VM_USER}@${VM_IP}:${REMOTE_DIR}/"

# 2. Run setup script (first time only)
echo "[2/4] Running setup script..."
ssh "${VM_USER}@${VM_IP}" "sudo bash ${REMOTE_DIR}/deploy/setup_azure_vm.sh"

# 3. Configure API keys (copy from local .env.production)
echo "[3/4] Configuring API keys..."
scp deploy/.env.production "${VM_USER}@${VM_IP}:${REMOTE_DIR}/.env"

# 4. Start service
echo "[4/4] Starting service..."
ssh "${VM_USER}@${VM_IP}" "sudo systemctl start longevity-quantum && sudo systemctl enable longevity-quantum"

echo ""
echo "=== DEPLOYMENT COMPLETE ==="
echo "Platform: http://${VM_IP}"
echo "API: http://${VM_IP}/api/qubilogic/status"
echo ""
echo "To run data pipeline:"
echo "  ssh ${VM_USER}@${VM_IP}"
echo "  cd ${REMOTE_DIR} && source .venv/bin/activate"
echo "  python src/longevity_data.py"
