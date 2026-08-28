#!/bin/bash
# deploy/deploy.sh
# Builds Docker image, pushes to Docker Hub, SSHs into EC2 and restarts container.
# Required env vars:
#   DOCKER_USERNAME   - Docker Hub username
#   DOCKER_PASSWORD   - Docker Hub access token
#   EC2_HOST          - EC2 public IP or DNS
#   EC2_USER          - SSH user (default: ubuntu)
#   EC2_SSH_KEY       - Path to PEM key file (for local runs)

set -e  # exit on any error

# --- Config ---
DOCKER_USERNAME=${DOCKER_USERNAME:?"DOCKER_USERNAME is required"}
EC2_HOST=${EC2_HOST:?"EC2_HOST is required"}
EC2_USER=${EC2_USER:-"ubuntu"}
IMAGE_NAME="${DOCKER_USERNAME}/mlops-app"
IMAGE_TAG="${GITHUB_SHA:-latest}"

echo "========================================"
echo " MLOps Deploy Script"
echo " Image : ${IMAGE_NAME}:${IMAGE_TAG}"
echo " EC2   : ${EC2_USER}@${EC2_HOST}"
echo "========================================"

# --- Step 1: Build Docker image ---
echo "[1/4] Building Docker image..."
docker build -t "${IMAGE_NAME}:latest" -t "${IMAGE_NAME}:${IMAGE_TAG}" .
echo "      Build complete."

# --- Step 2: Push to Docker Hub ---
echo "[2/4] Pushing to Docker Hub..."
echo "${DOCKER_PASSWORD}" | docker login -u "${DOCKER_USERNAME}" --password-stdin
docker push "${IMAGE_NAME}:latest"
docker push "${IMAGE_NAME}:${IMAGE_TAG}"
echo "      Push complete."

# --- Step 3: SSH into EC2 and deploy ---
echo "[3/4] Deploying to EC2..."

# Write SSH key to temp file if provided as env var (used in CI/CD)
if [ -n "${EC2_SSH_KEY_CONTENT}" ]; then
    KEY_FILE=$(mktemp)
    echo "${EC2_SSH_KEY_CONTENT}" > "${KEY_FILE}"
    chmod 600 "${KEY_FILE}"
    SSH_OPT="-i ${KEY_FILE}"
else
    SSH_OPT="-i ${EC2_SSH_KEY:-~/.ssh/id_rsa}"
fi

ssh -o StrictHostKeyChecking=no ${SSH_OPT} "${EC2_USER}@${EC2_HOST}" << EOF
    echo "Pulling latest image..."
    docker pull ${IMAGE_NAME}:latest

    echo "Stopping old container (if running)..."
    docker stop mlops-app 2>/dev/null || true
    docker rm mlops-app 2>/dev/null || true

    echo "Starting new container..."
    docker run -d \
        --name mlops-app \
        --restart unless-stopped \
        -p 8000:8000 \
        -e SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL}" \
        ${IMAGE_NAME}:latest

    echo "Container started."
EOF

# Cleanup temp key file
[ -n "${EC2_SSH_KEY_CONTENT}" ] && rm -f "${KEY_FILE}"

# --- Step 4: Health check ---
echo "[4/4] Verifying deployment..."
sleep 5  # give container time to start

HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://${EC2_HOST}:8000/health" || echo "000")

if [ "${HTTP_STATUS}" = "200" ]; then
    echo "      ✅ Health check passed (HTTP ${HTTP_STATUS})"
else
    echo "      ❌ Health check FAILED (HTTP ${HTTP_STATUS})"
    exit 1
fi

echo "========================================"
echo " Deployment complete!"
echo " /health  → http://${EC2_HOST}:8000/health"
echo " /predict → http://${EC2_HOST}:8000/predict"
echo " /metrics → http://${EC2_HOST}:8000/metrics"
echo "========================================"