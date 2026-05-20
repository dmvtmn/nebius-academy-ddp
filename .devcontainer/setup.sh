#!/bin/bash
set -e

echo "==> Installing Nebius CLI..."
curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
export PATH="$HOME/.nebius/bin:$PATH"
echo 'export PATH="$HOME/.nebius/bin:$PATH"' >> ~/.bashrc

echo "==> Installing uv + SkyPilot..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
uv tool install --with pip "skypilot[nebius]"

echo "==> Verifying tools..."
nebius version || echo "Nebius CLI installed - restart shell to use"
kubectl version --client
docker --version
sky --version

echo ""
echo "✅ Environment ready! Next step: run 'nebius profile create' to authenticate."
