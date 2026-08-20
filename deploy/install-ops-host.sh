#!/usr/bin/env bash
# Install the host-side pieces required by the Tencent operations dashboard.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sudo install -d -m 0755 /home/ubuntu/uplus-web-logs
sudo chown ubuntu:ubuntu /home/ubuntu/uplus-web-logs
sudo install -m 0644 \
  "$SCRIPT_DIR/uplus-nginx-logrotate" \
  /etc/logrotate.d/uplus-nginx

# Validate syntax only. Rotation remains managed by the host's normal timer.
sudo logrotate --debug /etc/logrotate.d/uplus-nginx >/dev/null
echo "operations host logging installed"
