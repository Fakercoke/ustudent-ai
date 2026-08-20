#!/usr/bin/env bash
# Install the host-side pieces required by the Tencent operations dashboard.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The Tencent compose deliberately treats the existing database volume as
# external so upgrades never make Compose think it owns/deletes student data.
docker volume inspect uplus-postgres-data >/dev/null 2>&1 \
  || docker volume create uplus-postgres-data >/dev/null

sudo install -d -m 0755 /home/ubuntu/uplus-web-logs
sudo chown ubuntu:ubuntu /home/ubuntu/uplus-web-logs
sudo install -m 0644 \
  "$SCRIPT_DIR/uplus-nginx-logrotate" \
  /etc/logrotate.d/uplus-nginx

# Validate syntax only. Rotation remains managed by the host's normal timer.
sudo logrotate --debug /etc/logrotate.d/uplus-nginx >/dev/null
echo "operations host logging installed"
