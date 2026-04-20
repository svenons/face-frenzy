#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
service_src="${repo_dir}/systemd/face-frenzy.service"
service_dst="/etc/systemd/system/face-frenzy.service"

sudo install -m 0644 "${service_src}" "${service_dst}"
sudo systemctl daemon-reload
sudo systemctl enable face-frenzy.service

echo "Installed and enabled face-frenzy.service"
echo "The service runs as root so PYNQ overlay/XRT/GPIO access works reliably."
echo "Start it now with: sudo systemctl start face-frenzy.service"
