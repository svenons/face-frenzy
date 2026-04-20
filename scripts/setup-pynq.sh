#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/xilinx/face-frenzy"

if [[ -d "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" ]]; then
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

cd "${REPO_DIR}"

echo "[setup] Installing OpenCV from apt for PYNQ armv7"
if [[ "${EUID}" -eq 0 ]]; then
    apt update
    apt install -y python3-opencv
else
    sudo apt update
    sudo apt install -y python3-opencv
fi

echo "[setup] Installing Python dependencies into the root PYNQ environment"
if [[ "${EUID}" -eq 0 ]]; then
    source /etc/profile.d/pynq_venv.sh
    source /etc/profile.d/xrt_setup.sh
    python3 -m pip install -r requirements.txt

    echo "[setup] Verifying root/PYNQ imports"
    python3 - <<'PY'
import bitstring
import cv2
import flask
import numpy

print("bitstring:", bitstring.__version__)
print("cv2:", cv2.__version__)
print("flask:", getattr(flask, "__version__", "ok"))
print("numpy:", numpy.__version__)
PY
else
    sudo bash -lc "cd '${REPO_DIR}' && \
        source /etc/profile.d/pynq_venv.sh && \
        source /etc/profile.d/xrt_setup.sh && \
        python3 -m pip install -r requirements.txt && \
        python3 scripts/verify-imports.py"
fi

echo "[setup] Done"
