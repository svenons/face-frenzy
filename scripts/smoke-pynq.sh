#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/xilinx/face-frenzy"

if [[ -d "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" ]]; then
    REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

cd "${REPO_DIR}"

if [[ "${EUID}" -ne 0 ]]; then
    echo "[smoke] Warning: not running as root. Overlay/XRT/GPIO access may fail on PYNQ."
fi

source /etc/profile.d/pynq_venv.sh
source /etc/profile.d/xrt_setup.sh

python3 - <<'PY'
import importlib
import time

for module_name in ("bitstring", "cv2", "flask", "numpy", "pynq"):
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", "ok")
    print(f"[smoke] import {module_name}: {version}")

from FaceDetector import load_finn_accelerator

ol = load_finn_accelerator("fpga/finn-accel.bit", "fpga")
gpio_names = [name for name in dir(ol) if "gpio" in name.lower()]
print("[smoke] gpio attributes:", gpio_names)

required = ("leds_gpio", "btns_gpio")
missing = [name for name in required if not hasattr(ol, name)]
if missing:
    raise RuntimeError(f"Missing expected GPIO attributes: {missing}")

print("[smoke] flashing LEDs")
ol.leds_gpio.write(0, 0b1111)
time.sleep(1)
ol.leds_gpio.write(0, 0)

print(f"[smoke] button bank: {ol.btns_gpio.read(0):04b}")
print("[smoke] OK")
PY
