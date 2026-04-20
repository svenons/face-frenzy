# Face Frenzy

Face Frenzy is a camera-based party game for the PYNQ-Z2. Players are asked to
show a target number of faces before the countdown ends. The board captures a
frame, counts faces, and gives score/strike feedback through the web interface,
buttons, and LEDs.

This version uses a GPIO-merged FINN bitstream for FPGA-backed face detection.
There is no CPU face-detection fallback; detection failures are surfaced as FPGA
errors so the hardware path stays honest.

## Features

- FPGA-backed LPYOLO/FINN face detector using the vendored `fpga/finn-accel.bit`
  and generated FINN driver.
- Browser UI with live camera feed, face boxes, face count, game state, and
  controls.
- AXI GPIO support for PYNQ-Z2 buttons and LEDs from the same merged overlay.
- USB camera capture with boot-time camera waiting instead of process exit.
- HDMI output is intentionally disabled for this FPGA build because the FINN
  design is close to the PYNQ-Z2 fabric limit.
- Flask web app at `http://<board-ip>:5000/`, plus JSON state at
  `http://<board-ip>:5000/api/state`.
- `systemd` autostart service included in the repository.
- App log at `/home/xilinx/face-frenzy/face-frenzy.log`, plus journald output.

## Hardware And Runtime

- PYNQ-Z2 board
- PYNQ 3.1 SD image
- USB webcam
- Network connection for SSH and the browser UI

The service and manual run commands expect the repository at:

```bash
/home/xilinx/face-frenzy
```

## Repository Layout

```text
face-frenzy/
|-- main.py
|-- CameraManager.py
|-- DisplayManager.py
|-- FaceDetector.py
|-- GameController.py
|-- GameStateExporter.py
|-- IOHandler.py
|-- WebServer.py
|-- requirements.txt
|-- fpga/
|   |-- finn-accel.bit
|   |-- finn-accel.hwh
|   |-- driver.py
|   |-- driver_base.py
|   |-- scale.npy              # optional, copied from the FINN deploy package
|   |-- finn/
|   |-- qonnx/
|   `-- runtime_weights/
|-- systemd/
|   `-- face-frenzy.service
`-- scripts/
    |-- setup-pynq.sh
    |-- smoke-pynq.sh
    |-- fpga-exec-smoke.py
    |-- fpga-frame-debug.py
    |-- verify-imports.py
    `-- install-service.sh
```

## How The Game Works

1. Select 1-4 players with BTN0/BTN1 or the browser controls.
2. Press BTN3 to start.
3. The game asks for a random number of visible faces.
4. The browser live view continuously shows detected faces and the current face count.
5. A short countdown runs in the browser and on the LEDs.
6. The browser feed flashes white and the camera captures the scoring frame.
7. The FPGA detector counts faces in that captured frame.
8. Matching the target count gives a point; missing it gives a strike.
9. The game resets after too many strikes.

## Setup On The Board

Clone or copy this repository onto the PYNQ-Z2:

```bash
cd /home/xilinx
git clone https://github.com/svenons/face-frenzy.git
cd face-frenzy
```

Install board packages and Python dependencies:

```bash
bash scripts/setup-pynq.sh
```

`scripts/setup-pynq.sh` deliberately installs OpenCV through apt:

```bash
sudo apt update
sudo apt install -y python3-opencv
```

Do not install `opencv-python` with pip on the PYNQ-Z2. The board is armv7, so
pip often tries to build OpenCV from source. The remaining Python dependencies
are installed into the root PYNQ environment with:

```bash
sudo bash -lc 'cd /home/xilinx/face-frenzy && source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh && python3 -m pip install -r requirements.txt'
```

`requirements.txt` includes `bitstring`, which the generated FINN driver needs
for tensor packing. Installing into the root environment matters because the
overlay smoke tests, manual run, and systemd service run as root on this board.

## Root Shells And PYNQ Env

Overlay/XRT/GPIO access on PYNQ commonly needs root privileges, especially when
running over SSH. Run smoke tests and manual game launches from a root shell:

```bash
sudo -i
cd /home/xilinx/face-frenzy
source /etc/profile.d/pynq_venv.sh
source /etc/profile.d/xrt_setup.sh
```

Important: `sudo -i` starts a new shell and loses the environment you sourced as
`xilinx`. Source both profile scripts again after becoming root.

You can also run one-off commands without an interactive root shell:

```bash
sudo bash -lc 'cd /home/xilinx/face-frenzy && source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh && python3 main.py'
```

## Smoke Tests

Run these before installing the service.

The recommended smoke test checks imports, `bitstring`, OpenCV, overlay load,
GPIO names, LED writes, and the button bank:

```bash
cd /home/xilinx/face-frenzy
sudo bash scripts/smoke-pynq.sh
```

Expected: all four user LEDs turn on for one second, GPIO attributes are listed,
and the script ends with `[smoke] OK`.

### Manual Overlay And LEDs

If you want to run the LED test by hand, use a root shell with the PYNQ
environment sourced:

```bash
sudo -i
cd /home/xilinx/face-frenzy
source /etc/profile.d/pynq_venv.sh
source /etc/profile.d/xrt_setup.sh

python3 - <<'PY'
from FaceDetector import load_finn_accelerator
import time

ol = load_finn_accelerator("fpga/finn-accel.bit", "fpga")
print([name for name in dir(ol) if "gpio" in name.lower()])
ol.leds_gpio.write(0, 0b1111)
time.sleep(1)
ol.leds_gpio.write(0, 0)
PY
```

Expected: all four user LEDs turn on for one second.

### Manual Buttons

```bash
sudo -i
cd /home/xilinx/face-frenzy
source /etc/profile.d/pynq_venv.sh
source /etc/profile.d/xrt_setup.sh

python3 - <<'PY'
from FaceDetector import load_finn_accelerator
import time

ol = load_finn_accelerator("fpga/finn-accel.bit", "fpga")
for _ in range(50):
    print(f"btns = {ol.btns_gpio.read(0):04b}")
    time.sleep(0.1)
PY
```

Expected: pressing BTN0 sets bit 0, BTN3 sets bit 3, and so on.

### FPGA Detector Path

```bash
sudo -i
cd /home/xilinx/face-frenzy
source /etc/profile.d/pynq_venv.sh
source /etc/profile.d/xrt_setup.sh

python3 - <<'PY'
import numpy as np
import FaceDetector as fd
from FaceDetector import FaceDetector, load_finn_accelerator

ol = load_finn_accelerator("fpga/finn-accel.bit", "fpga")
fd.set_accelerator(ol)
detector = FaceDetector()
print("backend:", detector.backend)
boxes, _ = detector.detect_faces_with_boxes(
    np.zeros((480, 640, 3), dtype=np.uint8)
)
print("boxes:", boxes)
PY
```

Expected: `backend: fpga`. The black frame is only a driver/DMA check, not an
accuracy test.

### FPGA Execute / DMA Path

If the game gets stuck on `Detecting...`, test whether one FINN execute returns
at all:

```bash
sudo -i
cd /home/xilinx/face-frenzy
source /etc/profile.d/pynq_venv.sh
source /etc/profile.d/xrt_setup.sh
python3 scripts/fpga-exec-smoke.py --timeout 20
```

Expected: `OK: FINN execute returned`, plus input/output shapes and output
min/max/mean. If this times out, the problem is below the game loop: bitstream,
HWH, FINN driver, IODMA, or XRT/overlay state.

## Manual Run

```bash
sudo -i
cd /home/xilinx/face-frenzy
source /etc/profile.d/pynq_venv.sh
source /etc/profile.d/xrt_setup.sh
python3 main.py
```

The app will wait until the USB camera is available. HDMI output is disabled
for this bitstream; use the browser UI instead:

```text
http://<board-ip>:5000/
```

The browser shows the live feed with green boxes around detected faces, a
`Faces: N` count, game state, selected players, score, strikes, and controls for
less/more/start/reset. The white flash still appears in the browser feed at the
end of the countdown before the scoring frame is captured.

The video stream is published independently of FPGA inference so the browser can
show camera frames even if the detector is slow. Live face boxes are updated by
a throttled background detector once per second by default. To tune that rate:

```bash
export FACE_FRENZY_DETECTION_INTERVAL_S=1.5
python3 main.py
```

Scoring detection has its own timeout so the game does not sit on `Detecting...`
forever if FINN execute hangs:

```bash
export FACE_FRENZY_SCORING_TIMEOUT_S=8
python3 main.py
```

If the FPGA path runs but reports zero boxes, the detector logs output min/max
and the best YOLO layout candidates every few seconds. You can tune the FPGA
decoder without rebuilding the bitstream:

```bash
find /home/xilinx -name scale.npy 2>/dev/null
export FACE_FRENZY_CONF_THRESHOLD=0.05
export FACE_FRENZY_OUTPUT_SCALE=64
export FACE_FRENZY_YOLO_LAYOUT=auto
export FACE_FRENZY_SCORE_MODE=product
export FACE_FRENZY_CHANNEL_ORDER=rgb
python3 main.py
```

If the FINN deploy package produced `scale.npy`, copy it to `fpga/scale.npy`.
`FaceDetector.py` loads it automatically unless `FACE_FRENZY_OUTPUT_SCALE` is
set. If no scale file is present, the detector defaults to `64`.

For one-frame camera debugging, run:

```bash
python3 scripts/fpga-frame-debug.py --out fpga-frame-debug.jpg
```

The script captures one camera frame, runs one FPGA inference, prints the output
statistics, selected decoder layout, and raw/kept box counts, then writes an
annotated JPEG. The detector rejects obvious non-face geometry such as full-width
horizontal bars and full-height vertical stripes.

Useful decoder diagnostics:

```bash
python3 scripts/fpga-frame-debug.py --sweep
python3 scripts/fpga-frame-debug.py --cell 6 6
python3 scripts/fpga-frame-debug.py --save-raw /tmp/raw_output.npy
```

Physical controls still work: player selection starts at 1, BTN0 increases
player count, BTN1 decreases it, and BTN3 starts the game. While idle, the LEDs
alternate between all four LEDs and the selected player count.

Useful logs:

```bash
tail -f /home/xilinx/face-frenzy/face-frenzy.log
```

## Deployment Last: Autostart Service

Install the service only after the smoke tests and manual run work:

```bash
cd /home/xilinx/face-frenzy
bash scripts/install-service.sh
sudo systemctl start face-frenzy.service
```

Watch the service:

```bash
sudo systemctl status face-frenzy.service
sudo journalctl -u face-frenzy.service -f
```

Enable is handled by `scripts/install-service.sh`, so the game starts on future
boots. The service runs as `root` so the process has the privileges normally
needed for PYNQ overlay/XRT/GPIO access. To test power-on behavior:

```bash
sudo reboot
```

After reboot, check:

```bash
sudo journalctl -u face-frenzy.service -f
tail -f /home/xilinx/face-frenzy/face-frenzy.log
```

Expected boot flow:

1. `face-frenzy.service` starts.
2. `fpga/finn-accel.bit` loads.
3. The app waits for the camera if needed.
4. The detector reports `fpga`.
5. The web server starts.
6. The browser UI is available at `http://<board-ip>:5000/`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Overlay` or `FINNExampleOverlay` fails to load | Bitstream/HWH mismatch or missing file | Re-copy `fpga/finn-accel.bit` and `fpga/finn-accel.hwh` together |
| `AttributeError: leds_gpio` or `btns_gpio` | Wrong bitstream, not the GPIO-merged build | Use the merged overlay in `fpga/` |
| `ModuleNotFoundError: No module named 'bitstring'` | FINN tensor-packing dependency missing from the root PYNQ Python environment | Run `bash scripts/setup-pynq.sh`, or run `sudo bash -lc 'cd /home/xilinx/face-frenzy && source /etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh && python3 -m pip install -r requirements.txt'` |
| `pip install opencv-python` tries to build from source | PYNQ-Z2 is armv7 and pip wheels are not reliable for OpenCV | Do not pip-install `opencv-python`; use `sudo apt install -y python3-opencv` |
| `RuntimeError: Could not open device with index '0'` | XRT/overlay access was attempted without the privileges this board needs | Run from `sudo -i` and source both env scripts again, or use the root systemd service |
| Overlay/XRT access fails under SSH or as `xilinx` | Insufficient privileges for FPGA manager, XRT, or GPIO access | Run smoke tests/manual launch from `sudo -i` after re-sourcing the PYNQ env; the systemd service runs as root |
| Browser shows `FPGA: ...` error | FPGA detector failed during live or scoring inference | Check `face-frenzy.log`; there is intentionally no CPU fallback |
| Game stays on `Detecting...` | FINN execute is not returning or the scoring timeout is too high | Run `python3 scripts/fpga-exec-smoke.py --timeout 20`; lower `FACE_FRENZY_SCORING_TIMEOUT_S` while debugging |
| Service cannot import `pynq` | PYNQ environment not sourced | Keep the `bash -lc 'source ...'` command in the service file |
| Game waits forever | USB camera missing or not detected | Plug in the camera and check `/dev/video*` |
| HDMI is blank | HDMI is intentionally not used with this FPGA build | Use `http://<board-ip>:5000/` for live video, overlays, and controls |
| Browser page loads but video is stale | Detector or camera loop is overloaded | Increase `FACE_FRENZY_DETECTION_INTERVAL_S`; the stream no longer waits for each detector pass |
| Face count stays at 0 | FPGA decoder scale/layout/channel order is not matching the bitstream output | Copy `scale.npy` into `fpga/` if available; run `python3 scripts/fpga-frame-debug.py --sweep`; try `FACE_FRENZY_OUTPUT_SCALE=64` |
| Pressing physical buttons destabilizes the board | GPIO/button IP or overlay is still suspect | Use browser controls first, then re-test `sudo bash scripts/smoke-pynq.sh`; GPIO polling and LED writes are rate-limited in the app |

## Notes

The generated FINN driver currently reports input shape `(1, 416, 416, 8)` and
output shape `(1, 13, 13, 18)`. `FaceDetector.py` follows those shapes at
runtime. If the bitstream is rebuilt with different I/O shapes, update or verify
the detector preprocessing and YOLO decoder.

Spatial same-face scoring by bounding-box overlap is still a separate follow-up
from FPGA bring-up.

> This repository is not affiliated with SDU or Krzysztof Sierszecki.
