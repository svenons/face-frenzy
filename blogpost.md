---
title: LPYOLO on PYNQ-Z2
description: Building a low-precision YOLO face detector bitstream for PYNQ-Z2 with FINN, Vivado, and a GPIO-capable overlay.
date: 2026-04-19
tags:
  - project
  - embedded
  - fpga
  - machine-learning
  - pynq
---

LPYOLO on PYNQ-Z2 is a build note from the Face Frenzy project, where I explored turning a low-precision YOLO face detector into an FPGA bitstream. The goal was to generate the FINN accelerator and keep the option of using the board's buttons, switches, and LEDs in the same overlay.

The final working setup uses the 2W4A LPYOLO model. I tried the larger variants too, but the PYNQ-Z2 was not exactly thrilled about them.

## Overview

This part of the project covered three main things:

- building the LPYOLO FPGA bitstream with FINN and Vivado
- generating the PYNQ-compatible driver and bitstream package
- restoring GPIO access so the FINN overlay could still be used with the board's physical inputs and LEDs

## Build Setup

The working toolchain was locked to FINN v0.10.1, Vivado 2022.2, PYNQ 3.1, and a Jupyter notebook workflow inside the FINN Docker environment.

The rest of this post is intentionally closer to a technical build log than a polished tutorial. It keeps the exact commands, patches, and fixes that made the setup reproducible.

## Runtime Corrections From The Face Frenzy Integration

The build below produces a valid FINN accelerator, but the first version of the app-level integration had a few wrong assumptions. The current Face Frenzy runtime uses these rules:

- Put the final bitstream, HWH, and generated driver under `/home/xilinx/face-frenzy/fpga/`, not the older `/home/xilinx/lpyolo/...` paths used in some standalone LPYOLO notes.
- Run overlay load, GPIO, FINN execution, smoke tests, and the app as root on PYNQ-Z2. Over SSH, non-root execution can fail with `RuntimeError: Could not open device with index '0'`.
- After `sudo -i`, source the PYNQ environment again:

```bash
source /etc/profile.d/pynq_venv.sh
source /etc/profile.d/xrt_setup.sh
```

- Install OpenCV through apt on PYNQ armv7:

```bash
sudo apt update
sudo apt install -y python3-opencv
```

Do not use `pip install opencv-python` on the board; it can try to build OpenCV from source.

- Install the generated FINN driver's Python dependency:

```bash
python3 -m pip install bitstring
```

- The generated `UINT4` input packer in FINN's generic Python path is too slow for full 416x416 frames on the Z2. The Face Frenzy repo patches `fpga/driver_base.py` with a vectorized UINT4 pack path. Without it, execution can look "stuck" even though DMA has not started yet.
- HDMI output is not part of the final FINN+GPIO bitstream. The game uses the browser UI at `http://<board-ip>:5000/` for live video, boxes, count, and controls.
- A passing overlay load is not enough. The useful smoke tests are:

```bash
python3 scripts/fpga-exec-smoke.py --timeout 30
FACE_FRENZY_MANUAL_DMA_SMOKE=0 python3 scripts/fpga-exec-smoke.py --timeout 30
python3 scripts/fpga-frame-debug.py --out fpga-frame-debug.jpg
```

At the time of writing, FPGA execution is confirmed at about 0.13-0.15 s per frame on the board. A good execute smoke looks like both IODMAs reaching `0x0e` and then returning to idle `0x04`.

Face box decoding is app-side. Raw `(1, 13, 13, 18)` INT8 output can produce obvious garbage geometry, such as full-height vertical stripes or full-width horizontal bars, if the output scale/layout is wrong or the confidence threshold is too low. The Face Frenzy detector now applies face-shape sanity filters and can be tuned with:

```bash
find /home/xilinx -name scale.npy 2>/dev/null
FACE_FRENZY_CONF_THRESHOLD=0.05
FACE_FRENZY_OUTPUT_SCALE=64
FACE_FRENZY_YOLO_LAYOUT=auto
FACE_FRENZY_SCORE_MODE=product
FACE_FRENZY_CHANNEL_ORDER=rgb
```

If the FINN deploy package contains `scale.npy`, copy it into `fpga/scale.npy`.
`FaceDetector.py` loads it automatically unless `FACE_FRENZY_OUTPUT_SCALE` is
set. If no scale file is present, the runtime defaults to `64`, which is a
better starting point for this INT8 head than `16`.

Use `scripts/fpga-frame-debug.py` for calibration. It captures one camera frame,
runs one FPGA inference, prints the chosen layout and raw/kept box counts, and
writes an annotated JPEG. Extra decoder probes:

```bash
python3 scripts/fpga-frame-debug.py --sweep
python3 scripts/fpga-frame-debug.py --cell 6 6
python3 scripts/fpga-frame-debug.py --save-raw /tmp/raw_output.npy
```

## Repository

[finn-quantized-yolo fork](https://github.com/svenons/finn-quantized-yolo)

## Resources and acknowledgements

This build was pieced together from a few useful starting points:

- [End-to-end LPYOLO deployment tutorial](https://medium.com/@bestamigunay1/end-to-end-deployment-of-lpyolo-low-precision-yolo-for-face-detection-on-fpga-13c3284ed14b) - a useful reference for building LPYOLO with an older FINN version.
- [finn-quantized-yolo](https://github.com/sefaburakokcu/finn-quantized-yolo) - the original repository for the quantized LPYOLO models.
- [Model mirror](https://files.svenons.xyz/share/A_iSSf1A) - backup downloads for the models if the original OneDrive links stop working.

## Build Notes

### Version Lock


| Component            | Version                                        |
| ---------------------- | ------------------------------------------------ |
| FINN compiler        | v0.10.1                                        |
| Vivado / Vitis HLS   | 2022.2 — installed inside WSL2                |
| Host OS              | Windows + WSL2 (Ubuntu 22.04)                  |
| Docker               | Docker Desktop with WSL2 backend               |
| PYNQ runtime         | v3.1.0                                         |
| Python (board)       | 3.10 (bundled with PYNQ 3.1)                   |
| Python (FINN Docker) | 3.10                                           |
| Board                | PYNQ-Z2 (XC7Z020-1CLG400C)                     |
| Model                | LPYOLO 2W4A (2-bit weights, 4-bit activations) |

## Part A - Host Machine Setup

### A1. Install WSL2

In an admin PowerShell:

```powershell
wsl --install
```

Reboot when prompted. This installs WSL2 with Ubuntu 22.04 by default.

### A2. Install Docker Desktop

1. Download from https://docs.docker.com/desktop/install/windows-install/
2. Run the installer — select **Use WSL2 instead of Hyper-V**
3. Open Docker Desktop settings:
   - Resources → WSL Integration → enable your Ubuntu distro
   - Resources → Advanced → set at least **8 GB RAM** and **4 CPUs**
4. Verify inside WSL2:

```bash
docker run hello-world
```

### A3. Install Vivado 2022.2 Inside WSL2

Vivado must be installed INSIDE WSL2, not in Windows. The FINN Docker container mounts the WSL2 Vivado installation.

1. Download the Linux installer from:
   https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vivado-design-tools/2022-2.html

   Get **Xilinx Unified Installer 2022.2: Linux Self Extracting Web Installer**
2. Install it:

```bash
cp /mnt/c/Users/YOU/Downloads/Xilinx_Unified_2022.2_*_Lin64.bin ~/
chmod +x ~/Xilinx_Unified_2022.2_*_Lin64.bin
~/Xilinx_Unified_2022.2_*_Lin64.bin
```

3. In the installer GUI:

   - Product: **Vivado**
   - Edition: **Vivado ML Standard** (free, no licence needed for Zynq 7020)
   - Devices: check **Zynq-7000 only** — deselect everything else to save ~30 GB
   - Install location: `/tools/Xilinx`
4. Installation takes 1–2 hours and needs ~45 GB.
5. Verify:

```bash
source /tools/Xilinx/Vivado/2022.2/settings64.sh
vivado -version   # must print: Vivado v2022.2
```

### A4. Clone FINN

```bash
cd ~
git clone https://github.com/Xilinx/finn.git
cd finn
git checkout 9d299689f2ec0895f208b8bfe3bcdcf6f450181a
git submodule update --init --recursive
```

### A5. Patch the FINN Dockerfile

Two fixes are required or the Docker build will fail. Both edits are in `~/finn/docker/Dockerfile.finn`.

**Fix 1** — pip timeout:

```bash
sed -i 's/RUN pip install -r \/tmp\/requirements.txt/RUN python3 -m pip install --default-timeout=120 --retries 10 --no-cache-dir -r \/tmp\/requirements.txt/' ~/finn/docker/Dockerfile.finn
```

**Fix 2** — anyio version conflict. Find the Jupyter install line:

```bash
grep -n "jupyter" ~/finn/docker/Dockerfile.finn
```

Open the file in a text editor and add this line immediately after the Jupyter install line:

```
RUN pip install "anyio==3.7.1"
```

### A6. Set Environment Variables

Create `~/finn-env.sh`. **All four env vars below are required** — skipping any one will cause a failure later.

```bash
#!/bin/bash
export FINN_XILINX_PATH=/tools/Xilinx
export FINN_XILINX_VERSION=2022.2
export PYNQ_BOARD=Pynq-Z2
export FINN_HOST_BUILD_DIR=$HOME/finn_build
export FINN_DOCKER_GPU=0

# Optional: more parallelism (uses more RAM)
export NUM_DEFAULT_WORKERS=4

# CRITICAL: forwards XILINX_LOCAL_USER_DATA into the container, which
# forces Vivado to read its Tcl store from the install dir (read-only)
# instead of the user home (shared/corruptible between parallel Vivado
# instances during HLS). Without this, HLS export_design will randomly
# fail with: "Unable to load Tcl app xilinx::xsim"
export XILINX_LOCAL_USER_DATA=no
export FINN_DOCKER_EXTRA=" -e XILINX_LOCAL_USER_DATA=no "
```

Source it in every new WSL2 terminal before working with FINN:

```bash
source ~/finn-env.sh
```

### A7. Patch FINN to use area-optimized Vivado strategies

FINN v0.10.1 hardcodes `Flow_PerfOptimized_high` / `Performance_ExtraTimingOpt` in its Vivado synthesis templates, which favour speed over area. On the Z2 this leaves the design a few hundred LUTs short of fitting. Switch to area-optimized strategies before the first build:

```bash
sed -i 's/set_property strategy Flow_PerfOptimized_high/set_property strategy Flow_AreaOptimized_high/' ~/finn/src/finn/transformation/fpgadataflow/templates.py
sed -i 's/set_property strategy Performance_ExtraTimingOpt/set_property strategy Area_Explore/' ~/finn/src/finn/transformation/fpgadataflow/templates.py

# Insert four placer-relaxation lines before launch_runs: ignore minor LUT over-utilization,
# relax control-set packing, enable physical opt passes (recover LUTs post-route).
sed -i '/^launch_runs -to_step write_bitstream impl_1/i set_param drc.disableLUTOverUtilError 1\nset_property STEPS.SYNTH_DESIGN.ARGS.CONTROL_SET_OPT_THRESHOLD 16 [get_runs synth_1]\nset_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]\nset_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]\n' ~/finn/src/finn/transformation/fpgadataflow/templates.py

# clear the cached bytecode so the edit takes effect
rm -f ~/finn/src/finn/transformation/fpgadataflow/__pycache__/templates.cpython-310.pyc
```

Verify:

```bash
grep -nE "strategy|disableLUTOverUtilError|CONTROL_SET_OPT|PHYS_OPT_DESIGN.IS_ENABLED|launch_runs" ~/finn/src/finn/transformation/fpgadataflow/templates.py
```

Expected output (line numbers may vary):

```
set_property strategy Flow_AreaOptimized_high [get_runs synth_1]
set_property strategy Area_Explore [get_runs impl_1]
set_param drc.disableLUTOverUtilError 1
set_property STEPS.SYNTH_DESIGN.ARGS.CONTROL_SET_OPT_THRESHOLD 16 [get_runs synth_1]
set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
launch_runs -to_step write_bitstream impl_1
```

If you had Jupyter running with FINN already imported, restart the kernel after editing.

### A8. Build and Test the FINN Docker Image

```bash
cd ~/finn
source ~/finn-env.sh
./run-docker.sh quicktest
```

Builds the Docker image (~20–30 min first time) and runs the self-test suite. Must print `PASSED` before continuing.

### A9. Launch the Jupyter Notebook Server

This is the workflow for all subsequent build work:

```bash
cd ~/finn
source ~/finn-env.sh
./run-docker.sh notebook
```

Container starts and prints a Jupyter URL. Open it in a browser. All subsequent steps happen in notebook cells.

**Critical: the `finn_build` directory must exist before launching Docker** or you will hit a ghost-inode bug where directories appear to exist but can't be written:

```bash
mkdir -p ~/finn_build
```

If you ever `rm -rf ~/finn_build` while the container is running, you MUST exit and relaunch the container — otherwise all subsequent writes fail with misleading `FileNotFoundError` errors even after recreating the directory.

---

## Part B - Prepare the Model and Config Files

### B1. Get the ONNX Model

1. Go to: https://github.com/sefaburakokcu/finn-quantized-yolo
2. In the README table, find the **2w4a** row and download the ONNX file from OneDrive
3. Place it in a new folder under FINN's notebooks dir:

```bash
mkdir -p ~/finn/notebooks/lpyolo
cp /mnt/c/Users/YOU/Downloads/2w4a.onnx ~/finn/notebooks/lpyolo/
```

Open Jupyter, navigate to `notebooks/lpyolo/`, and create a new notebook there (e.g. `build_lpyolo.ipynb`) for all subsequent cells.

### B2. Create the Folding Config

**This config is the working one for Z2.** Earlier folding configs online (from the LPYOLO paper era) target FINN v0.7 with different node names (`StreamingFCLayer_Batch_0` etc.) and silently fail to apply in v0.10.1, leading to PE=1/SIMD=1 defaults which blow out BRAM and LUTs by 10×+.

The key insights baked into this config:

- Names must be `MVAU_N` (pre-specialize). After `step_specialize_layers`, names get wiped — so folding MUST happen before specialize.
- `mem_mode` value is `"internal_decoupled"` in v0.10.1, not the old `"decoupled"`.
- For Z2's 53,200 LUT and 280 RAMB18 budget, PE/SIMD values from the LPYOLO paper are too aggressive. These values were tuned down until `place_design` actually fits.
- `MVAU_6` uses `"resType": "dsp"` to offload one of the larger compute layers to DSP slices, relieving LUT pressure.
- `StreamingFIFO` default depth of 32 keeps FIFO RAM usage negligible; `auto_fifo_depths` is disabled in the build config because it otherwise sizes FIFOs to full tensor-size (1M+ entries, which blows RAMB36 usage to thousands even after `split_large_fifos`).

Save as `~/finn/notebooks/lpyolo/Pynq-Z2_folding_config.json`:

```json
{
  "Defaults": {
    "depth": [32, ["StreamingFIFO"]]
  },

  "MVAU_0": { "PE": 4, "SIMD": 3, "ram_style": "block", "resType": "lut", "mem_mode": "internal_decoupled", "runtime_writeable_weights": 0 },
  "MVAU_1": { "PE": 4, "SIMD": 4, "ram_style": "block", "resType": "lut", "mem_mode": "internal_decoupled", "runtime_writeable_weights": 0 },
  "MVAU_2": { "PE": 4, "SIMD": 4, "ram_style": "block", "resType": "lut", "mem_mode": "internal_decoupled", "runtime_writeable_weights": 0 },
  "MVAU_3": { "PE": 2, "SIMD": 4, "ram_style": "block", "resType": "lut", "mem_mode": "internal_decoupled", "runtime_writeable_weights": 0 },
  "MVAU_4": { "PE": 1, "SIMD": 4, "ram_style": "block", "resType": "lut", "mem_mode": "internal_decoupled", "runtime_writeable_weights": 0 },
  "MVAU_5": { "PE": 1, "SIMD": 4, "ram_style": "block", "resType": "lut", "mem_mode": "internal_decoupled", "runtime_writeable_weights": 0 },
  "MVAU_6": { "PE": 2, "SIMD": 4, "ram_style": "block", "resType": "dsp", "mem_mode": "internal_decoupled", "runtime_writeable_weights": 0 },
  "MVAU_7": { "PE": 1, "SIMD": 4, "ram_style": "block", "resType": "lut", "mem_mode": "internal_decoupled", "runtime_writeable_weights": 0 },
  "MVAU_8": { "PE": 1, "SIMD": 4, "ram_style": "block", "resType": "lut", "mem_mode": "internal_decoupled", "runtime_writeable_weights": 0 }
}
```

Dimension / divisibility verification (MH/MW are the actual LPYOLO 2W4A layer dims after FINN's streamlining):


| Node   | MH  | MW  | PE | MH%PE | SIMD | MW%SIMD |
| -------- | ----- | ----- | ---- | ------- | ------ | --------- |
| MVAU_0 | 8   | 72  | 4  | 0     | 3    | 0       |
| MVAU_1 | 16  | 72  | 4  | 0     | 4    | 0       |
| MVAU_2 | 32  | 144 | 4  | 0     | 4    | 0       |
| MVAU_3 | 56  | 288 | 2  | 0     | 4    | 0       |
| MVAU_4 | 104 | 504 | 2  | 0     | 4    | 0       |
| MVAU_5 | 208 | 936 | 1  | 0     | 4    | 0       |
| MVAU_6 | 56  | 208 | 2  | 0     | 4    | 0       |
| MVAU_7 | 104 | 504 | 2  | 0     | 4    | 0       |
| MVAU_8 | 18  | 936 | 3  | 0     | 4    | 0       |

All divisibility constraints satisfied.

---

## Part C - Run the FINN Build in Jupyter

All cells below run in the notebook created in step B1.

### C1. Cell 1: Verify Environment

```python
import os
print("FINN_XILINX_PATH:     ", os.environ.get("FINN_XILINX_PATH"))
print("FINN_XILINX_VERSION:  ", os.environ.get("FINN_XILINX_VERSION"))
print("XILINX_LOCAL_USER_DATA:", os.environ.get("XILINX_LOCAL_USER_DATA"))
print("cwd:                  ", os.getcwd())
```

Expected output:

```
FINN_XILINX_PATH:      /tools/Xilinx
FINN_XILINX_VERSION:   2022.2
XILINX_LOCAL_USER_DATA: no
cwd:                   /workspace/finn/notebooks/lpyolo
```

**If `XILINX_LOCAL_USER_DATA` is `None`:** you didn't source `~/finn-env.sh` before launching the container, or the `FINN_DOCKER_EXTRA` line is missing from it. Exit the container, fix, relaunch.

### C2. Cell 2: Define the Build

This is the full build. The critical quirks baked in:

- Custom `step_rename_empty_nodes` inserted after specialize: FINN v0.10.1 wipes node names during `step_specialize_layers`, and without names the generated HLS Tcl has empty `set_top` calls which fail.
- `auto_fifo_depths=False, split_large_fifos=False`: see B2 rationale.
- Step order: folding BEFORE specialize (so node names are still `MVAU_N`).

```python
import os
os.chdir("/home/svenons/finn/notebooks/lpyolo")

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_steps import *
from qonnx.core.modelwrapper import ModelWrapper

def step_rename_empty_nodes(model, cfg):
    for i, n in enumerate(model.graph.node):
        if not n.name or n.name.strip() == "":
            n.name = f"{n.op_type}_{i}"
    return model

cfg = build.DataflowBuildConfig(
    output_dir               = "output_full",
    folding_config_file      = "Pynq-Z2_folding_config.json",
    auto_fifo_depths         = False,
    split_large_fifos        = False,
    synth_clk_period_ns      = 10.0,
    board                    = "Pynq-Z2",
    shell_flow_type          = build_cfg.ShellFlowType.VIVADO_ZYNQ,
    save_intermediate_models = True,
    generate_outputs         = [
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.BITFILE,
        build_cfg.DataflowOutputType.PYNQ_DRIVER,
        build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
    ],
    steps = [
        step_qonnx_to_finn,
        step_tidy_up,
        step_streamline,
        step_convert_to_hw,
        step_create_dataflow_partition,
        step_target_fps_parallelization,
        step_apply_folding_config,
        step_specialize_layers,
        step_rename_empty_nodes,
        step_minimize_bit_width,
        step_generate_estimate_reports,
        step_hw_codegen,
        step_hw_ipgen,
        step_set_fifo_depths,
        step_create_stitched_ip,
        step_measure_rtlsim_performance,
        step_out_of_context_synthesis,
        step_synthesize_bitfile,
        step_make_pynq_driver,
        step_deployment_package,
    ]
)

build.build_dataflow_cfg("2w4a.onnx", cfg)
```

### C3. Run the Cell - What to Expect

Total runtime: **2–6 hours**. Progress prints per step:

```
Running step: step_qonnx_to_finn             [1/20]
Running step: step_tidy_up                   [2/20]
Running step: step_streamline                [3/20]
Running step: step_convert_to_hw             [4/20]
Running step: step_create_dataflow_partition [5/20]
Running step: step_target_fps_parallelization [6/20]
Running step: step_apply_folding_config      [7/20]
Running step: step_specialize_layers         [8/20]
Running step: step_rename_empty_nodes        [9/20]
Running step: step_minimize_bit_width        [10/20]
Running step: step_generate_estimate_reports [11/20]
Running step: step_hw_codegen                [12/20]
Running step: step_hw_ipgen                  [13/20]   ← LONGEST, 1–2 hours
Running step: step_set_fifo_depths           [14/20]
Running step: step_create_stitched_ip        [15/20]
Running step: step_measure_rtlsim_performance [16/20]
Running step: step_out_of_context_synthesis  [17/20]
Running step: step_synthesize_bitfile        [18/20]   ← Vivado impl, ~30 min
Running step: step_make_pynq_driver [19/20]
Running step: step_deployment_package [20/20]
Completed successfully
```

If a step fails, the error shows immediately — no need to wait for the rest. Common failures and fixes are in the troubleshooting table.

### C4. Verify the Output

```python
import os
deploy_dir = "output_full/deploy"
for f in ["finn-accel.bit", "finn-accel.hwh", "driver.py"]:
    path = f"{deploy_dir}/{f}"
    print(f"{path}: {'OK' if os.path.isfile(path) else 'MISSING'}")

# scale.npy may be one folder deeper
import subprocess
r = subprocess.run(["find", deploy_dir, "-name", "scale.npy"], capture_output=True, text=True)
print("scale.npy paths:", r.stdout.strip())
```

---

## Part D - Copy the Build Output

From WSL2:

```bash
scp -r ~/finn/notebooks/lpyolo/output_full/deploy/ xilinx@pynq:/home/xilinx/lpyolo/
```

---

## Part E - GPIO Access Alongside the FINN Overlay

Loading the FINN overlay unloads the base overlay, which means the board interfaces provided by `base.bit` stop working. For buttons, switches, and LEDs that is fixable by adding AXI GPIO IP to the FINN design. HDMI output is different: it is not just a GPIO pin bundle, but a full video pipeline from the base overlay, including the HDMI controller / video DMA path and its supporting clocks and constraints. The GPIO merge below does **not** recreate that pipeline, so HDMI out will not work from the final FINN+GPIO bitstream. Obvious in hindsight, slightly annoying in practice.

You cannot have both bitstreams loaded simultaneously on a Z2. Two workable patterns:

### E1. Swap overlays on demand

Load the base overlay for GPIO operations, then load the FINN overlay for inference, then swap back. Each overlay load takes ~1 second. Good for interactive/test use, bad for realtime apps that need both.

```python
from pynq import Overlay

def use_gpio():
    return Overlay("base.bit")          # restores buttons, LEDs, switches, HDMI

def use_fpga_inference():
    from pynq import Overlay
    return Overlay("/home/xilinx/lpyolo/finn-accel.bit")
```

### E2. Rebuild the FINN bitstream with GPIO merged in (Vivado)

To have GPIO (BTN0–3, SW0–1, LD0–3, RGB LEDs) work while the FINN accelerator is loaded, you need a single bitstream that contains both the FINN dataflow partition and the extra AXI GPIO blocks. There is no way around this on the Z2 — every user-facing button, switch, and LED is routed through the PL. This section walks through adding those GPIO blocks in Vivado.

This still does **not** restore HDMI. The PYNQ-Z2 HDMI output in `base.bit` depends on the base overlay's video subsystem, not only on board pin constraints. Adding that subsystem back would be a separate video-design integration task, and was not needed for Face Frenzy.

Total time: 60–90 minutes on top of the FINN build.

#### E2.1. Prerequisites: WSL2 locale and Vivado system libraries

Vivado 2022.2 on Ubuntu 22.04 needs a few packages that aren't there by default, or it fails to launch:

```bash
# Fix locale
sudo apt-get install -y locales
sudo locale-gen en_US.UTF-8
sudo update-locale LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

# libtinfo5 (removed from Ubuntu 22.04 main repos, fetch manually)
wget http://archive.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb
sudo dpkg -i libtinfo5_6.3-2ubuntu0.1_amd64.deb
sudo apt-get install -y libncurses5
```

WSL2's WSLg has OpenGL rendering issues that freeze Vivado on complex dialogs. Force software rendering before launching:

```bash
export LIBGL_ALWAYS_SOFTWARE=1
export _JAVA_OPTIONS="-Dsun.java2d.opengl=false"
```

#### E2.2. Install the PYNQ-Z2 board files

Vivado needs the TUL PYNQ-Z2 board definition for the pin constraints of buttons, switches, and LEDs:

```bash
cd ~
git clone https://github.com/Xilinx/XilinxBoardStore.git
sudo mkdir -p /tools/Xilinx/Vivado/2022.2/data/boards/board_files/
sudo cp -r ~/XilinxBoardStore/boards/TUL/pynq-z2 /tools/Xilinx/Vivado/2022.2/data/boards/board_files/
```

Verify:

```bash
ls /tools/Xilinx/Vivado/2022.2/data/boards/board_files/pynq-z2/
# Should show: A.0
```

Note the path is `TUL/pynq-z2`, not `Xilinx/pynq-z2` — the Z2 is a TUL product, not Xilinx.

#### E2.3. Open the FINN-generated Vivado project

FINN leaves a complete Vivado project behind from its synthesis step. Find it:

```bash
ls -d /home/svenons/finn_build/vivado_zynq_proj_*/
```

There will be one directory. Launch Vivado on its `.xpr` file:

```bash
source /tools/Xilinx/Vivado/2022.2/settings64.sh
vivado /home/svenons/finn_build/vivado_zynq_proj_XXXXXX/finn_zynq_link.xpr &
```

The GUI opens with the existing block design containing the Zynq PS, FINN dataflow partitions (StreamingDataflowPartition_0/1/2), DMA engines (idma0, odma0), and AXI infrastructure.

In **Tools → Settings → General → Project device**, confirm the board is set to **TUL PYNQ-Z2**. If not, select it and click OK.

#### E2.4. Add AXI GPIO IPs to the block design

Open the block design from the Flow Navigator: **IP INTEGRATOR → Open Block Design**.

You'll add four AXI GPIO IPs — one each for buttons, switches, LEDs, and RGB LEDs.

For each of the four, repeat these steps:

1. Right-click in empty canvas area → **Add IP**
2. Search for `AXI GPIO`, double-click to add it
3. Double-click the new `axi_gpio_0` block in the diagram to open its "Re-customize IP" dialog
4. On the **Board** tab, set **GPIO → Board Interface** to one of these (naming depends on the board file):

| Block to add | Board Interface value | Rename to |
|---|---|---|
| 1st | `btns 4bits` (check **Enable Interrupt** at the bottom) | `btns_gpio` |
| 2nd | `sws 2bits` | `switches_gpio` |
| 3rd | `leds 4bits` | `leds_gpio` |
| 4th | `rgb led` (exact label varies — whatever the board file uses for 6-bit RGB) | `rgbleds_gpio` |

5. Click **OK** to close the dialog
6. In the block diagram, right-click the block → **Rename...** → use the name from the table above → OK

After all four are added, click the green **"Run Connection Automation"** banner at the top of the Diagram view. In the dialog:

- Check the top-level "All Automation" checkbox (or manually expand each GPIO block and check both `S_AXI` and `GPIO` sub-items)
- Click **OK**

Vivado will wire each block's `S_AXI` through the existing AXI interconnect to the PS, and route each `GPIO` interface out to external pins using the board file's constraints. The AXI interconnect auto-expands to accommodate the new masters.

#### E2.5. Validate and save

- **Tools → Validate Design** (or F6) — must return 0 errors
- **File → Save Block Design** (Ctrl-S)

In the **Sources** tab (top-left), expand `top_wrapper` and find `top_i : top (top.bd)` underneath it. Right-click it → **Generate Output Products...**:

- Synthesis Options: **Global** (not Out of Context)
- Click **Generate** (~1 min)

Note: FINN ships a pre-generated HDL wrapper. Do not click "Create HDL Wrapper" — it exists already. But the shipped wrapper is out of date after you add new IPs, so Generate Output Products writes a *second* wrapper under `finn_zynq_link.gen/...`. You must swap the project's top file to this new wrapper in E2.6, otherwise synthesis runs against the stale one.

#### E2.6. Replace the stale wrapper with the regenerated one

This is the single most common way E2 silently fails. FINN originally ships a static `top_wrapper.v` under `finn_zynq_link.srcs/sources_1/imports/hdl/` that matches the block design *before* GPIO was added. Generate Output Products writes a *new* wrapper under `finn_zynq_link.gen/sources_1/bd/top/hdl/` with the button/LED ports, but the project still lists the old one as a source. Synthesis then runs against the old wrapper and produces a bitstream where BTN0–3 / SW0–1 / LD0–3 / RGB LEDs are internally connected but never routed to the physical FPGA pins.

Symptom: `ol.btns_gpio.read(0)` always returns `0` regardless of which button you press, and `ol.leds_gpio.write(0, 0xF)` does not turn any LED on, even though `ip_dict` shows `btns_gpio` / `leds_gpio` present.

Confirm before fixing — look at both wrappers:

````bash
find /home/svenons/finn_build/vivado_zynq_proj_*/ -name top_wrapper.v
grep -E "btn|sws|leds|rgb" \
    /home/svenons/finn_build/vivado_zynq_proj_*/finn_zynq_link.srcs/sources_1/imports/hdl/top_wrapper.v
grep -E "btn|sws|leds|rgb" \
    /home/svenons/finn_build/vivado_zynq_proj_*/finn_zynq_link.gen/sources_1/bd/top/hdl/top_wrapper.v | head
````

If the `imports` file returns nothing and the `gen` file lists `btns_4bits_tri_i`, `leds_4bits_tri_io`, etc., the wrapper is stale. Fix via the Tcl console in Vivado:

````tcl
remove_files [get_files */imports/hdl/top_wrapper.v]
add_files    -norecurse [glob */finn_zynq_link.gen/sources_1/bd/top/hdl/top_wrapper.v]
set_property top top_wrapper [current_fileset]
update_compile_order -fileset sources_1
````

Then proceed to the re-synthesis in E2.7.

#### E2.7. Re-synthesize and generate bitstream

The FINN project already has the synthesis/implementation strategies we want. In the Tcl console (bottom of window) paste:

````tcl
reset_run synth_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 4
````

Wait ~45 minutes total.

Once it finishes, before exporting hardware, run a sanity check:

````bash
cat /home/svenons/finn_build/vivado_zynq_proj_*/finn_zynq_link.runs/impl_1/top_wrapper_io_placed.rpt \
  | grep -E "D19|D20|L19|L20"
````

The "Signal Name" column (2nd one) on each of those four rows must be non-empty — it should read something like `btns_4bits_tri_i[0]`. If the column is still blank, the wrapper is still stale and synthesis used the wrong one; redo E2.6.

**If implementation fails** with "write_bitstream out-of-date" errors:

````tcl
reset_run impl_1
launch_runs impl_1 -to_step write_bitstream -jobs 4
````

**If synthesis fails with "black box" errors** about `StreamingDataflowPartition_X_0`, it means the output products regenerated but the dataflow partition OOC checkpoints weren't re-linked. Fix:

````tcl
reset_runs [get_runs -filter {IS_SYNTHESIS}]
launch_runs synth_1 -jobs 4
````

When everything succeeds, Vivado shows "Bitstream Generation successfully completed." Click **Cancel** on that dialog.



#### E2.8. Export the hardware

In the Tcl console:

```tcl
write_hw_platform -fixed -include_bit -force -file /home/svenons/top_wrapper.xsa
```

Then in a WSL2 terminal (outside Vivado):

```bash
cd /tmp
mkdir -p xsa_extract && cd xsa_extract
unzip -o /home/svenons/top_wrapper.xsa
ls *.bit *.hwh
```

You'll see `top_wrapper.bit` and `top.hwh` (ignore `top_smartconnect_0_0.hwh` — that's a sub-IP export).

#### E2.9. Copy the updated bitstream files

Rename to match what the FINN driver expects, copy over:

```bash
cp /tmp/xsa_extract/top_wrapper.bit ~/finn-accel.bit
cp /tmp/xsa_extract/top.hwh       ~/finn-accel.hwh
scp ~/finn-accel.bit ~/finn-accel.hwh xilinx@pynq:/home/xilinx/lpyolo/deploy/bitfile/
```

For the Face Frenzy repo, copy them into `fpga/` instead:

```bash
scp ~/finn-accel.bit ~/finn-accel.hwh xilinx@pynq:/home/xilinx/face-frenzy/fpga/
```

Also copy `scale.npy` from the FINN deploy package if it exists:

```bash
find ~/finn/notebooks/lpyolo/output_full/deploy -name scale.npy
scp /path/to/scale.npy xilinx@pynq:/home/xilinx/face-frenzy/fpga/scale.npy
```

#### E2.10. Use the restored GPIO from Python

Once the new bitstream is loaded with `Overlay("finn-accel.bit")`, the GPIO IPs are accessible as named attributes on the overlay object:

```python
from pynq import Overlay
ol = Overlay("/home/xilinx/lpyolo/deploy/bitfile/finn-accel.bit")

# Buttons: read the 4-bit value; each bit = one BTNx
btns = ol.btns_gpio.read(0)            # bit 0 = BTN0, bit 3 = BTN3
btn0_pressed = bool(btns & 0x1)

# Switches: 2-bit value
sws = ol.switches_gpio.read(0)
sw0 = bool(sws & 0x1)
sw1 = bool(sws & 0x2)

# LEDs: write a 4-bit value, bit N = LDN
ol.leds_gpio.write(0, 0b1010)          # LD1 and LD3 on

# RGB LEDs: 6-bit value; bit order depends on board file —
# typically [LD4_B, LD4_G, LD4_R, LD5_B, LD5_G, LD5_R]
ol.rgbleds_gpio.write(0, 0b000100)     # LD4 green

# The FINN accelerator is still there. In Face Frenzy, use the repo helper.
from FaceDetector import load_finn_accelerator
accel = load_finn_accelerator("fpga/finn-accel.bit", "fpga")
# ...use accel.execute(...) as before
```

The FINN accelerator and the GPIO now coexist in a single bitstream. Loading the overlay configures both, and you can use them independently in the same Python script.
---

## Troubleshooting


| Symptom                                                                                | Cause                                                                | Fix                                                                                                    |
| ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `ERROR [Synth 8-5834] Design needs N RAMB18 which is more than device capacity of 280` | Folding didn't apply — node names mismatched                        | Confirm config in B2 is exact; verify step order has folding BEFORE specialize                         |
| `AssertionError: mem_mode = decoupled not in {...internal_decoupled...}`               | Old v0.7 mem_mode value                                              | Use`"internal_decoupled"` in the folding config                                                        |
| `set_top: Argument 'name' requires a value!` in HLS log                                | Empty node names after specialize                                    | `step_rename_empty_nodes` in the steps list                                                            |
| `ERROR [Common 17-685] Unable to load Tcl app xilinx::xsim`                            | Vivado tclstore corruption from parallel HLS                         | `export FINN_DOCKER_EXTRA=" -e XILINX_LOCAL_USER_DATA=no "` in `finn-env.sh`                           |
| `ERROR [DRC UTLZ-1] LUT as Logic over-utilized`                                        | Folding too aggressive, design too large                             | Lower PE on the biggest MVAU layers (usually MVAU_5 which is MH=208)                                   |
| `ERROR [DRC UTLZ-1] RAMB18/RAMB36 over-utilized` with 1000+ BRAMs needed               | `auto_fifo_depths=True` sized FIFOs to tensor size                   | Set`auto_fifo_depths=False` and use `"Defaults": {"depth": [32, ["StreamingFIFO"]]}` in folding config |
| `FileNotFoundError` on makedirs when dir exists                                        | Ghost-inode bug after`rm -rf finn_build` while container was running | Exit and relaunch container                                                                            |
| FINN build fails at`step_hw_ipgen` on multiple nodes                                   | Vivado not mounted in container                                      | Verify`FINN_XILINX_PATH=/tools/Xilinx` before `./run-docker.sh`                                        |
| `Value '524288' is out of range for FIFO depth`                                        | auto_fifo_depths sized a FIFO beyond Vivado's 32768 cap              | Disable`auto_fifo_depths` or enable `split_large_fifos=True`                                           |
| Buttons / LEDs / switches blank after loading FINN overlay                            | PL overlay replaced the base overlay                                 | See Part E                                                                                             |
| HDMI out blank after loading the FINN+GPIO overlay                                    | HDMI pipeline from `base.bit` was not recreated                       | Use `base.bit` for HDMI, or integrate a full video subsystem separately                                |
| `RuntimeError: Could not open device with index '0'` on PYNQ                          | Overlay/XRT access attempted without root                             | Use `sudo -i`, then source `pynq_venv.sh` and `xrt_setup.sh` again                                     |
| `ModuleNotFoundError: No module named 'bitstring'`                                    | Generated FINN packing dependency missing from root PYNQ env           | Install `bitstring` inside the sourced root PYNQ environment                                           |
| `pip install opencv-python` tries to compile for hours                                | PYNQ-Z2 is armv7 and pip wheels are unreliable                         | Install `python3-opencv` with apt instead                                                             |
| `accel.execute(...)` appears stuck before DMA starts                                  | Generic FINN UINT4 packing path is too slow on the board               | Use the Face Frenzy `fpga/driver_base.py` fast UINT4 pack patch                                        |
| FPGA execute passes but face count stays at zero                                      | App-side decoder scale/layout/channel order is not calibrated          | Run `scripts/fpga-frame-debug.py` and tune `FACE_FRENZY_OUTPUT_SCALE` / layout env vars                |
| WSL2 out of disk during Vivado install                                                 | Default vhdx is 256GB but partitioned smaller                        | Follow A2 to resize                                                                                    |
