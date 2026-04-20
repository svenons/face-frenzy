import argparse
import os
import sys
import time

import cv2
import numpy as np


REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

import FaceDetector as face_detector_module
from FaceDetector import FaceDetector, _sigmoid, load_finn_accelerator


def main():
    parser = argparse.ArgumentParser(description="Run one camera frame through the FINN face detector.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--bitfile", default="fpga/finn-accel.bit")
    parser.add_argument("--driver-dir", default="fpga")
    parser.add_argument("--out", default="fpga-frame-debug.jpg")
    parser.add_argument("--save-raw", metavar="PATH",
                        help="save the raw INT8 output tensor as a .npy file")
    parser.add_argument("--cell", nargs=2, type=int, metavar=("ROW", "COL"),
                        help="print the full 18-channel vector at grid cell (ROW, COL)")
    parser.add_argument("--sweep", action="store_true",
                        help="sweep all 18 score indices and print raw/kept/max for each")
    args = parser.parse_args()

    os.environ.setdefault("FACE_FRENZY_DEBUG_DETECTOR", "1")

    print("[frame-debug] loading overlay")
    start = time.time()
    accel = load_finn_accelerator(args.bitfile, args.driver_dir)
    print("[frame-debug] overlay_load_s:", f"{time.time() - start:.3f}")
    face_detector_module.set_accelerator(accel)
    detector = FaceDetector()
    scale_path = getattr(face_detector_module, "_finn_output_scale_path", None)
    print("[frame-debug] effective_output_scale:", detector._output_scale)
    print("[frame-debug] scale_path:", scale_path if scale_path else "not found")

    cam = cv2.VideoCapture(args.camera)
    if not cam.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    frame = None
    for _ in range(8):
        ok, frame = cam.read()
        if ok and frame is not None:
            break
        time.sleep(0.1)
    cam.release()

    if frame is None:
        raise RuntimeError("Camera opened but did not return a frame")

    print("[frame-debug] frame_shape:", frame.shape)
    print("[frame-debug] frame_mean:", f"{float(frame.mean()):.2f}")

    inp = detector._preprocess_for_finn(frame)
    print("[frame-debug] input_min_max:", int(inp.min()), int(inp.max()))

    start = time.time()
    output = np.asarray(accel.execute(inp), dtype=np.float32)
    print("[frame-debug] execute_s:", f"{time.time() - start:.3f}")
    print("[frame-debug] output_shape:", output.shape)
    print("[frame-debug] output_min_max_mean:", float(output.min()), float(output.max()), float(output.mean()))

    if args.save_raw:
        np.save(args.save_raw, output)
        print("[frame-debug] raw output saved to:", args.save_raw)

    raw = output[0] if output.shape[0] == 1 else output
    n_channels = raw.shape[-1]

    # Print full channel vector at a specific grid cell
    if args.cell:
        row, col = args.cell
        grid_h, grid_w = raw.shape[:2]
        if 0 <= row < grid_h and 0 <= col < grid_w:
            vec = raw[row, col, :]
            print(f"[frame-debug] cell ({row},{col}) raw INT8 values:")
            for i, v in enumerate(vec):
                anchor = i // 6
                field = ["tx", "ty", "tw", "th", "conf", "cls"][i % 6]
                scale = detector._output_scale
                fval = float(v) / scale
                if field in ("tx", "ty", "conf", "cls"):
                    decoded = f"sigmoid={_sigmoid(fval):.4f}"
                else:
                    decoded = f"exp(clip)={np.exp(np.clip(fval,-6,6)):.3f}"
                print(f"  c{i:2d} (anchor{anchor} {field}): int8={int(v):5d}  float={fval:7.3f}  {decoded}")
        else:
            print(f"[frame-debug] --cell {row} {col} is out of range (grid is {raw.shape[0]}x{raw.shape[1]})")
    else:
        # Always print center cell for free
        cr, cc = raw.shape[0] // 2, raw.shape[1] // 2
        vec = raw[cr, cc, :]
        scale = detector._output_scale
        conf_vals = [
            _sigmoid(float(vec[4]) / scale) * _sigmoid(float(vec[5]) / scale),
            _sigmoid(float(vec[10]) / scale) * _sigmoid(float(vec[11]) / scale),
            _sigmoid(float(vec[16]) / scale) * _sigmoid(float(vec[17]) / scale),
        ]
        print(f"[frame-debug] center cell ({cr},{cc}) conf (anchor0,1,2): "
              f"{conf_vals[0]:.4f} {conf_vals[1]:.4f} {conf_vals[2]:.4f}  "
              f"raw_c4={int(vec[4])} raw_c10={int(vec[10])} raw_c16={int(vec[16])}")

    # Sweep all score channels to find which has the most discriminative signal.
    if args.sweep:
        print("[frame-debug] sweeping all 18 raw channels as possible score channels:")
        scale = detector._output_scale
        for ch in range(n_channels):
            vals = raw[:, :, ch].flatten()
            scores = _sigmoid(vals / scale)
            above = int((scores >= detector.CONF_THRESHOLD).sum())
            print(
                f"  c{ch:02d}: raw_min={int(vals.min()):4d} raw_max={int(vals.max()):4d} "
                f"raw_mean={float(vals.mean()):7.2f} raw_std={float(vals.std()):7.2f} "
                f"sig_min={scores.min():.4f} sig_max={scores.max():.4f} "
                f"sig_mean={scores.mean():.4f} above_thresh={above}/{len(scores)}"
            )

        print("[frame-debug] anchor-major confidence channels:")
        for ch in (4, 10, 16):
            vals = raw[:, :, ch].flatten()
            scores = _sigmoid(vals / scale)
            print(
                f"  c{ch}: raw_unique_sample={np.unique(vals.astype(np.int16))[:10].tolist()} "
                f"sigmoid_range=({scores.min():.4f}, {scores.max():.4f})"
            )

    boxes = detector._decode_yolo(output, frame.shape[1], frame.shape[0])
    print("[frame-debug] boxes:", boxes)
    print("[frame-debug] count:", len(boxes))

    annotated = frame.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        annotated,
        f"Faces: {len(boxes)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )
    cv2.imwrite(args.out, annotated)
    print("[frame-debug] wrote:", args.out)


if __name__ == "__main__":
    main()
