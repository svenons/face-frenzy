import importlib
import os
import sys
import time

import cv2
import numpy as np


_shared_accel = None
_finn_output_scale = None
_finn_output_scale_path = None


def set_accelerator(accel):
    """Share the already-loaded FINN overlay with the detector."""
    global _shared_accel
    _shared_accel = accel


class FaceDetector:
    CONF_THRESHOLD = float(os.environ.get("FACE_FRENZY_CONF_THRESHOLD", "0.38"))
    IOU_THRESHOLD = 0.4
    # LPYOLO anchors for 13×13 grid (stride 32) at 416px input.
    # From YOLOv3-tiny-Face full anchor list, mask indices 3,4,5 (largest three):
    # 81×82, 135×169, 344×319
    ANCHORS = np.array([[81, 82], [135, 169], [344, 319]], dtype=np.float32)

    def __init__(self):
        self.backend = os.environ.get("FACE_FRENZY_BACKEND", "fpga").lower()
        self._max_boxes = int(os.environ.get("FACE_FRENZY_MAX_BOXES", "8"))

        if self.backend == "cpu":
            self._init_cpu()
            return

        # FPGA path
        self._accel = None
        self._input_shape = (1, 416, 416, 8)
        self._output_shape = (1, 13, 13, 18)
        self._last_empty_log = 0
        self._output_scale = _resolve_output_scale()
        self._channel_order = os.environ.get("FACE_FRENZY_CHANNEL_ORDER", "rgb").lower()
        self._decode_layout = os.environ.get("FACE_FRENZY_YOLO_LAYOUT", "auto").lower()
        self._score_indices = _parse_int_list(os.environ.get("FACE_FRENZY_SCORE_INDICES", "4,5"))
        self._score_mode = os.environ.get("FACE_FRENZY_SCORE_MODE", "product").lower()
        self._min_box_px = int(os.environ.get("FACE_FRENZY_MIN_BOX_PX", "18"))
        self._max_box_width_ratio = float(os.environ.get("FACE_FRENZY_MAX_BOX_WIDTH_RATIO", "0.85"))
        self._max_box_height_ratio = float(os.environ.get("FACE_FRENZY_MAX_BOX_HEIGHT_RATIO", "0.85"))
        self._min_box_area_ratio = float(os.environ.get("FACE_FRENZY_MIN_BOX_AREA_RATIO", "0.001"))
        self._max_box_area_ratio = float(os.environ.get("FACE_FRENZY_MAX_BOX_AREA_RATIO", "0.55"))
        self._min_box_aspect = float(os.environ.get("FACE_FRENZY_MIN_BOX_ASPECT", "0.45"))
        self._max_box_aspect = float(os.environ.get("FACE_FRENZY_MAX_BOX_ASPECT", "1.8"))
        self._debug_detector = os.environ.get("FACE_FRENZY_DEBUG_DETECTOR", "0") == "1"

        if _shared_accel is None:
            raise RuntimeError("FINN accelerator has not been loaded")

        self._accel = _shared_accel
        self._input_shape = tuple(self._accel.ishape_normal())
        self._output_shape = tuple(self._accel.oshape_normal())
        print(
            "[FaceDetector] FPGA backend active "
            f"input_shape={self._input_shape} output_shape={self._output_shape} "
            f"output_scale={self._output_scale} channel_order={self._channel_order} "
            f"layout={self._decode_layout} score_indices={self._score_indices} "
            f"score_mode={self._score_mode} max_boxes={self._max_boxes}"
        )

    def _init_cpu(self):
        cascade_path = os.environ.get("FACE_FRENZY_CASCADE_PATH") or _find_cascade_file()
        if cascade_path is None:
            raise RuntimeError(
                "Haar cascade XML not found for CPU backend. "
                "Install python3-opencv (apt) or set FACE_FRENZY_CASCADE_PATH."
            )
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
        self._cpu_scale = float(os.environ.get("FACE_FRENZY_CPU_SCALE", "0.5"))
        print(f"[FaceDetector] CPU backend active cascade={cascade_path} cpu_scale={self._cpu_scale}")

    def detect_faces_with_boxes(self, frame):
        if frame is None:
            return [], time.time()

        if self.backend == "cpu":
            boxes = self._detect_cpu(frame)
        else:
            boxes = self._detect_fpga(frame)
        return boxes, time.time()

    def _detect_cpu(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        scale = self._cpu_scale
        if scale != 1.0:
            h, w = frame_bgr.shape[:2]
            small = cv2.resize(gray, (int(w * scale), int(h * scale)))
        else:
            small = gray

        dets = self._cascade.detectMultiScale(
            small,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(int(24 * scale), int(24 * scale)),
        )

        if not len(dets):
            return []
        inv = 1.0 / scale
        boxes = []
        for x, y, w, h in dets[: self._max_boxes]:
            boxes.append((int(x * inv), int(y * inv), int(w * inv), int(h * inv)))
        return boxes

    def _detect_fpga(self, frame_bgr):
        inp = self._preprocess_for_finn(frame_bgr)
        out = self._accel.execute(inp)
        return self._decode_yolo(np.asarray(out), frame_bgr.shape[1], frame_bgr.shape[0])

    def _preprocess_for_finn(self, frame_bgr):
        if len(self._input_shape) != 4:
            raise RuntimeError(f"Unexpected FINN input shape {self._input_shape}")

        _, height, width, channels = self._input_shape
        resized = cv2.resize(frame_bgr, (width, height))
        if self._channel_order == "bgr":
            network_image = resized
        elif self._channel_order == "gray":
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            network_image = np.repeat(gray[:, :, None], 3, axis=2)
        else:
            network_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # The generated driver reports UINT4 input. Quantize RGB to 0..15 and
        # zero-pad any extra channels expected by the synthesized network.
        quantized = (network_image.astype(np.float32) / 255.0 * 15.0).round().astype(np.uint8)
        inp = np.zeros(self._input_shape, dtype=np.uint8)
        copy_channels = min(quantized.shape[2], channels)
        inp[0, :, :, :copy_channels] = quantized[:, :, :copy_channels]
        return inp

    def _decode_yolo(self, output, w_orig, h_orig):
        if len(self._output_shape) != 4:
            print(f"[FaceDetector] unexpected output shape {output.shape}")
            return []

        _, grid_h, grid_w, output_channels = self._output_shape
        anchors = len(self.ANCHORS)
        values_per_anchor = output_channels // anchors
        if output_channels % anchors != 0 or values_per_anchor < 5:
            print(f"[FaceDetector] unsupported YOLO output shape {output.shape}")
            return []

        output = np.asarray(output, dtype=np.float32)
        if output.shape[0] == 1:
            output = output[0]
        if output.shape != (grid_h, grid_w, output_channels):
            print(f"[FaceDetector] unexpected output shape {output.shape}")
            return []

        candidates = []
        for layout in self._candidate_layouts():
            for score_index in self._score_indices:
                if score_index >= values_per_anchor:
                    continue
                dets, raw_boxes, max_conf, top = self._decode_yolo_layout(
                    output,
                    layout,
                    score_index,
                    values_per_anchor,
                    grid_h,
                    grid_w,
                    w_orig,
                    h_orig,
                )
                candidates.append(
                    {
                        "layout": layout,
                        "score_index": score_index,
                        "boxes": _nms(dets, self.IOU_THRESHOLD),
                        "raw_boxes": raw_boxes,
                        "max_conf": max_conf,
                        "top": top,
                    }
                )

        if not candidates:
            print(f"[FaceDetector] no valid YOLO decode candidates for output shape {output.shape}")
            return []

        best = max(candidates, key=lambda item: (len(item["boxes"]), item["max_conf"]))
        if best["boxes"]:
            boxes = best["boxes"][: self._max_boxes]
            if self._debug_detector:
                print(
                    "[FaceDetector] selected "
                    f"layout={best['layout']} score_index={best['score_index']} "
                    f"boxes={len(boxes)} raw_boxes={best['raw_boxes']} max_conf={best['max_conf']:.3f}"
                )
            return boxes

        if time.time() - self._last_empty_log > 5:
            self._last_empty_log = time.time()
            diag = ", ".join(
                f"{item['layout']}:score{item['score_index']} raw={item['raw_boxes']} "
                f"kept={len(item['boxes'])} max={item['max_conf']:.3f} top={item['top']}"
                for item in sorted(candidates, key=lambda item: item["max_conf"], reverse=True)[:4]
            )
            print(
                "[FaceDetector] FPGA produced 0 boxes; "
                f"output_shape={output.shape}, min={output.min()}, max={output.max()}, "
                f"threshold={self.CONF_THRESHOLD}, output_scale={self._output_scale}, "
                f"candidates=[{diag}], channels={_channel_summary(output)}"
            )

        return []

    def _candidate_layouts(self):
        if self._decode_layout in ("anchor", "anchor-major", "anchormajor"):
            return ["anchor-major"]
        if self._decode_layout in ("value", "value-major", "valuemajor"):
            return ["value-major"]
        return ["anchor-major", "value-major"]

    def _decode_yolo_layout(
        self,
        output,
        layout,
        score_index,
        values_per_anchor,
        grid_h,
        grid_w,
        w_orig,
        h_orig,
    ):
        dets = []
        raw_boxes = 0
        max_conf = 0.0
        top = None
        for cy in range(grid_h):
            for cx in range(grid_w):
                for anchor_index in range(len(self.ANCHORS)):
                    raw = self._raw_anchor_values(output, layout, anchor_index, values_per_anchor, cy, cx)
                    scaled = raw / self._output_scale
                    conf = self._confidence_from_values(scaled, score_index)
                    max_conf = max(max_conf, conf)
                    if top is None or conf > top["conf"]:
                        top = {
                            "conf": round(conf, 3),
                            "cell": (cy, cx),
                            "anchor": anchor_index,
                            "raw": [int(x) for x in raw[:values_per_anchor]],
                        }
                    if conf < self.CONF_THRESHOLD:
                        continue

                    # LPYOLO uses HardTanh activations instead of sigmoid/exp.
                    # tx,ty in [-0.5,0.5] → add 0.5 to get cell offset [0,1].
                    # tw,th in [-1,1] → (x+1)² gives scale factor [0,4].
                    bx = (scaled[0] + 0.5 + cx) / grid_w
                    by = (scaled[1] + 0.5 + cy) / grid_h
                    bw = max(0.0, (np.clip(scaled[2], -1.0, 1.0) + 1.0) ** 2 * self.ANCHORS[anchor_index, 0]) / 416
                    bh = max(0.0, (np.clip(scaled[3], -1.0, 1.0) + 1.0) ** 2 * self.ANCHORS[anchor_index, 1]) / 416

                    x = int((bx - bw / 2) * w_orig)
                    y = int((by - bh / 2) * h_orig)
                    w = int(bw * w_orig)
                    h = int(bh * h_orig)
                    if w <= 0 or h <= 0:
                        continue
                    if x >= w_orig or y >= h_orig or x + w <= 0 or y + h <= 0:
                        continue
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, w_orig - x)
                    h = min(h, h_orig - y)
                    if w > 0 and h > 0:
                        raw_boxes += 1
                    if self._is_plausible_face_box(w, h, w_orig, h_orig):
                        dets.append((x, y, w, h, conf))

        return dets, raw_boxes, max_conf, top

    def _raw_anchor_values(self, output, layout, anchor_index, values_per_anchor, cy, cx):
        if layout == "value-major":
            channels = [value_index * len(self.ANCHORS) + anchor_index for value_index in range(values_per_anchor)]
            return output[cy, cx, channels]
        start = anchor_index * values_per_anchor
        return output[cy, cx, start : start + values_per_anchor]

    def _confidence_from_values(self, scaled, score_index):
        # LPYOLO uses HardTanh [-1,1] activations. Map to [0,1] via (x+1)/2.
        primary = float(np.clip(scaled[score_index], -1.0, 1.0) + 1.0) / 2.0
        if self._score_mode == "single" or len(scaled) < 6 or score_index not in (4, 5):
            return primary

        partner_index = 5 if score_index == 4 else 4
        partner = float(np.clip(scaled[partner_index], -1.0, 1.0) + 1.0) / 2.0
        return primary * partner

    def _is_plausible_face_box(self, width, height, frame_width, frame_height):
        if width < self._min_box_px or height < self._min_box_px:
            return False
        if width > frame_width * self._max_box_width_ratio:
            return False
        if height > frame_height * self._max_box_height_ratio:
            return False

        area_ratio = (width * height) / float(frame_width * frame_height)
        if area_ratio < self._min_box_area_ratio or area_ratio > self._max_box_area_ratio:
            return False

        aspect = width / float(height)
        return self._min_box_aspect <= aspect <= self._max_box_aspect

def load_finn_accelerator(bitfile, driver_dir):
    global _finn_output_scale, _finn_output_scale_path

    try:
        import bitstring  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing Python dependency 'bitstring'. On the PYNQ-Z2, run "
            "'cd /home/xilinx/face-frenzy && bash scripts/setup-pynq.sh' "
            "or install it inside the root PYNQ environment with "
            "'sudo bash -lc \"cd /home/xilinx/face-frenzy && source "
            "/etc/profile.d/pynq_venv.sh && source /etc/profile.d/xrt_setup.sh "
            "&& python3 -m pip install -r requirements.txt\"'."
        ) from exc

    if not os.path.exists(bitfile):
        raise FileNotFoundError(f"FINN bitstream not found: {bitfile}")

    sys.path.insert(0, driver_dir)
    try:
        driver = importlib.import_module("driver")
        driver_base = importlib.import_module("driver_base")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Could not import the FINN driver from {driver_dir}: {exc}. "
            "Verify fpga/driver.py, fpga/driver_base.py, and the vendored "
            "finn/qonnx helper packages are present."
        ) from exc

    runtime_weight_dir = _runtime_weight_dir_or_missing(driver_dir)
    _finn_output_scale, _finn_output_scale_path = _load_output_scale(driver_dir)
    try:
        return driver_base.FINNExampleOverlay(
            bitfile_name=bitfile,
            platform="zynq-iodma",
            io_shape_dict=driver.io_shape_dict,
            runtime_weight_dir=runtime_weight_dir,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load FINN overlay '{bitfile}': {exc}. On PYNQ-Z2 this "
            "usually means the PYNQ/XRT environment is not sourced, the process "
            "does not have enough privileges for overlay/XRT access, or the "
            ".bit/.hwh files do not match."
        ) from exc


def _runtime_weight_dir_or_missing(driver_dir):
    runtime_weight_dir = os.path.join(driver_dir, "runtime_weights")
    if not os.path.isdir(runtime_weight_dir):
        return runtime_weight_dir

    weight_files = [
        name
        for name in os.listdir(runtime_weight_dir)
        if name.endswith(".dat") or name.endswith(".npy")
    ]
    if weight_files:
        return runtime_weight_dir

    # FINN's generated driver calls execute_on_buffers() at the end of
    # load_runtime_weights() whenever runtime_weight_dir exists, even when it
    # contains no weights. With this design that early flush can hang the IODMA
    # path before the app has even started. Point at a missing directory when no
    # runtime weights are present so the generated driver skips that flush.
    return os.path.join(driver_dir, "__no_runtime_weights__")


def _resolve_output_scale():
    env_scale = os.environ.get("FACE_FRENZY_OUTPUT_SCALE")
    if env_scale:
        return float(env_scale)
    if _finn_output_scale is not None:
        return float(_finn_output_scale)
    return 127.0


def _load_output_scale(driver_dir):
    scale_path = _find_scale_file(driver_dir)
    if scale_path is None:
        print("[FaceDetector] scale.npy not found; using default FACE_FRENZY_OUTPUT_SCALE=64")
        return None, None

    try:
        raw_scale = float(np.asarray(np.load(scale_path)).reshape(-1)[0])
        # FINN stores a multiplicative scale (float = int8 * raw_scale).
        # The rest of the decoder divides by output_scale (float = int8 / output_scale),
        # so convert: output_scale = 1 / raw_scale when raw_scale looks multiplicative.
        if 0 < raw_scale < 1.0:
            output_scale = 1.0 / raw_scale
        else:
            output_scale = raw_scale
        print(f"[FaceDetector] loaded scale.npy raw={raw_scale:g} → output_scale={output_scale:g} from {scale_path}")
        return output_scale, scale_path
    except Exception as exc:
        print(f"[FaceDetector] could not read scale.npy at {scale_path}: {exc}; using default scale 64")
        return None, scale_path


def _find_cascade_file():
    candidates = [
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"),
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_default.xml",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _find_scale_file(driver_dir):
    candidates = [
        os.path.join(driver_dir, "scale.npy"),
        os.path.join(driver_dir, "runtime_weights", "scale.npy"),
        os.path.join(os.getcwd(), "scale.npy"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path

    if os.path.isdir(driver_dir):
        for root, _, files in os.walk(driver_dir):
            if "scale.npy" in files:
                return os.path.join(root, "scale.npy")
    return None


def _parse_int_list(value):
    parsed = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed.append(int(item))
    return parsed or [4]


def _channel_summary(output):
    flat = output.reshape(-1, output.shape[-1])
    parts = []
    for channel in range(output.shape[-1]):
        values = flat[:, channel]
        parts.append(
            f"c{channel}:min={int(values.min())}/max={int(values.max())}/mean={float(values.mean()):.1f}"
        )
    return "; ".join(parts)


class FaceCounter:
    """Converts a list of raw FPGA boxes into a face count.

    Implementations can swap freely without touching the game loop —
    the only contract is `count(boxes, max_faces) -> int`.
    """

    def count(self, boxes, max_faces=4):
        raise NotImplementedError


class BoxCountFaceCounter(FaceCounter):
    """Divides the raw box count by an empirical boxes-per-face constant.

    LPYOLO produces ~3-4 overlapping boxes per face on this setup.
    Calibration points collected on the PYNQ-Z2:
      0 people → 0 boxes, 1 person → 4 boxes, 2 people → 7 boxes.
    """

    def __init__(self, boxes_per_face=4.0):
        self.boxes_per_face = float(boxes_per_face)

    def count(self, boxes, max_faces=4):
        n = len(boxes)
        if n == 0:
            return 0
        return min(max(1, round(n / self.boxes_per_face)), max_faces)


_default_counter = BoxCountFaceCounter(
    boxes_per_face=float(os.environ.get("FACE_FRENZY_BOXES_PER_FACE", "4.0"))
)


def count_faces(boxes, frame_width=640, max_faces=4):
    """Module-level convenience wrapper around the default FaceCounter."""
    return _default_counter.count(boxes, max_faces=max_faces)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _nms(dets, iou_thresh):
    if not dets:
        return []

    dets = sorted(dets, key=_det_sort_value, reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [det for det in dets if _iou(best, det) < iou_thresh]
    return [det[:4] for det in keep]


def _det_sort_value(det):
    if len(det) >= 5:
        return det[4]
    return det[2] * det[3]


def _iou(a, b):
    ax1, ay1, aw, ah = a[:4]
    bx1, by1, bw, bh = b[:4]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0
