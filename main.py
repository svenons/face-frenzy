import logging
import os
import sys
import threading
import time

import cv2
import numpy as np

import FaceDetector as face_detector_module
from CameraManager import CameraManager
from FaceDetector import BoxCountFaceCounter, FaceDetector, load_finn_accelerator
from GameController import GameController
from GameStateExporter import GameStateExporter
from IOHandler import IOHandler
from WebServer import WebServer


APP_DIR = os.path.dirname(os.path.abspath(__file__))
FPGA_DIR = os.path.join(APP_DIR, "fpga")
BITFILE = os.path.join(FPGA_DIR, "finn-accel.bit")
LOG_FILE = os.path.join(APP_DIR, "face-frenzy.log")


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE)],
    )
    return logging.getLogger("face-frenzy")


log = configure_logging()
DETECTION_INTERVAL_S = float(os.environ.get("FACE_FRENZY_DETECTION_INTERVAL_S", "0.3"))
BOXES_PER_FACE = float(os.environ.get("FACE_FRENZY_BOXES_PER_FACE", "4.0"))


class NullDisplay:
    def __init__(self):
        self.overlay_text = ""

    def set_overlay_text(self, text):
        self.overlay_text = text

    def clear_overlay_text(self):
        self.overlay_text = ""

    def show_frame(self, frame_bgr, boxes=None, face_count=None):
        pass

    def white_screen(self):
        pass

    def stop(self):
        pass


class _NullIOHandler:
    def read_buttons(self):
        return {"btn0": 0, "btn1": 0, "btn2": 0, "btn3": 0}

    def set_led_countdown(self, remaining, total):
        pass

    def show_player_select(self, selected_players, flash_all):
        pass

    def show_result(self, success):
        pass

    def clear_leds(self):
        pass


def wait_for_camera(timeout_s=None):
    start = time.time()
    while True:
        try:
            camera = CameraManager()
            frame = camera.get_frame()
            if frame is not None:
                log.info("Camera ready")
                return camera
            camera.release()
        except Exception as exc:
            log.info("Camera not ready: %s", exc)

        if timeout_s is not None and time.time() - start > timeout_s:
            raise RuntimeError("Camera never appeared")
        time.sleep(1)


def render_web_frame(frame_bgr, boxes, face_count, overlay_text="", flash=False):
    if flash:
        frame = np.ones_like(frame_bgr) * 255
    else:
        frame = frame_bgr.copy()

    for x, y, w, h in boxes or []:
        p1 = (max(0, int(x)), max(0, int(y)))
        p2 = (
            min(frame.shape[1] - 1, int(x + w)),
            min(frame.shape[0] - 1, int(y + h)),
        )
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"Faces: {face_count}",
        (10, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )

    if overlay_text:
        for index, line in enumerate(overlay_text.split("\n")):
            y = 45 + index * 36
            cv2.putText(
                frame,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return encoded.tobytes() if ok else None


def detection_worker(detector, detector_lock, game, stop_event):
    next_detection = 0
    while not stop_event.is_set():
        scoring_frame = game.pop_scoring_frame()
        if scoring_frame is not None:
            job_id, frame = scoring_frame
            try:
                with detector_lock:
                    faces, _ = detector.detect_faces_with_boxes(frame)
                game.complete_scoring(job_id, faces)
            except Exception as exc:
                log.exception("FPGA scoring detector failed")
                game.fail_scoring(str(exc), job_id=job_id)
            continue

        now = time.time()
        if now < next_detection:
            time.sleep(0.05)
            continue

        with game.lock:
            frame = game.latest_frame

        if frame is None:
            time.sleep(0.05)
            continue

        try:
            with detector_lock:
                faces, _ = detector.detect_faces_with_boxes(frame)
            game.update_live_faces(faces)
        except Exception:
            log.exception("Live detector pass failed")

        next_detection = time.time() + DETECTION_INTERVAL_S


def main():
    log.info("Face Frenzy starting")

    backend = os.environ.get("FACE_FRENZY_BACKEND", "fpga").lower()
    accelerator = None
    try:
        accelerator = load_finn_accelerator(BITFILE, FPGA_DIR)
        face_detector_module.set_accelerator(accelerator)
        log.info("Overlay loaded: %s", BITFILE)
    except Exception:
        if backend == "fpga":
            log.exception("FPGA backend requested but overlay failed to load")
            raise
        log.warning("Overlay failed to load; GPIO unavailable (CPU detection mode)")

    camera = wait_for_camera(timeout_s=None)
    display = NullDisplay()
    log.info("HDMI output disabled; using web interface for live view and controls")
    detector = FaceDetector()
    io = IOHandler(accelerator) if accelerator is not None else _NullIOHandler()

    log.info("Detector backend: %s", detector.backend)
    detector_lock = threading.Lock()

    face_counter = BoxCountFaceCounter(boxes_per_face=BOXES_PER_FACE)
    log.info("Face counter: %s (boxes_per_face=%.2f)", type(face_counter).__name__, BOXES_PER_FACE)

    game = GameController(
        camera=camera,
        detector=detector,
        display=display,
        io=io,
        max_players=4,
        detector_lock=detector_lock,
        face_counter=face_counter,
    )

    exporter = GameStateExporter(game)
    web_server = WebServer(exporter, game)
    web_server.start()
    log.info("Web server up on http://<board-ip>:5000")

    stop_event = threading.Event()
    detector_thread = threading.Thread(
        target=detection_worker,
        args=(detector, detector_lock, game, stop_event),
        daemon=True,
    )
    detector_thread.start()
    log.info("Live detection worker running every %.2fs", DETECTION_INTERVAL_S)

    try:
        while True:
            frame = camera.get_frame()
            if frame is not None:
                game.update_latest_frame(frame)
                game.tick()
                with game.lock:
                    flash = time.time() < game.flash_until
                    overlay_text = game.display.overlay_text
                    live_faces = list(game.live_faces)
                    live_face_count = game.live_face_count
                jpeg = render_web_frame(
                    frame,
                    boxes=live_faces,
                    face_count=live_face_count,
                    overlay_text=overlay_text,
                    flash=flash,
                )
                if jpeg is not None:
                    game.update_live_jpeg(jpeg)
                time.sleep(0.05)
            else:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                jpeg = render_web_frame(blank, boxes=[], face_count=0, overlay_text="Waiting for camera")
                if jpeg is not None:
                    game.update_live_jpeg(jpeg)
                time.sleep(0.1)
    except KeyboardInterrupt:
        log.info("Interrupt received")
    finally:
        stop_event.set()
        camera.release()
        display.stop()
        io.clear_leds()
        log.info("Shutdown complete")


if __name__ == "__main__":
    main()
