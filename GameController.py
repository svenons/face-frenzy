import random
import os
import threading
import time

from FaceDetector import BoxCountFaceCounter


SCORING_TIMEOUT_S = float(os.environ.get("FACE_FRENZY_SCORING_TIMEOUT_S", "8.0"))

class GameState:
    def __init__(self, controller):
        self.controller = controller

    def run_once(self):
        raise NotImplementedError("State must implement run_once()")

class IdleState(GameState):
    def __init__(self, controller):
        super().__init__(controller)
        self.last_button_time = 0
        self.last_button_poll = 0
        self.last_led_time = 0
        self.flash_all = True

    def run_once(self):
        btn0, btn1, btn3 = self.controller.io.btn0, self.controller.io.btn1, self.controller.io.btn3
        now = time.time()

        if now - self.last_led_time >= 0.4:
            self.flash_all = not self.flash_all
            self.last_led_time = now
        self.controller.io.show_player_select(self.controller.selected_players, self.flash_all)

        if now - self.last_button_poll >= 0.2:
            self.last_button_poll = now
            if btn0.read() and now - self.last_button_time >= 0.4:
                self.controller.increment_players()
                self.last_button_time = now

            if btn1.read() and now - self.last_button_time >= 0.4:
                self.controller.decrement_players()
                self.last_button_time = now

        self.controller.display.set_overlay_text(
            f"Select players: {self.controller.selected_players}\nPress BTN3 or Start"
        )

        start_pressed = False
        if now - self.last_button_poll < 0.05:
            start_pressed = btn3.read()

        if start_pressed or self.controller.consume_start_request():
            self.controller.max_players = self.controller.selected_players
            self.controller.display.clear_overlay_text()
            self.controller.io.clear_leds()
            self.controller.set_state(GetReadyState(self.controller))
            time.sleep(0.5)

class GetReadyState(GameState):
    def __init__(self, controller):
        super().__init__(controller)
        self.start_time = time.time()
        self.timeout = 2
        self.controller.round += 1
        self.controller.target_faces = random.randint(0, self.controller.max_players)
        self.controller.display.set_overlay_text(f"Show {self.controller.target_faces} face(s)!")

    def run_once(self):
        if time.time() - self.start_time >= self.timeout:
            self.controller.display.clear_overlay_text()
            self.controller.set_state(CountdownState(self.controller))

class CountdownState(GameState):
    def __init__(self, controller):
        super().__init__(controller)
        self.total_time = random.randint(3, 5)
        self.remaining = self.total_time
        self.last_tick = time.time()
        self.controller.display.set_overlay_text(str(self.remaining))
        self.controller.io.set_led_countdown(self.remaining, self.total_time)

    def run_once(self):
        now = time.time()
        if now - self.last_tick >= 1:
            self.remaining -= 1
            self.last_tick = now
            if self.remaining > 0:
                self.controller.display.set_overlay_text(str(self.remaining))
                self.controller.io.set_led_countdown(self.remaining, self.total_time)
            else:
                self.controller.display.clear_overlay_text()
                self.controller.io.clear_leds()
                self.controller.set_state(CaptureState(self.controller))

class CaptureState(GameState):
    def __init__(self, controller):
        super().__init__(controller)
        self.capture_time = time.time()
        self.captured = False
        self.controller.flash_until = self.capture_time + 0.3
        self.controller.display.white_screen()

    def run_once(self):
        if not self.captured and time.time() - self.capture_time >= 0.3:
            if not self.controller.request_scoring_from_latest_frame():
                self.controller.display.set_overlay_text("Waiting for frame")
                return
            self.captured = True
            self.controller.set_state(DetectState(self.controller))


class DetectState(GameState):
    def __init__(self, controller):
        super().__init__(controller)
        self.start_time = time.time()
        self.job_id = controller.current_scoring_job_id()
        self.controller.display.set_overlay_text("Detecting...")

    def run_once(self):
        result = self.controller.consume_scoring_result()
        if result is None:
            elapsed = time.time() - self.start_time
            if elapsed < SCORING_TIMEOUT_S:
                self.controller.display.set_overlay_text(f"Detecting... {elapsed:.0f}s")
                return
            self.controller.fail_scoring(
                f"FPGA scoring timed out after {SCORING_TIMEOUT_S:.1f}s",
                job_id=self.job_id,
                force=True,
            )
            result = self.controller.consume_scoring_result()
            if result is None:
                return

        faces, error = result
        if error:
            self.controller.last_error = error
            self.controller.detected_faces = -1
        else:
            self.controller.detected_faces = self.controller.face_counter.count(
                faces, max_faces=self.controller.max_players
            )
        self.controller.set_state(EvaluateState(self.controller))


class EvaluateState(GameState):
    def __init__(self, controller):
        super().__init__(controller)
        self.start_time = time.time()
        if self.controller.detected_faces < 0:
            success = False
            self.controller.strikes += 1
            message = "FPGA Error"
        elif self.controller.detected_faces == self.controller.target_faces:
            success = True
            self.controller.score += 1
            message = "Correct!"
        else:
            success = False
            self.controller.strikes += 1
            message = "Wrong!"

        self.controller.io.show_result(success)
        self.controller.display.set_overlay_text(message)

    def run_once(self):
        if time.time() - self.start_time >= 2:
            self.controller.display.clear_overlay_text()
            if self.controller.strikes >= 2:
                self.controller.set_state(GameOverState(self.controller))
            else:
                self.controller.set_state(GetReadyState(self.controller))


class GameOverState(GameState):
    def __init__(self, controller):
        super().__init__(controller)
        self.start_time = time.time()
        self.controller.display.set_overlay_text(f"Game Over\nScore: {self.controller.score}")

    def run_once(self):
        if time.time() - self.start_time >= 3:
            self.controller.display.clear_overlay_text()
            self.controller.reset()
            self.controller.set_state(IdleState(self.controller))


class GameController:
    def __init__(self, camera, detector, display, io, max_players=4, detector_lock=None, face_counter=None):
        self.lock = threading.RLock()
        self.detector_lock = detector_lock or threading.Lock()
        self.camera = camera
        self.detector = detector
        self.display = display
        self.io = io
        self.face_counter = face_counter or BoxCountFaceCounter()
        self.max_players = max_players
        self.selected_players = 1
        self.start_requested = False
        self.latest_jpeg = None
        self.latest_frame = None
        self.scoring_frame = None
        self.scoring_result = None
        self.scoring_pending = False
        self.scoring_job_id = 0
        self.last_error = ""
        self.jpeg_seq = 0
        self.state_seq = 0
        self._jpeg_cond = threading.Condition()
        self._state_cond = threading.Condition()
        self.reset()
        self.state = IdleState(self)

    def reset(self):
        with self.lock:
            self.score = 0
            self.strikes = 0
            self.round = 0
            self.target_faces = 0
            self.detected_faces = 0
            self.live_faces = []
            self.live_face_count = 0
            self.flash_until = 0
            self.captured_frame = None
            self.start_requested = False
            self.scoring_frame = None
            self.scoring_result = None
            self.scoring_pending = False
            self.scoring_job_id = 0
            self.last_error = ""

    def set_state(self, new_state):
        with self.lock:
            self.state = new_state
        self._notify_state()

    def update_live_faces(self, faces):
        with self.lock:
            self.live_faces = list(faces)
            self.live_face_count = self.face_counter.count(self.live_faces, max_faces=self.max_players)
        self._notify_state()

    def update_live_jpeg(self, jpeg_bytes):
        with self._jpeg_cond:
            self.latest_jpeg = jpeg_bytes
            self.jpeg_seq += 1
            self._jpeg_cond.notify_all()

    def update_latest_frame(self, frame):
        with self.lock:
            self.latest_frame = frame

    def request_scoring_from_latest_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return False
            self.captured_frame = self.latest_frame
            self.scoring_frame = self.latest_frame
            self.scoring_result = None
            self.scoring_job_id += 1
            self.scoring_pending = True
            return True

    def pop_scoring_frame(self):
        with self.lock:
            if not self.scoring_pending or self.scoring_frame is None:
                return None
            frame = self.scoring_frame
            job_id = self.scoring_job_id
            self.scoring_frame = None
            return job_id, frame

    def _notify_state(self):
        with self._state_cond:
            self.state_seq += 1
            self._state_cond.notify_all()

    def wait_for_jpeg(self, last_seq, timeout):
        with self._jpeg_cond:
            if self.jpeg_seq == last_seq:
                self._jpeg_cond.wait(timeout)
            return self.jpeg_seq, self.latest_jpeg

    def wait_for_state(self, last_seq, timeout):
        with self._state_cond:
            if self.state_seq == last_seq:
                self._state_cond.wait(timeout)
            return self.state_seq

    def complete_scoring(self, job_id, faces):
        with self.lock:
            if job_id != self.scoring_job_id or not self.scoring_pending:
                return
            self.scoring_result = (list(faces), "")
            self.scoring_pending = False
        self._notify_state()

    def fail_scoring(self, error, job_id=None, force=False):
        with self.lock:
            if job_id is not None and job_id != self.scoring_job_id:
                return
            if not force and not self.scoring_pending:
                return
            self.scoring_result = ([], error)
            self.scoring_pending = False
        self._notify_state()

    def current_scoring_job_id(self):
        with self.lock:
            return self.scoring_job_id

    def consume_scoring_result(self):
        with self.lock:
            result = self.scoring_result
            self.scoring_result = None
            return result

    def increment_players(self):
        with self.lock:
            if self.state.__class__.__name__ == "IdleState":
                self.selected_players = min(4, self.selected_players + 1)
        self._notify_state()

    def decrement_players(self):
        with self.lock:
            if self.state.__class__.__name__ == "IdleState":
                self.selected_players = max(1, self.selected_players - 1)
        self._notify_state()

    def request_start(self):
        with self.lock:
            if self.state.__class__.__name__ == "IdleState":
                self.start_requested = True
        self._notify_state()

    def reset_game(self):
        with self.lock:
            self.io.clear_leds()
            self.display.clear_overlay_text()
            self.reset()
            self.state = IdleState(self)
        self._notify_state()

    def consume_start_request(self):
        with self.lock:
            requested = self.start_requested
            self.start_requested = False
            return requested

    def tick(self):
        with self.lock:
            self.state.run_once()
