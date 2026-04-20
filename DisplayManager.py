import cv2
import numpy as np
from pynq.lib.video import VideoMode


class DisplayManager:
    def __init__(self, overlay, resolution=(640, 480)):
        self.overlay = overlay
        self.hdmi_out = overlay.video.hdmi_out
        self.width, self.height = resolution
        self.overlay_text = ""
        self.active = False

        self.hdmi_out.configure(VideoMode(self.width, self.height, 24))
        self.hdmi_out.start()
        self.active = True

    def set_overlay_text(self, text):
        self.overlay_text = text

    def clear_overlay_text(self):
        self.overlay_text = ""

    def show_frame(self, frame_bgr, boxes=None, face_count=None):
        if not self.active:
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if frame_rgb.shape[1] != self.width or frame_rgb.shape[0] != self.height:
            frame_rgb = cv2.resize(frame_rgb, (self.width, self.height))

        scale_x = self.width / frame_bgr.shape[1]
        scale_y = self.height / frame_bgr.shape[0]

        for x, y, w, h in boxes or []:
            p1 = (max(0, int(x * scale_x)), max(0, int(y * scale_y)))
            p2 = (
                min(self.width - 1, int((x + w) * scale_x)),
                min(self.height - 1, int((y + h) * scale_y)),
            )
            cv2.rectangle(frame_rgb, p1, p2, (0, 255, 0), 2)

        if face_count is not None:
            cv2.putText(
                frame_rgb,
                f"Faces: {face_count}",
                (10, self.height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        if self.overlay_text:
            for index, line in enumerate(self.overlay_text.split("\n")):
                y = 50 + index * 40
                cv2.putText(
                    frame_rgb,
                    line,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

        out_frame = self.hdmi_out.newframe()
        out_frame[:, :, :] = frame_rgb
        self.hdmi_out.writeframe(out_frame)

    def white_screen(self):
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        self.show_frame(frame)

    def stop(self):
        if self.active:
            self.hdmi_out.stop()
            self.active = False


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
