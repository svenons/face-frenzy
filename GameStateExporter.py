class GameStateExporter:
    def __init__(self, controller):
        self.controller = controller

    def export(self):
        with self.controller.lock:
            return {
                "score": self.controller.score,
                "round": self.controller.round,
                "strikes": self.controller.strikes,
                "target_faces": self.controller.target_faces,
                "detected_faces": self.controller.detected_faces,
                "live_face_count": self.controller.live_face_count,
                "selected_players": self.controller.selected_players,
                "max_players": self.controller.max_players,
                "scoring_pending": self.controller.scoring_pending,
                "last_error": self.controller.last_error,
                "current_state": self.controller.state.__class__.__name__,
            }
