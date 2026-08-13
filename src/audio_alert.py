"""Non-blocking local audio alerts for Phase 13."""

from __future__ import annotations

import sys
import threading


class AudioAlertPlayer:
    """Play simple warning tones without blocking the video loop."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def play(self, level: str) -> bool:
        if not self.enabled or level not in {"CAUTION", "DANGER"}:
            return False
        thread = threading.Thread(target=self._play_worker, args=(level,), daemon=True)
        thread.start()
        return True

    def _play_worker(self, level: str) -> None:
        try:
            if sys.platform.startswith("win"):
                import winsound

                if level == "DANGER":
                    for _ in range(2):
                        winsound.Beep(1400, 180)
                        winsound.Beep(1800, 180)
                else:
                    winsound.Beep(900, 180)
            else:
                print("\a", end="", flush=True)
        except Exception as error:
            print(f"Audio warning failed: {error}")
