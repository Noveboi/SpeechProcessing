"""
Usage:
    python visualizer.py <audio_file> <csv_file>
"""

import csv
import sys
import tkinter as tk
from pathlib import Path

import pygame

COLOR_FOREGROUND = "#2ECC40"  # GREEN
COLOR_BACKGROUND = "#FF4136"  # RED
COLOR_IDLE = "#AAAAAA"  # grey before playback starts


def load_segments(csv_path: str) -> list[dict]:
    """Return a list of {start, end, cls} dicts sorted by start time."""
    segments = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            segments.append(
                {
                    "start": float(row["start"]),
                    "end": float(row["end"]),
                    "cls": row["class"].strip().lower(),
                }
            )
    segments.sort(key=lambda s: s["start"])
    return segments


def class_at(segments: list[dict], t: float) -> str | None:
    """Return the class label for time t, or None if outside all segments."""
    for seg in segments:
        if seg["start"] <= t < seg["end"]:
            return seg["cls"]
    # past the last segment end → use the last segment's class
    if segments and t >= segments[-1]["end"]:
        return segments[-1]["cls"]
    return None


class Visualizer:
    """
    Simple polling-based window app that shows voice activity based on predicted data from the VAD system.
    """

    POLL_MS = 20  # how often (ms) to refresh the color

    def __init__(self, audio_path: str, csv_path: str):
        self.audio_path = audio_path
        self.segments = load_segments(csv_path)

        # Tkinter window
        self.root = tk.Tk()
        self.root.title("VAD Visualizer")
        self.root.geometry("640x380")
        self.root.resizable(True, True)
        self.root.configure(bg=COLOR_IDLE)

        # Label showing current class + timestamp
        self.status_var = tk.StringVar(value="Press  ▶ Play  to start")
        self.status_lbl = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Helvetica", 18, "bold"),
            bg=COLOR_IDLE,
            fg="white",
        )
        self.status_lbl.pack(expand=True)

        # Time label
        self.time_var = tk.StringVar(value="")
        self.time_lbl = tk.Label(
            self.root,
            textvariable=self.time_var,
            font=("Helvetica", 12),
            bg=COLOR_IDLE,
            fg="white",
        )
        self.time_lbl.pack(pady=(0, 10))

        # Play button
        self.play_btn = tk.Button(
            self.root,
            text="▶  Play",
            font=("Helvetica", 14),
            command=self.start_playback,
            width=10,
        )
        self.play_btn.pack(pady=(0, 20))

        # Pygame (audio only — no display)
        pygame.mixer.init()

        self._playing = False
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # Playback
    def start_playback(self):
        if self._playing:
            return
        try:
            pygame.mixer.music.load(self.audio_path)
        except pygame.error as exc:
            self.status_var.set(f"Error: {exc}")
            return

        pygame.mixer.music.play()
        self._playing = True
        self.play_btn.config(state=tk.DISABLED)
        self._poll()

    def _poll(self):
        """Called every POLL_MS ms while audio is playing."""
        if not pygame.mixer.music.get_busy():
            self._finish()
            return

        elapsed = pygame.mixer.music.get_pos() / 1000.0  # get_pos() → ms since play()
        cls = class_at(self.segments, elapsed)

        if cls == "foreground":
            color = COLOR_FOREGROUND
            label = "FOREGROUND"
        elif cls == "background":
            color = COLOR_BACKGROUND
            label = "BACKGROUND"
        else:
            color = COLOR_IDLE
            label = "…"

        self._set_color(color)
        self.status_var.set(label)
        self.time_var.set(f"{elapsed:.2f} s")

        self.root.after(self.POLL_MS, self._poll)

    def _finish(self):
        self._playing = False
        self._set_color(COLOR_IDLE)
        self.status_var.set("Playback finished")
        self.time_var.set("")
        self.play_btn.config(state=tk.NORMAL, text="▶  Replay")

    def _set_color(self, color: str):
        self.root.configure(bg=color)
        self.status_lbl.configure(bg=color)
        self.time_lbl.configure(bg=color)

    # Lifecycle
    def on_close(self):
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    if len(sys.argv) != 3:
        print("Usage: python visualizer.py <audio_file> <csv_file>")
        sys.exit(1)

    audio_path, csv_path = sys.argv[1], sys.argv[2]

    if not Path(audio_path).exists():
        sys.exit(f"Audio file not found: {audio_path}")
    if not Path(csv_path).exists():
        sys.exit(f"CSV file not found: {csv_path}")

    app = Visualizer(audio_path, csv_path)
    app.run()


if __name__ == "__main__":
    main()
