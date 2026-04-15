import time
import wave
from pathlib import Path
import threading

def get_wav_duration(audio_file: str) -> float | None:
    try:
        with wave.open(audio_file, "rb") as wav_file:
            frame_count = wav_file.getnframes()
            frame_rate = wav_file.getframerate()
            if frame_rate > 0:
                return frame_count / float(frame_rate)
    except (wave.Error, FileNotFoundError):
        return None
    return None

def build_output_path(audio_path: Path, suffix: str) -> Path:
    return audio_path.with_suffix(suffix)

def render_progress(current_end: float, total_duration: float | None, width: int = 30) -> None:
    if not total_duration or total_duration <= 0:
        print(f"Processed up to {current_end:.2f}s", flush=True)
        return

    percent = min(max(current_end / total_duration, 0.0), 1.0)
    filled = int(width * percent)
    bar = "#" * filled + "-" * (width - filled)
    print(f"[{bar}] {percent * 100:6.2f}% ({current_end:.2f}s / {total_duration:.2f}s)", flush=True)

def render_waiting_status(
    stop_event: threading.Event,
    start_time: float,
    last_segment_end: list[float],
    total_duration: float | None,
    stage_label: str,
) -> None:
    while not stop_event.wait(5.0):
        elapsed = time.time() - start_time
        processed = last_segment_end[0]
        if total_duration and total_duration > 0:
            percent = min(max(processed / total_duration, 0.0), 1.0) * 100
            print(
                f"Still working on {stage_label}... elapsed {elapsed:.1f}s | last segment {processed:.2f}s / {total_duration:.2f}s ({percent:.2f}%)",
                flush=True,
            )
        else:
            print(
                f"Still working on {stage_label}... elapsed {elapsed:.1f}s | last segment {processed:.2f}s",
                flush=True,
            )
