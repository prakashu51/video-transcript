import io
import sys
import threading
import time
import wave
from pathlib import Path

from faster_whisper import WhisperModel

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
PRINT_LOCK = threading.Lock()


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


def render_progress(current_end: float, total_duration: float | None, width: int = 30) -> None:
    if not total_duration or total_duration <= 0:
        with PRINT_LOCK:
            print(f"\rProcessed up to {current_end:.2f}s", end="", flush=True)
        return

    percent = min(max(current_end / total_duration, 0.0), 1.0)
    filled = int(width * percent)
    bar = "#" * filled + "-" * (width - filled)
    with PRINT_LOCK:
        print(
            f"\r[{bar}] {percent * 100:6.2f}% ({current_end:.2f}s / {total_duration:.2f}s)",
            end="",
            flush=True,
        )


def render_waiting_status(
    stop_event: threading.Event,
    start_time: float,
    last_segment_end: list[float],
    total_duration: float | None,
) -> None:
    spinner_frames = "|/-\\"
    spinner_index = 0

    while not stop_event.is_set():
        elapsed = time.time() - start_time
        processed = last_segment_end[0]
        if total_duration and total_duration > 0:
            percent = min(max(processed / total_duration, 0.0), 1.0) * 100
            status = (
                f"\r{spinner_frames[spinner_index]} Working... "
                f"elapsed {elapsed:6.1f}s | last segment {processed:6.2f}s / {total_duration:.2f}s "
                f"({percent:5.2f}%)"
            )
        else:
            status = (
                f"\r{spinner_frames[spinner_index]} Working... "
                f"elapsed {elapsed:6.1f}s | last segment {processed:6.2f}s"
            )

        with PRINT_LOCK:
            print(status, end="", flush=True)

        spinner_index = (spinner_index + 1) % len(spinner_frames)
        stop_event.wait(0.1)


def transcribe(audio_file: str, language: str | None = None, model_size: str = "medium") -> None:
    audio_path = Path(audio_file)
    if not audio_path.exists():
        print(f"Error: audio file not found: {audio_file}")
        sys.exit(1)

    total_duration = get_wav_duration(audio_file)
    output_file = audio_path.with_suffix(".txt")

    print(f"Preparing transcription for: {audio_file}")
    requested_language = language if language else "auto-detect"
    print(f"Requested language: {requested_language}")
    print(f"Selected model: {model_size}")
    print("Step 1/3: Loading Whisper model...")
    print("If this is the first run for this model, it may download files before transcription begins.")
    model = WhisperModel(model_size, compute_type="int8")
    print("Step 1/3 complete: Model is ready.")

    print("Step 2/3: Starting transcription...")
    print(f"Transcribing: {audio_file}")
    if total_duration:
        print(f"Audio length: {total_duration:.2f}s")
    print(f"Writing transcript live to: {output_file}")

    segments, info = model.transcribe(
        audio_file,
        language=language,
        beam_size=5,
    )

    print("Step 2/3 active: Receiving segments from the model...")
    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

    last_segment_end = [0.0]
    spinner_stop_event = threading.Event()
    spinner_start_time = time.time()
    spinner_thread = threading.Thread(
        target=render_waiting_status,
        args=(spinner_stop_event, spinner_start_time, last_segment_end, total_duration),
        daemon=True,
    )
    spinner_thread.start()

    try:
        with open(output_file, "w", encoding="utf-8") as transcript_file:
            try:
                for segment in segments:
                    line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text.strip()}"
                    transcript_file.write(line + "\n")
                    transcript_file.flush()
                    last_segment_end[0] = segment.end
                    render_progress(segment.end, total_duration)
            finally:
                spinner_stop_event.set()
                spinner_thread.join()
    except KeyboardInterrupt:
        spinner_stop_event.set()
        if spinner_thread.is_alive():
            spinner_thread.join()
        with PRINT_LOCK:
            print("\nTranscription cancelled by user.")
            print(f"Partial transcript saved to: {output_file}")
        sys.exit(130)

    if total_duration:
        render_progress(total_duration, total_duration)
    with PRINT_LOCK:
        print("\nStep 3/3 complete: Transcript written successfully.")
        print("Transcription saved to:", output_file)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_file> [language|auto] [model_size]")
        print("  language:   auto (default), zh, en, ja, hi, fr, de, etc.")
        print("  model_size: tiny, base, small, medium (default), large-v3")
        sys.exit(1)

    audio = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else "auto"
    lang = None if lang.lower() == "auto" else lang
    model = sys.argv[3] if len(sys.argv) > 3 else "medium"

    transcribe(audio, lang, model)
