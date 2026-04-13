import io
import os
import sys
import threading
import time
import wave
from pathlib import Path

from faster_whisper import WhisperModel

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


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
        print(f"Processed up to {current_end:.2f}s", flush=True)
        return

    percent = min(max(current_end / total_duration, 0.0), 1.0)
    filled = int(width * percent)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"[{bar}] {percent * 100:6.2f}% ({current_end:.2f}s / {total_duration:.2f}s)"
        ,
        flush=True,
    )


def render_waiting_status(
    stop_event: threading.Event,
    start_time: float,
    last_segment_end: list[float],
    total_duration: float | None,
) -> None:
    while not stop_event.wait(5.0):
        elapsed = time.time() - start_time
        processed = last_segment_end[0]
        if total_duration and total_duration > 0:
            percent = min(max(processed / total_duration, 0.0), 1.0) * 100
            print(
                f"Still working... elapsed {elapsed:.1f}s | last segment {processed:.2f}s / {total_duration:.2f}s ({percent:.2f}%)",
                flush=True,
            )
        else:
            print(
                f"Still working... elapsed {elapsed:.1f}s | last segment {processed:.2f}s",
                flush=True,
            )


def resolve_device_and_compute_type(device_preference: str) -> tuple[str, str, str]:
    preference = device_preference.lower()

    if preference not in {"auto", "cpu", "cuda"}:
        print(f"Error: unsupported device option: {device_preference}")
        print("Use one of: auto, cpu, cuda")
        sys.exit(1)

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    cuda_blocked = cuda_visible == "-1"

    if preference == "cpu":
        return "cpu", "int8", "CPU forced by user"

    if preference == "cuda":
        return "cuda", "float16", "CUDA requested by user"

    if cuda_blocked:
        return "cpu", "int8", "CPU selected because CUDA_VISIBLE_DEVICES disables GPU"

    return "cuda", "float16", "Auto mode: trying NVIDIA CUDA GPU first"


def transcribe(
    audio_file: str,
    language: str | None = None,
    model_size: str = "medium",
    device_preference: str = "auto",
) -> None:
    audio_path = Path(audio_file)
    if not audio_path.exists():
        print(f"Error: audio file not found: {audio_file}")
        sys.exit(1)

    transcription_start_time = time.time()
    total_duration = get_wav_duration(audio_file)
    output_file = audio_path.with_suffix(".txt")
    selected_device, compute_type, device_reason = resolve_device_and_compute_type(device_preference)

    print(f"Preparing transcription for: {audio_file}")
    requested_language = language if language else "auto-detect"
    print(f"Requested language: {requested_language}")
    print(f"Selected model: {model_size}")
    print(f"Device preference: {device_preference}")
    if device_preference.lower() == "auto":
        print("Device mode: using GPU if available, otherwise falling back to CPU.")
    print(f"Initial execution target: {selected_device} ({compute_type})")
    print(f"Device selection note: {device_reason}")
    print("Step 1/3: Loading Whisper model...")
    print("If this is the first run for this model, it may download files before transcription begins.")
    try:
        model = WhisperModel(model_size, device=selected_device, compute_type=compute_type)
        print(f"Step 1/3 complete: Model is ready on {selected_device}.")
    except Exception as exc:
        if selected_device == "cuda" and device_preference.lower() == "auto":
            print("CUDA setup is not available. Falling back to CPU.")
            print(f"CUDA load error: {exc}")
            selected_device = "cpu"
            compute_type = "int8"
            model = WhisperModel(model_size, device=selected_device, compute_type=compute_type)
            print("Step 1/3 complete: Model is ready on cpu.")
        else:
            raise

    print("Step 2/3: Starting transcription...")
    print(f"Transcribing: {audio_file}")
    if total_duration:
        print(f"Audio length: {total_duration:.2f}s")
    print(f"Writing transcript live to: {output_file}")
    print(f"Running on device: {selected_device} ({compute_type})")

    segments, info = model.transcribe(
        audio_file,
        language=language,
        beam_size=5,
    )

    print("Step 2/3 active: Receiving segments from the model...")
    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

    last_segment_end = [0.0]
    heartbeat_stop_event = threading.Event()
    heartbeat_start_time = time.time()
    heartbeat_thread = threading.Thread(
        target=render_waiting_status,
        args=(heartbeat_stop_event, heartbeat_start_time, last_segment_end, total_duration),
        daemon=True,
    )
    heartbeat_thread.start()

    try:
        with open(output_file, "w", encoding="utf-8") as transcript_file:
            for segment in segments:
                line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text.strip()}"
                transcript_file.write(line + "\n")
                transcript_file.flush()
                last_segment_end[0] = segment.end
                render_progress(segment.end, total_duration)
    except KeyboardInterrupt:
        heartbeat_stop_event.set()
        heartbeat_thread.join()
        print("Transcription cancelled by user.")
        print(f"Partial transcript saved to: {output_file}")
        sys.exit(130)
    finally:
        heartbeat_stop_event.set()
        heartbeat_thread.join()

    if total_duration:
        render_progress(total_duration, total_duration)
    total_elapsed = time.time() - transcription_start_time
    print("Step 3/3 complete: Transcript written successfully.")
    print("Transcription saved to:", output_file)
    print(f"Total processing time: {total_elapsed:.2f}s")
    if total_duration and total_duration > 0:
        print(f"Processing speed: {total_elapsed / total_duration:.2f}x audio duration")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_file> [language|auto] [model_size] [device]")
        print("  language:   auto (default), zh, en, ja, hi, fr, de, etc.")
        print("  model_size: tiny, base, small, medium (default), large-v3")
        print("  device:     auto (default), cpu, cuda")
        sys.exit(1)

    audio = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else "auto"
    lang = None if lang.lower() == "auto" else lang
    model = sys.argv[3] if len(sys.argv) > 3 else "medium"
    device = sys.argv[4] if len(sys.argv) > 4 else "auto"

    transcribe(audio, lang, model, device)
