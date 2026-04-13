import io
import sys
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
        print(f"\rProcessed up to {current_end:.2f}s", end="", flush=True)
        return

    percent = min(max(current_end / total_duration, 0.0), 1.0)
    filled = int(width * percent)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\r[{bar}] {percent * 100:6.2f}% ({current_end:.2f}s / {total_duration:.2f}s)",
        end="",
        flush=True,
    )


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

    with open(output_file, "w", encoding="utf-8") as transcript_file:
        for segment in segments:
            line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text.strip()}"
            transcript_file.write(line + "\n")
            transcript_file.flush()
            render_progress(segment.end, total_duration)

    if total_duration:
        render_progress(total_duration, total_duration)
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
