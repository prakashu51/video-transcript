import io
import os
import sys
import threading
import time
import wave
from pathlib import Path

from faster_whisper import WhisperModel

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

NLLB_LANGUAGE_MAP = {
    "ar": "arb_Arab",
    "de": "deu_Latn",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "hi": "hin_Deva",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
}


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


def load_whisper_model(model_size: str, device_preference: str) -> tuple[WhisperModel, str, str]:
    selected_device, compute_type, device_reason = resolve_device_and_compute_type(device_preference)

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
        return model, selected_device, compute_type
    except Exception as exc:
        if selected_device == "cuda" and device_preference.lower() == "auto":
            print("CUDA setup is not available. Falling back to CPU.")
            print(f"CUDA load error: {exc}")
            selected_device = "cpu"
            compute_type = "int8"
            model = WhisperModel(model_size, device=selected_device, compute_type=compute_type)
            print("Step 1/3 complete: Model is ready on cpu.")
            return model, selected_device, compute_type
        raise


def build_output_path(audio_path: Path, suffix: str) -> Path:
    return audio_path.with_suffix(suffix)


def write_segments_to_file(
    segments,
    output_file: Path,
    total_duration: float | None,
    stage_label: str,
) -> None:
    last_segment_end = [0.0]
    heartbeat_stop_event = threading.Event()
    heartbeat_start_time = time.time()
    heartbeat_thread = threading.Thread(
        target=render_waiting_status,
        args=(heartbeat_stop_event, heartbeat_start_time, last_segment_end, total_duration, stage_label),
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


def translate_text_lines(
    input_file: Path,
    output_file: Path,
    source_language: str,
    target_language: str,
) -> None:
    source_code = NLLB_LANGUAGE_MAP.get(source_language)
    target_code = NLLB_LANGUAGE_MAP.get(target_language)

    if not source_code:
        print(f"Error: source language '{source_language}' is not mapped for NLLB translation yet.")
        print(f"Currently supported NLLB language codes: {', '.join(sorted(NLLB_LANGUAGE_MAP))}")
        sys.exit(1)

    if not target_code:
        print(f"Error: target language '{target_language}' is not mapped for NLLB translation yet.")
        print(f"Currently supported NLLB language codes: {', '.join(sorted(NLLB_LANGUAGE_MAP))}")
        sys.exit(1)

    print("Step 3/4: Loading translation model for non-English target output...")
    print("If this is the first translation run, it may download model files before translation begins.")

    try:
        import torch
        from transformers import pipeline
    except ImportError:
        print("Error: translation dependencies are missing.")
        print("Install them with: pip install -r requirements.txt")
        sys.exit(1)

    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else None

    translator = pipeline(
        task="translation",
        model="facebook/nllb-200-distilled-600M",
        src_lang=source_code,
        tgt_lang=target_code,
        device=device,
        dtype=dtype,
    )

    print("Step 3/4 complete: Translation model is ready.")
    print(f"Step 4/4: Translating transcript into '{target_language}'...")

    with open(input_file, "r", encoding="utf-8") as source_file:
        source_lines = [line.rstrip("\n") for line in source_file]

    translatable_lines = []
    line_metadata = []
    for line in source_lines:
        if "] " in line:
            timestamp, text = line.split("] ", 1)
            line_metadata.append(timestamp + "] ")
            translatable_lines.append(text)
        else:
            line_metadata.append("")
            translatable_lines.append(line)

    total_lines = len(translatable_lines)
    with open(output_file, "w", encoding="utf-8") as translated_file:
        for index, (prefix, text) in enumerate(zip(line_metadata, translatable_lines), start=1):
            if text.strip():
                result = translator(text, max_length=512)
                translated_text = result[0]["translation_text"].strip()
            else:
                translated_text = ""

            translated_file.write(f"{prefix}{translated_text}\n")
            translated_file.flush()

            percent = (index / total_lines) * 100 if total_lines else 100.0
            print(f"Translated {index}/{total_lines} lines ({percent:.2f}%)", flush=True)

    print(f"Translated transcript saved to: {output_file}")


def run_whisper_pass(
    model: WhisperModel,
    audio_file: str,
    language: str | None,
    output_file: Path,
    total_duration: float | None,
    task: str,
    stage_name: str,
) -> str:
    print(f"Step 2/3: Starting {stage_name}...")
    print(f"Transcribing: {audio_file}")
    if total_duration:
        print(f"Audio length: {total_duration:.2f}s")
    print(f"Writing transcript live to: {output_file}")

    segments, info = model.transcribe(
        audio_file,
        language=language,
        beam_size=5,
        task=task,
    )

    print(f"Running Whisper task: {task}")
    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    write_segments_to_file(segments, output_file, total_duration, stage_name)
    if total_duration:
        render_progress(total_duration, total_duration)
    return info.language


def transcribe(
    audio_file: str,
    language: str | None = None,
    model_size: str = "medium",
    device_preference: str = "auto",
    target_language: str | None = None,
) -> None:
    audio_path = Path(audio_file)
    if not audio_path.exists():
        print(f"Error: audio file not found: {audio_file}")
        sys.exit(1)

    start_time = time.time()
    total_duration = get_wav_duration(audio_file)
    source_output_file = build_output_path(audio_path, ".txt")

    requested_language = language if language else "auto-detect"
    requested_target = target_language if target_language else "original transcript only"

    print(f"Preparing transcription for: {audio_file}")
    print(f"Requested source language: {requested_language}")
    print(f"Requested target language: {requested_target}")
    print(f"Selected model: {model_size}")

    model, selected_device, compute_type = load_whisper_model(model_size, device_preference)
    print(f"Running on device: {selected_device} ({compute_type})")

    detected_language = run_whisper_pass(
        model=model,
        audio_file=audio_file,
        language=language,
        output_file=source_output_file,
        total_duration=total_duration,
        task="transcribe",
        stage_name="transcription",
    )

    if not target_language:
        total_elapsed = time.time() - start_time
        print("Step 3/3 complete: Transcript written successfully.")
        print("Transcription saved to:", source_output_file)
        print(f"Total processing time: {total_elapsed:.2f}s")
        if total_duration and total_duration > 0:
            print(f"Processing speed: {total_elapsed / total_duration:.2f}x audio duration")
        return

    normalized_target = target_language.lower()
    normalized_source = detected_language.lower()

    if normalized_target == normalized_source:
        total_elapsed = time.time() - start_time
        print("Target language matches the detected source language.")
        print("Using the original transcript as the final output.")
        print("Transcript saved to:", source_output_file)
        print(f"Total processing time: {total_elapsed:.2f}s")
        if total_duration and total_duration > 0:
            print(f"Processing speed: {total_elapsed / total_duration:.2f}x audio duration")
        return

    if normalized_target == "en" and normalized_source != "en":
        english_output_file = build_output_path(audio_path, ".en.txt")
        print("English target requested. Running a second Whisper pass with task='translate'.")
        run_whisper_pass(
            model=model,
            audio_file=audio_file,
            language=language,
            output_file=english_output_file,
            total_duration=total_duration,
            task="translate",
            stage_name="English translation",
        )
        total_elapsed = time.time() - start_time
        print("Step 3/3 complete: Source transcript and English translation are ready.")
        print("Source transcript saved to:", source_output_file)
        print("English translation saved to:", english_output_file)
        print(f"Total processing time: {total_elapsed:.2f}s")
        if total_duration and total_duration > 0:
            print(f"Processing speed: {total_elapsed / total_duration:.2f}x audio duration")
        return

    translated_output_file = build_output_path(audio_path, f".{normalized_target}.txt")
    print(f"Non-English target requested. Translating '{normalized_source}' transcript into '{normalized_target}'.")
    translate_text_lines(
        input_file=source_output_file,
        output_file=translated_output_file,
        source_language=normalized_source,
        target_language=normalized_target,
    )

    total_elapsed = time.time() - start_time
    print("Translation pipeline complete.")
    print("Source transcript saved to:", source_output_file)
    print(f"Translated transcript saved to: {translated_output_file}")
    print(f"Total processing time: {total_elapsed:.2f}s")
    if total_duration and total_duration > 0:
        print(f"Processing speed: {total_elapsed / total_duration:.2f}x audio duration")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <audio_file> [source_language|auto] [model_size] [device] [target_language]")
        print("  source_language: auto (default), zh, en, ja, hi, fr, de, etc.")
        print("  model_size:      tiny, base, small, medium (default), large-v3")
        print("  device:          auto (default), cpu, cuda")
        print("  target_language: optional. Use en for Whisper translation or another mapped language for NLLB.")
        sys.exit(1)

    audio = sys.argv[1]
    source_lang = sys.argv[2] if len(sys.argv) > 2 else "auto"
    source_lang = None if source_lang.lower() == "auto" else source_lang
    model = sys.argv[3] if len(sys.argv) > 3 else "medium"
    device = sys.argv[4] if len(sys.argv) > 4 else "auto"
    target = sys.argv[5] if len(sys.argv) > 5 else None

    transcribe(audio, source_lang, model, device, target)
