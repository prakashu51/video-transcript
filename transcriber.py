import sys
import time
import threading
from pathlib import Path
from faster_whisper import WhisperModel
from config import resolve_device_and_compute_type
from audio_utils import render_progress, render_waiting_status
from diarizer import align_speaker_with_segment

def load_whisper_model(model_size: str, device_preference: str) -> tuple[WhisperModel, str, str]:
    selected_device, compute_type, device_reason = resolve_device_and_compute_type(device_preference)

    print(f"Device preference: {device_preference}")
    print(f"Initial execution target: {selected_device} ({compute_type})")
    print(f"Device selection note: {device_reason}")
    print("Loading Whisper model...")

    try:
        model = WhisperModel(model_size, device=selected_device, compute_type=compute_type)
        print(f"Model is ready on {selected_device}.")
        return model, selected_device, compute_type
    except Exception as exc:
        if selected_device == "cuda" and device_preference.lower() == "auto":
            print("CUDA setup is not available. Falling back to CPU.")
            print(f"CUDA load error: {exc}")
            selected_device = "cpu"
            compute_type = "int8"
            model = WhisperModel(model_size, device=selected_device, compute_type=compute_type)
            print("Model is ready on cpu.")
            return model, selected_device, compute_type
        raise

def write_segments_to_file(
    segments,
    output_file: Path,
    total_duration: float | None,
    stage_label: str,
    speaker_segments: list[dict]
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
                # Align speaker
                speaker_label = align_speaker_with_segment(segment.start, segment.end, speaker_segments)
                
                # Format output
                line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] [{speaker_label}]: {segment.text.strip()}"
                
                transcript_file.write(line + "\n")
                transcript_file.flush()
                last_segment_end[0] = segment.end
                render_progress(segment.end, total_duration)
    except KeyboardInterrupt:
        heartbeat_stop_event.set()
        heartbeat_thread.join()
        print("\nTranscription cancelled by user.")
        print(f"Partial transcript saved to: {output_file}")
        sys.exit(130)
    finally:
        heartbeat_stop_event.set()
        heartbeat_thread.join()

def run_whisper_pass(
    model: WhisperModel,
    audio_file: str,
    language: str | None,
    output_file: Path,
    total_duration: float | None,
    task: str,
    stage_name: str,
    speaker_segments: list[dict]
) -> str:
    print(f"\nStarting {stage_name}...")
    print(f"Writing transcript live to: {output_file}")

    segments, info = model.transcribe(
        audio_file,
        language=language,
        beam_size=5,
        task=task,
    )

    print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
    write_segments_to_file(segments, output_file, total_duration, stage_name, speaker_segments)
    if total_duration:
        render_progress(total_duration, total_duration)
    return info.language
