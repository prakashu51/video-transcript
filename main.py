import sys
import time
import io
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Suppress annoying pyannote torchcodec warnings
warnings.filterwarnings("ignore")

from config import resolve_device_and_compute_type
from audio_utils import get_wav_duration, build_output_path
from diarizer import run_diarization
from transcriber import load_whisper_model, run_whisper_pass
from translator import translate_text_lines

# Ensure utf-8 output matching old transcribe.py
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <audio_file> [source_language|auto] [model_size] [device] [target_language]")
        print("  source_language: auto (default), zh, en, ja, hi, fr, de, etc.")
        print("  model_size:      tiny, base, small, medium (default), large-v3")
        print("  device:          auto (default), cpu, cuda")
        print("  target_language: optional. Use en for Whisper translation or another mapped language for NLLB.")
        sys.exit(1)

    audio_file = sys.argv[1]
    source_lang = sys.argv[2] if len(sys.argv) > 2 else "auto"
    source_lang = None if source_lang.lower() == "auto" else source_lang
    model_size = sys.argv[3] if len(sys.argv) > 3 else "medium"
    device_pref = sys.argv[4] if len(sys.argv) > 4 else "auto"
    target_lang = sys.argv[5] if len(sys.argv) > 5 else None

    # Load HF_TOKEN from .env
    load_dotenv()

    audio_path = Path(audio_file)
    if not audio_path.exists():
        print(f"Error: audio file not found: {audio_file}")
        sys.exit(1)

    start_time = time.time()
    total_duration = get_wav_duration(audio_file)
    source_output_file = build_output_path(audio_path, ".txt")

    requested_language = source_lang if source_lang else "auto-detect"
    requested_target = target_lang if target_lang else "original transcript only"

    print(f"Preparing transcription for: {audio_file}")
    print(f"Requested source language: {requested_language}")
    print(f"Requested target language: {requested_target}")
    print(f"Selected model: {model_size}")

    # Phase 1: Diarization
    print("\n--- Phase 1: Diarization ---")
    speaker_segments = run_diarization(audio_file)

    # Phase 2: Transcription
    print("\n--- Phase 2: Transcription ---")
    model, selected_device, compute_type = load_whisper_model(model_size, device_pref)
    
    detected_language = run_whisper_pass(
        model=model,
        audio_file=audio_file,
        language=source_lang,
        output_file=source_output_file,
        total_duration=total_duration,
        task="transcribe",
        stage_name="transcription",
        speaker_segments=speaker_segments,
    )

    if not target_lang:
        total_elapsed = time.time() - start_time
        print("\nProcess complete: Transcript written successfully.")
        print(f"Total processing time: {total_elapsed:.2f}s")
        return

    # Phase 3: Translation (Optional)
    print("\n--- Phase 3: Translation ---")
    normalized_target = target_lang.lower()
    normalized_source = detected_language.lower()

    if normalized_target == normalized_source:
        total_elapsed = time.time() - start_time
        print("Target language matches the detected source language. No translation needed.")
        print(f"Total processing time: {total_elapsed:.2f}s")
        return

    english_output_file = build_output_path(audio_path, ".en.txt")
    bridge_file_to_use = source_output_file
    bridge_language = normalized_source

    if normalized_source != "en":
        print("Non-English audio detected. Running a Whisper pass with task='translate' to create an English bridge transcript.")
        run_whisper_pass(
            model=model,
            audio_file=audio_file,
            language=source_lang,
            output_file=english_output_file,
            total_duration=total_duration,
            task="translate",
            stage_name="English translation",
            speaker_segments=speaker_segments,
        )
        if normalized_target == "en":
            total_elapsed = time.time() - start_time
            print("\nProcess complete: Source transcript and English translation are ready.")
            print(f"Total processing time: {total_elapsed:.2f}s")
            return
            
        print("Using the vastly superior English translation as a bridge for the final target language...")
        bridge_file_to_use = english_output_file
        bridge_language = "en"

    translated_output_file = build_output_path(audio_path, f".{normalized_target}.txt")
    translate_text_lines(
        input_file=bridge_file_to_use,
        output_file=translated_output_file,
        source_language=bridge_language,
        target_language=normalized_target,
    )

    total_elapsed = time.time() - start_time
    print("\nProcess complete.")
    print(f"Total processing time: {total_elapsed:.2f}s")

if __name__ == "__main__":
    main()
