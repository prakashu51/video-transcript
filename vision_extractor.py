"""
Vision Extractor Module for Scene-Aware Context.
Extracts frames at regular intervals and uses Ollama Vision Models (LLava)
to generate textual descriptions of the scenes.
"""
import sys
import base64
import ollama
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None

from config import VISION_MODEL, VISION_PROMPT, VISION_INTERVAL_SEC

def _encode_frame_to_base64(frame) -> str:
    """Encode OpenCV frame to base64 JPEG format."""
    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        return ""
    return base64.b64encode(buffer).decode('utf-8')

def _is_video_file(file_path: str) -> bool:
    """Check if the file is a likely video format."""
    ext = Path(file_path).suffix.lower()
    return ext in [".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"]

def extract_visual_context(video_path: str, interval_sec: float = VISION_INTERVAL_SEC) -> list[dict]:
    """
    Scans the video and extracts a visual description every `interval_sec` seconds.
    Returns a list of dicts: {"start": float, "end": float, "text": str}.
    """
    if cv2 is None:
        print("Warning: opencv-python is not installed. Skipping Vision Context Extraction.")
        print("Please install via: pip install opencv-python")
        return []

    if not _is_video_file(video_path):
        # Audio-only files don't need vision extraction
        return []

    # Check if the Ollama model exists
    try:
        ollama.show(VISION_MODEL)
    except ollama.ResponseError:
        print(f"\nWarning: Ollama vision model '{VISION_MODEL}' not found locally.")
        print(f"To use Visual Context, please run: ollama pull {VISION_MODEL}")
        print("Skipping Vision Context Extraction...\n")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video file for vision extraction: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0 or total_frames == 0:
        return []
        
    duration_sec = total_frames / fps
    total_intervals = int(duration_sec // interval_sec)
    
    print(f"\n--- Phase 1.5: Scene-Aware Vision Extraction ---")
    print(f"Scanning video ({duration_sec:.2f}s) every {interval_sec} seconds...")
    print(f"Processing approximate {total_intervals} frames using '{VISION_MODEL}'...")

    visual_contexts = []
    
    for i in range(1, total_intervals + 1):
        target_sec = i * interval_sec
        # Set video to timestamp
        cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        print(f"  Analyzing frame at {target_sec:.1f}s... ", end="", flush=True)
        img_b64 = _encode_frame_to_base64(frame)
        if not img_b64:
            print("Failed to encode frame.")
            continue
            
        try:
            response = ollama.generate(
                model=VISION_MODEL,
                prompt=VISION_PROMPT,
                images=[img_b64]
            )
            description = response.get('response', '').strip()
            # Clean up newlines if the model was hallucinating long blocks
            description = description.replace("\n", " ")
            
            # Start and End will be the exact same timestamp to denote an instantaneous event
            visual_contexts.append({
                "start": target_sec,
                # Give it a 0.01 gap so formatting logic renders it cleanly 
                "end": target_sec + 0.01,
                "text": description
            })
            print("Done")
        except Exception as e:
            print(f"Error calling VLM: {e}")

    cap.release()
    print("Vision Extraction complete!\n")
    return visual_contexts
