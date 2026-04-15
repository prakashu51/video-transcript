"""
Extract audio from a video file and save as 16kHz mono WAV.
Uses faster-whisper's bundled ffmpeg, so no system ffmpeg installation needed.

Usage:
    python extract_audio.py input.mp4
    python extract_audio.py input.mp4 output.wav
"""
import sys
from pathlib import Path
from faster_whisper.audio import decode_audio
import soundfile as sf

def extract_audio(input_file: str, output_file: str = None) -> str:
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"Error: file not found: {input_file}")
        sys.exit(1)
    
    if output_file is None:
        output_file = str(input_path.with_suffix(".wav"))
    
    print(f"Extracting audio from: {input_file}")
    print(f"Output: {output_file}")
    
    audio = decode_audio(input_file, sampling_rate=16000)
    sf.write(output_file, audio, 16000)
    
    duration = len(audio) / 16000
    print(f"Done! Duration: {duration:.1f}s")
    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_audio.py <input_video> [output_wav]")
        sys.exit(1)
    
    input_f = sys.argv[1]
    output_f = sys.argv[2] if len(sys.argv) > 2 else None
    extract_audio(input_f, output_f)
