import os
import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from pyannote.audio import Pipeline

def run_diarization(audio_file: str) -> list[dict]:
    """
    Runs pyannote.audio diarization pipeline on the given audio.
    Returns a list of dictionaries with structure:
    [{'start': 0.0, 'end': 2.5, 'speaker': 'SPEAKER_00'}, ...]
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable not set. Please set it in your .env file.")
        sys.exit(1)

    print("Loading Speaker Diarization model (Pyannote)...")
    
    try:
        # pyannote.audio >= 3.x uses 'token' instead of the deprecated 'use_auth_token'
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except Exception as e:
        print(f"Error loading Pyannote model: {e}")
        print("Please ensure you have accepted the agreements on Hugging Face for pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0")
        sys.exit(1)

    print(f"Running speaker diarization on {audio_file}... This may take a while.")
    
    # Completely sidestep torchaudio to prevent it from invoking broken codecs
    waveform_np, sample_rate = sf.read(audio_file, dtype='float32')
    if waveform_np.ndim == 1:
        waveform_np = waveform_np[:, np.newaxis]
    waveform = torch.from_numpy(waveform_np.T)
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    # In newer pyannote versions, the pipeline returns a DiarizeOutput object
    annotation = getattr(diarization, "speaker_diarization", diarization)

    speaker_segments = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        speaker_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    print(f"Diarization complete! Found {len(speaker_segments)} speech segments.")
    return speaker_segments

def align_speaker_with_segment(segment_start: float, segment_end: float, speaker_segments: list[dict]) -> str:
    """
    Given a whisper segment start and end, find the speaker who was most active during this time.
    """
    # Calculate overlap between whisper segment and each speaker segment
    speaker_overlaps = {}
    
    for spk_seg in speaker_segments:
        # Check if there is an overlap
        overlap_start = max(segment_start, spk_seg["start"])
        overlap_end = min(segment_end, spk_seg["end"])
        
        if overlap_start < overlap_end:
            overlap_duration = overlap_end - overlap_start
            speaker = spk_seg["speaker"]
            if speaker not in speaker_overlaps:
                speaker_overlaps[speaker] = 0.0
            speaker_overlaps[speaker] += overlap_duration
            
    if not speaker_overlaps:
        return "Unknown"
    
    # Return speaker with maximum overlap
    dominant_speaker = max(speaker_overlaps.items(), key=lambda x: x[1])[0]
    
    # Optional: Format the generic "SPEAKER_00" to "Speaker 1"
    spk_num = dominant_speaker.split("_")[-1]
    try:
        formatted_speaker = f"Speaker {int(spk_num) + 1}"
    except ValueError:
        formatted_speaker = dominant_speaker
        
    return formatted_speaker
