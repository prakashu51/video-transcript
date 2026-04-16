import os
import sys
import time
import queue
import threading
import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
import warnings

from config import (
    LIVE_SAMPLE_RATE,
    LIVE_VAD_CHUNK_MS,
    LIVE_SILENCE_THRESHOLD,
    LIVE_SPEECH_THRESHOLD,
    LIVE_PRE_ROLL_MS,
    LIVE_MIN_SPEECH_DURATION,
    LIVE_DEFAULT_MODEL,
    resolve_device_and_compute_type,
)

warnings.filterwarnings("ignore", category=FutureWarning)

class LiveTranscriber:
    """
    Captures microphone audio, uses Silero VAD to detect speech segments,
    and runs faster-whisper on each segment for real-time transcription.
    """
    def __init__(self, model_size=LIVE_DEFAULT_MODEL, language=None, device_pref="auto", task="transcribe", enable_emotion=False):
        # Import here to avoid slow startup if just viewing help
        from faster_whisper import WhisperModel
        
        self.sample_rate = LIVE_SAMPLE_RATE
        self.language = language
        self.task = task
        self.enable_emotion = enable_emotion
        self.emotion_analyzer = None
        
        # Audio and VAD parameters
        self.chunk_size = int(self.sample_rate * LIVE_VAD_CHUNK_MS / 1000)
        self.silence_chunks = int(LIVE_SILENCE_THRESHOLD * 1000 / LIVE_VAD_CHUNK_MS)
        self.pre_roll_chunks = int(LIVE_PRE_ROLL_MS / LIVE_VAD_CHUNK_MS)
        self.min_speech_chunks = int(LIVE_MIN_SPEECH_DURATION * 1000 / LIVE_VAD_CHUNK_MS)
        
        print(f"Loading Whisper model '{model_size}'...")
        selected_device, compute_type, _ = resolve_device_and_compute_type(device_pref)
        try:
            self.model = WhisperModel(model_size, device=selected_device, compute_type=compute_type)
        except Exception as e:
            if selected_device == "cuda":
                print("CUDA failed, falling back to CPU")
                self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            else:
                raise e
                
        if self.enable_emotion:
            from emotion_analyzer import EmotionAnalyzer
            self.emotion_analyzer = EmotionAnalyzer(device=selected_device)
            
        print("Loading Silero VAD...")
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        self.get_speech_timestamps = utils[0]
        
        # State
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.transcript_history = []
        
        # Callbacks
        self.on_segment_ready = None

    def _audio_callback(self, indata, frames, time, status):
        """Called by sounddevice for each audio chunk."""
        # Never print inside the audio callback, it blocks and causes input overflow!
        # Convert to float32 numpy array
        self.audio_queue.put(indata.copy().flatten())

    def start_listening(self):
        """Starts the microphone and processing loop."""
        self.is_recording = True
        
        # Start capture thread
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=self.chunk_size,
            callback=self._audio_callback
        )
        self.stream.start()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_audio)
        self.process_thread.start()

    def stop_listening(self):
        """Stops recording and processing."""
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        if hasattr(self, 'process_thread'):
            self.process_thread.join()

    def _process_audio(self):
        """Main loop: Detects speech via VAD and triggers Whisper transcription."""
        ring_buffer = []  # stores audio chunks for pre-roll
        speech_buffer = []  # stores active speech chunks
        
        silence_counter = 0
        is_speaking = False
        
        print("Ready! Start speaking...")
        
        while self.is_recording or not self.audio_queue.empty():
            try:
                # Wait for next audio chunk
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Maintain pre-roll ring buffer
            ring_buffer.append(chunk)
            if len(ring_buffer) > self.pre_roll_chunks:
                ring_buffer.pop(0)

            # Check VAD
            tensor_chunk = torch.from_numpy(chunk)
            speech_prob = self.vad_model(tensor_chunk, self.sample_rate).item()
            
            if speech_prob > LIVE_SPEECH_THRESHOLD:
                # Speech detected
                if not is_speaking:
                    is_speaking = True
                    # Initialize speech buffer with pre-roll
                    speech_buffer = []
                    for b in ring_buffer:
                        speech_buffer.extend(b)
                else:
                    speech_buffer.extend(chunk)
                silence_counter = 0
                
            else:
                # Silence
                if is_speaking:
                    speech_buffer.extend(chunk)
                    silence_counter += 1
                    
                    if silence_counter >= self.silence_chunks:
                        # End of speech segment
                        is_speaking = False
                        
                        # Only transcribe if the speech was long enough
                        if len(speech_buffer) >= self.min_speech_chunks * self.chunk_size:
                            audio_data = np.array(speech_buffer, dtype=np.float32)
                            self._transcribe_segment(audio_data)
                            
                        speech_buffer = []
                        silence_counter = 0

    def _transcribe_segment(self, audio_data: np.ndarray):
        """Runs Whisper on a specific audio segment."""
        # Note: faster-whisper accepts float32 numpy arrays directly
        segments, info = self.model.transcribe(
            audio_data,
            beam_size=5,
            language=self.language,
            task=self.task,
            condition_on_previous_text=False  # Crucial for live modes to prevent hallucination loops
        )
        
        segment_text = "".join([segment.text for segment in segments]).strip()
        
        if segment_text:
            emotion_tag = ""
            if self.enable_emotion and self.emotion_analyzer is not None:
                # audio_data is already float32, which the pipeline accepts
                emotion = self.emotion_analyzer.detect_emotion(audio_data)
                if emotion:
                    emotion_tag = f"[{emotion}] "

            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            entry = f"[{timestamp}] {emotion_tag}{segment_text}"
            self.transcript_history.append(entry)
            
            if self.on_segment_ready:
                self.on_segment_ready(entry)
            else:
                print(entry)

    def save_session(self, output_path=None) -> str:
        """Saves the transcribed session to a file."""
        if not self.transcript_history:
            return ""
            
        if output_path:
            path = Path(output_path)
        else:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            lang_code = f".{self.language}" if self.language else ""
            if self.task == "translate":
                lang_code = f"{lang_code}.en"
            path = Path(f"live_session_{stamp}{lang_code}.txt")
            
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.transcript_history))
            
        return str(path)

# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live Real-time Transcription")
    parser.add_argument("--model", type=str, default=LIVE_DEFAULT_MODEL, help="Whisper model size")
    parser.add_argument("--lang", type=str, default=None, help="Source language code (e.g. 'zh', 'en')")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda)")
    parser.add_argument("--translate", action="store_true", help="Translate to English in real-time")
    parser.add_argument("--emotion", action="store_true", help="Enable Real-time Tone/Emotion detection")
    parser.add_argument("--save", action="store_true", help="Auto-save the session to a text file on exit")
    
    args = parser.parse_args()
    
    task = "translate" if args.translate else "transcribe"
    
    transcriber = LiveTranscriber(
        model_size=args.model,
        language=args.lang,
        device_pref=args.device,
        task=task,
        enable_emotion=args.emotion
    )
    
    transcriber.start_listening()
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
        transcriber.stop_listening()
        
        if args.save:
            path = transcriber.save_session()
            if path:
                print(f"Session saved to: {path}")
