"""
Modular Emotion Analyzer
Handles loading and inference of Speech Emotion Recognition (SER) models.
"""
import numpy as np
import warnings
from transformers import pipeline

from config import EMOTION_MODEL, LIVE_SAMPLE_RATE

warnings.filterwarnings("ignore", category=FutureWarning)

class EmotionAnalyzer:
    def __init__(self, device="cpu"):
        """
        Initializes the huggingface audio-classification pipeline for emotion recognition.
        """
        print(f"Loading Emotion Model '{EMOTION_MODEL}'...")
        # Pipeline handles loading the feature extractor and the model
        device_id = 0 if device == "cuda" else -1
        try:
            self.classifier = pipeline(
                "audio-classification", 
                model=EMOTION_MODEL,
                device=device_id
            )
        except Exception as e:
            if device == "cuda":
                print("CUDA failed for Emotion Model, falling back to CPU")
                self.classifier = pipeline(
                    "audio-classification", 
                    model=EMOTION_MODEL,
                    device=-1
                )
            else:
                raise e

    def detect_emotion(self, audio_data: np.ndarray) -> str:
        """
        Analyzes the float32 audio numpy array and returns the predicted emotion label.
        """
        if len(audio_data) == 0:
            return ""

        # The pipeline accepts raw numpy arrays (must be 16kHz float32, which it is)
        results = self.classifier(audio_data)
        
        # results is typically a list of dicts: [{'score': 0.8, 'label': 'happy'}, ...]
        # Grab the highest confidence label
        if results:
            best_match = results[0]
            label = best_match['label'].title() # e.g. "Happy"
            return label
            
        return "Neutral"
