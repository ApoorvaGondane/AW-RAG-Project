
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, pipeline
import torch
import librosa
import numpy as np
import torch.nn.functional as F

class EmotionEngine:
    def __init__(self):
        model_path = "./emotion_model_weights"
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path, local_files_only=True)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        #We keep the baseline, but we make it "Sticky"
        self.base_peak_rms = None
        self.base_peak_centroid = None

    def detect_emotion(self, audio_path):
        y, sr = librosa.load(audio_path, sr=16000)
        if len(y) == 0: return "Standard"

        # PEAK ANALYSIS
        current_peak_rms = np.percentile(np.abs(y), 95)
        centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        current_peak_centroid = np.percentile(centroids, 95)

        if self.base_peak_rms is None:
            self.base_peak_rms = current_peak_rms
            self.base_peak_centroid = current_peak_centroid
            return "Standard"

        rms_ratio = current_peak_rms / (self.base_peak_rms + 1e-6)
        centroid_ratio = current_peak_centroid / (self.base_peak_centroid + 1e-6)

        print(f"🔍 Analysis: Vol Ratio {rms_ratio:.2f} | Tone Ratio {centroid_ratio:.2f}")

        # NEW THRESHOLD: 1.15 (This will catch your 1.18 'Pay Attention' segment)
        if rms_ratio > 1.15 or centroid_ratio > 1.10:
            print(f"🔥 [AW-RAG] Instructional Saliency Detected (Weight 2.0)")
            return "Emphasized" 
        
        return "Standard"
    

if __name__ == "__main__":
    ee = EmotionEngine()
    print(f"Offline Emotion Result: {ee.detect_emotion('melaningirl-hello-hello-pick-up-the-phone-463178.mp3')}")