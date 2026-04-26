import numpy as np
from faster_whisper import WhisperModel

class TranscriptionEngine:
    def __init__(self, model_size="tiny.en", device="cpu", compute_type="int8"):
        """
        Initializes the Edge Engine.
        Using 'int8' is a research-backed method to reduce CPU computation 
        on embedded or edge devices.
        """
        print(f"Loading Edge Model ({model_size})...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_local(self, audio_data):
        """
        Processes audio data locally on the 'Edge'.
        Uses beam_size=1 for 'Ultra-Low Latency' as discussed in your 
        literature review.
        """
        # Transcribe audio (accepts file path or numpy array)
        segments, info = self.model.transcribe(audio_data, beam_size=1)
        
        full_text = ""
        min_confidence = 0.0
        
        # We extract text and the 'avg_logprob' for our Hybrid Router logic
        for segment in segments:
            full_text += segment.text
            min_confidence = min(min_confidence, segment.avg_logprob)
            
        return {
            "text": full_text.strip(),
            "confidence": min_confidence,
            "language": info.language
        }

# For standalone testing of the module
if __name__ == "__main__":
    engine = TranscriptionEngine()
    # Replace with your actual test file name
    result = engine.transcribe_local("melaningirl-hello-hello-pick-up-the-phone-463178.mp3")
    print(f"Result: {result['text']} | Confidence: {result['confidence']}")