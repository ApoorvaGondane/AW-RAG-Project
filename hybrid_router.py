import time
from engine import TranscriptionEngine

class HybridRouter:
    def __init__(self, threshold=-1.0):
        # Initialize the Edge Engine we just built
        self.edge_engine = TranscriptionEngine()
        self.threshold = threshold
        self.use_simulated_cloud = True # Set to False once you have an API key

    def get_transcription(self, audio_path):
        print(f"\n--- Processing Audio: {audio_path} ---")
        
        # 1. Try the Edge Engine first
        result = self.edge_engine.transcribe_local(audio_path)
        text = result['text']
        conf = result['confidence']

        # 2. Decision Logic: Is the confidence high enough?
        # A lightweight approach to reduce computation [cite: 32]
        if conf > self.threshold:
            print(f"✅ Edge Success (Conf: {conf:.2f})")
            return text, "Edge"
        else:
            print(f"⚠️ Low Confidence ({conf:.2f}). Switching to Cloud...")
            return self.call_cloud_sim(audio_path)

    def call_cloud_sim(self, audio_path):
        """
        Simulates the OpenAI Whisper-1 Cloud API.
        In a real scenario, this would send audio over a network.
        """
        print("☁️ Cloud API: Processing high-accuracy correction...")
        time.sleep(1.5) # Simulate network latency
        return "This is a high-accuracy correction from the Cloud.", "Cloud"

if __name__ == "__main__":
    router = HybridRouter(threshold=-0.4) # Setting high threshold to force a cloud test
    final_text, source = router.get_transcription("melaningirl-hello-hello-pick-up-the-phone-463178.mp3")
    print(f"FINAL RESULT [{source}]: {final_text}")