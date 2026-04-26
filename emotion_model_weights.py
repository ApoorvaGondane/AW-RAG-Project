from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

# Download and save locally
model = AutoModelForAudioClassification.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

model.save_pretrained("./emotion_model_weights")
feature_extractor.save_pretrained("./emotion_model_weights")
print("✅ Model downloaded and saved locally!")