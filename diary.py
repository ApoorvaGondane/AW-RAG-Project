import torch
import torchaudio
import os 
import sys
from pathlib import Path
# --- THE ARCHITECT'S COMPATIBILITY PATCH ---
# Fix 1: Add missing list_audio_backends
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["ffmpeg"]

# Fix 2: Add missing ffmpeg_utils module placeholder
if not hasattr(torchaudio, 'utils'):
    # Create the utils module if it doesn't exist
    from types import ModuleType
    torchaudio.utils = ModuleType('utils')
    sys.modules['torchaudio.utils'] = torchaudio.utils

if not hasattr(torchaudio.utils, 'ffmpeg_utils'):
    from types import ModuleType
    ffmpeg_utils = ModuleType('ffmpeg_utils')
    # Add dummy functions that Pyannote/SpeechBrain expect
    ffmpeg_utils.get_audio_decoders = lambda: {"ffmpeg": "ffmpeg"}
    ffmpeg_utils.get_versions = lambda: {"libavcodec": (0, 0, 0)}
    torchaudio.utils.ffmpeg_utils = ffmpeg_utils
    sys.modules['torchaudio.utils.ffmpeg_utils'] = ffmpeg_utils

# Now proceed with your imports

from pyannote.audio import Pipeline, Model
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.pipelines.utils import PipelineModel # Secret import
from pyannote.audio.core.inference import Inference

# 1. Get the directory where THIS script is saved
# Use __file__ to make the path relative to the script itself
BASE_DIR = Path(__file__).resolve().parent
class DiarizationEngine:
    class DiarizationEngine:
        def __init__(self):
            os.environ["HF_HUB_OFFLINE"] = "1"
            
            
            # This will work on any Windows or Mac machine automatically
            base_path = BASE_DIR / "models_diarization"
            print(f"🚀 Performing Deep Binary Injection...")

            try:
                # 1. Load Segmentation
                seg_dir = os.path.join(base_path, "segmentation-3.0")
                # We manually load the architecture first
                segmentation_model = Model.from_pretrained(os.path.join(seg_dir, "config.yaml"))
                # Then we force-load the weights into the architecture
                seg_weights = torch.load(os.path.join(seg_dir, "pytorch_model.bin"), map_location="cpu")
                segmentation_model.load_state_dict(seg_weights)
                
                # 2. Load Embedding
                emb_dir = os.path.join(base_path, "wespeaker-voxceleb-resnet34-LM")
                embedding_model = Model.from_pretrained(os.path.join(emb_dir, "config.yaml"))
                emb_weights = torch.load(os.path.join(emb_dir, "pytorch_model.bin"), map_location="cpu")
                embedding_model.load_state_dict(emb_weights)

                # 3. Assemble Pipeline
                self.pipeline = SpeakerDiarization(
                    segmentation=segmentation_model,
                    embedding=embedding_model,
                    clustering="AgglomerativeClustering"
                )

                # 4. Standard Hyper-params
                self.pipeline.instantiate({
                    "clustering": {
                        "method": "centroid",
                        "min_clusters": 1,
                        "max_clusters": 10
                    },
                    "segmentation": {
                        "threshold": 0.5,
                        "min_duration_off": 0.0
                    }
                })

                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.pipeline.to(self.device)
                print(f"✅ [SUCCESS] Offline Engine active on {self.device}")
                
            except Exception as e:
                print(f"❌ Deep Load Error: {e}")
                print("\n🚨 CRITICAL CHECK: Is your pytorch_model.bin size > 100MB?")

    def process_file(self, audio_path):
        """
        Tests the diarization on a local audio file.
        Outputs: A list of speaker turns with start/end times.
        """
        print(f"--- Diarizing: {audio_path} ---")
        
        # ✅ NEW WAY (Pyannote 3.1+ / 4.x)
        output = self.pipeline(audio_path)

        # Access the underlying Annotation object
        diarization = output.speaker_diarization
        
        results = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            results.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker": speaker
            })
            print(f"[{turn.start:.2f}s - {turn.end:.2f}s] {speaker}")
            
        return results

# --- STANDALONE TEST BLOCK ---
if __name__ == "__main__":
    
    
    # Initialize the engine
    engine = DiarizationEngine()
    
    # Test with your sample audio file
    test_file = "melaningirl-hello-hello-pick-up-the-phone-463178.mp3"
    engine.process_file(test_file)