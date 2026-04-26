from huggingface_hub import snapshot_download


MY_TOKEN = "" 

local_dir = "./models_diarization/"

# Download the models using the token
snapshot_download(repo_id="pyannote/speaker-diarization-3.1", 
                  local_dir=f"{local_dir}/speaker-diarization-3.1",
                  use_auth_token=MY_TOKEN)

snapshot_download(repo_id="pyannote/segmentation-3.0", 
                  local_dir=f"{local_dir}/segmentation-3.0",
                  use_auth_token=MY_TOKEN)

snapshot_download(repo_id="pyannote/wespeaker-voxceleb-resnet34-LM", 
                  local_dir=f"{local_dir}/wespeaker-voxceleb-resnet34-LM",
                  use_auth_token=MY_TOKEN)