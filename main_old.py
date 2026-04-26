import os
import numpy as np
from fastapi import FastAPI
from fastapi_socketio import SocketManager
from hybrid_router import HybridRouter

app = FastAPI()
sio = SocketManager(app=app)

# Initialize the Brain (Member 1's Hybrid System)
# We set threshold back to -1.0 for balanced performance
router = HybridRouter(threshold=-1.0)

@app.get("/")
async def health_check():
    return {"status": "LiveSpeak Server is Online"}

@sio.on('audio_stream')
async def handle_audio(sid, data):
    """
    Receives live audio bytes from Member 3 (Frontend).
    """
    try:
        # 1. Convert incoming bytes to a temporary wav-like buffer
        # In a real build, we save to a small 'buffer.wav' for the router
        temp_filename = f"temp_{sid}.wav"
        with open(temp_filename, "wb") as f:
            f.write(data)

        # 2. Pass to the Hybrid Router
        text, source = router.get_transcription(temp_filename)

        # 3. Send the result back to the specific student
        await sio.emit('caption_update', {
            'text': text,
            'source': source,
            'is_final': True
        }, to=sid)
        
        # Cleanup temporary audio
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    except Exception as e:
        print(f"Error in streaming pipeline: {e}")

if __name__ == "__main__":
    import uvicorn
    # Member 1 starts the server here
    uvicorn.run(app, host="0.0.0.0", port=8000)