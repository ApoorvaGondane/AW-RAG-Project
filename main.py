import socketio
import uvicorn
import os
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma  # Updated from langchain_community
from langchain_community.embeddings import DeterministicFakeEmbedding 
from langchain_ollama import OllamaLLM  # Updated from langchain_community.llms
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    import audioop
except ImportError:
    import audioop_lts as audioop
    import sys
    sys.modules['audioop'] = audioop
from pydub import AudioSegment
import io
import warnings
# Filter out the SpeechBrain redirect warning
warnings.filterwarnings("ignore", message="Module 'speechbrain.pretrained' was deprecated")
from deep_translator import GoogleTranslator
import warnings
import logging
# Initialize the Sentence-Aware Buffer (Global)
from sentence_buffer import SentenceAwareBuffer # Assuming you saved the previous snippet as sentence_buffer.py
sentence_buffer = SentenceAwareBuffer(weight_threshold=1.5)
# Silence those specific math warnings
warnings.filterwarnings("ignore", message="std(): degrees of freedom is <= 0")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
# 1. SETUP SOCKET.IO & APP (Must be first)
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*',max_http_buffer_size=100 * 1024 * 1024, ping_timeout=60,ping_interval=25)
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],allow_credentials=True,)
combined_app = socketio.ASGIApp(sio, app)

# 2. INITIALIZE MEMBER 1 & 2 ENGINES
from engine import TranscriptionEngine
from hybrid_router import HybridRouter
from diary import DiarizationEngine
from EmotionEngine import EmotionEngine

router = HybridRouter(threshold=-1.0)
# Remember to update your token!
diarizer = DiarizationEngine()

emotion_model = EmotionEngine()
# 3. RAG SETUP (Member 3)
vectorstore = Chroma(embedding_function=DeterministicFakeEmbedding(size=1536), persist_directory="./chroma_db")
llm = OllamaLLM(model="llama3.2:3b")

def add_to_memory(text, emotion, intensity_score=1.0):
    if not text.strip():
        return
        
    # 1. Calculate Priority Weight for your research contribution
    # priority = 2.0 if emotion in ["Happy", "Angry", "Fear", "Surprise"] else 1.0
    priority = intensity_score
    # Updated Settings for Educational Saliency
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # Increased from 500 to capture full definitions
        chunk_overlap=200,    # Increased from 50 to ensure no sentence is cut in half
        separators=["\n\n", "\n", ".", "!", "?", " "] # Priority given to sentence ends
    )
    
    # 2. Create documents with metadata BUILT-IN
    # This avoids the "multiple values for argument 'metadatas'" error
    docs = text_splitter.create_documents(
        texts=[text], 
        metadatas=[{"emotion": emotion, "priority": priority, "intensity": intensity_score}]
    )
    
    # 3. Add to store without passing separate metadatas list
    vectorstore.add_documents(docs)
    print(f"📦 AW-RAG: Indexed with weight {priority} ({emotion})")


# Create a global or session-based buffer
audio_buffer_bytes = b""
first_header_bytes = b""
@sio.on('audio-chunk')
async def handle_audio(sid, data):
    global audio_buffer_bytes, first_header_bytes
    
    # Capture the header from the very first chunk of the session
    if not first_header_bytes and len(data) > 1000:
        first_header_bytes = data[:500] # Usually the first 500 bytes contain the EBML header
    # FIX: Ensure data is treated as bytes
    if isinstance(data, str):
        data = data.encode('utf-8')
    audio_buffer_bytes += data
    if len(audio_buffer_bytes) < 160000:
        return
# 2. ATOMIC CAPTURE: Move data to a local variable and clear global immediately
    # This prevents new incoming data from being wiped during the 'reset'
    processing_data = audio_buffer_bytes
    audio_buffer_bytes = b"" 

    try:
        temp_raw = f"raw_{sid}.webm"
        with open(temp_raw, "wb") as f:
            # RESEARCH FIX: Inject the header if it's missing from this chunk
            if not processing_data.startswith(b'\x1a\x45\xdf\xa3'): # Standard EBML/WebM start
                f.write(first_header_bytes)
            f.write(processing_data)
        

        try:
            audio = AudioSegment.from_file(temp_raw, format="webm")
        except Exception as decode_err:
            # THIS IS THE CRITICAL LOG: Why is the 2nd chunk failing?
            print(f"❌ DECODE ERROR on chunk: {decode_err}")
            # Put data back so we don't lose it
            audio_buffer_bytes = processing_data + audio_buffer_bytes
            return
        # Success: Clear global buffer to prepare for next stream chunk
        if os.path.exists(temp_raw):
            os.remove(temp_raw)
        # audio_buffer_bytes = b"" 

        # 3. STANDARDIZE & EXPORT
        audio = audio.set_frame_rate(16000).set_channels(1)
        temp_file = f"temp_{sid}.wav"
        audio.export(temp_file, format="wav")

        # 4. ENGINE PIPELINE (Member 1 & 2)
        # ASR Transcription
        text, source = router.get_transcription(temp_file)
        print(f"--- AI Result: [{text}] ---")
        # Diarization (Who is speaking?)
        current_speaker = "Unknown"
        if audio.duration_seconds > 1.5: # Safety check for PyTorch math
            try:
                speaker_info = diarizer.process_file(temp_file)
                if speaker_info:
                    current_speaker = speaker_info[-1]['speaker']
            except Exception:
                pass

        # Emotion Detection (How are they speaking?)
        current_emotion = "Neutral"
        
        try:
            current_emotion = emotion_model.detect_emotion(temp_file)
        except Exception as e:
            print(f"Emotion detection failed: {e}")

        # 5. TRANSLATION (Member 1 Middleware)
        session = await sio.get_session(sid)
        target_lang_code = session.get('language', 'en')
        translated_text = text

        if target_lang_code != 'en' and text.strip():
            try:
                translated_text = GoogleTranslator(source='auto', target=target_lang_code).translate(text)
            except Exception as e:
                print(f"Translation Error: {e}")

        # 6. MEMORY & EMIT
        if text and text.strip():
            # Add English version to RAG for consistent bot memory
            #add_to_memory(text, current_emotion, 1.0)
            # --- NEW: SENTENCE-AWARE BUFFERING LOGIC ---
            # Instead of adding raw text, we process it through the buffer
            # We pass the current_emotion as the 'weight' for this chunk
            
            # Map emotion string to numerical weight for the buffer
            # High arousal/emphasis emotions = 2.0, others = 1.0
            priority_emotions = ['Happy','happy','surprise','angry','fear','Emphasized', 'Surprise', 'Angry', 'Fear']
            weight = 2.0 if current_emotion in priority_emotions else 1.0
            
            # Process the chunk through our buffer
            completed_sentences = sentence_buffer.process_chunk(text, weight)
            # Only add to RAG memory if a full sentence is completed
            for item in completed_sentences:
                # item['text'] is the full sentence, item['weight'] is the highest weight found
                add_to_memory(item['text'], current_emotion, item['weight'])
                print(f"📦 Sent to RAG Database: [{item['text']}] (Weight: {item['weight']})")
            # -------------------------------------------
            print(f"DEBUG: [{source}] {text}")
            if translated_text != text:
                print(f"🌍 Translated: {translated_text}")

            await sio.emit('caption_update', {
                'text': text.strip(),
                'translated_text': translated_text,
                'speaker': current_speaker,
                'emotion': current_emotion,
                'source': source,
                'weight':item['weight']
            })

        # 7. CLEANUP
        if os.path.exists(temp_file):
            os.remove(temp_file)

    except Exception as e:
        print(f"❌ Handle Audio Error: {e}")
        # If decode fails, keep buffer for next chunk
        if len(audio_buffer_bytes) > 500000:
             audio_buffer_bytes = b"" 
        return

@sio.on('connect')
async def handle_connect(sid, environ):
    global audio_buffer_bytes
    print(f"🚀 New Session Started: {sid}. Clearing buffers.")
    audio_buffer_bytes = b"" # Reset memory for the new speaker

@sio.on('reset_buffer')
async def reset_buffer(sid):
    global audio_buffer_bytes
    audio_buffer_bytes = b""
    print(f"🧹 Buffer Cleared for new recording session: {sid}")

@sio.on('set_language')
async def set_language(sid, data):
    lang_code = data.get('language', 'en')
    # Save the choice into the user's socket session
    await sio.save_session(sid, {'language': lang_code})
    print(f"🌐 Language set to {lang_code} for session {sid}")

@sio.on('flush_buffer')
async def flush_buffer(sid):
    global audio_buffer_bytes
    
    if not audio_buffer_bytes or len(audio_buffer_bytes) < 5000:
        audio_buffer_bytes = b""
        return

    try:
        print(f"📥 Final Flush: Processing {len(audio_buffer_bytes)} bytes...")
        audio_stream = io.BytesIO(audio_buffer_bytes)
        
        try:
            audio = AudioSegment.from_file(audio_stream)
        except Exception:
            try:
                print("🔄 Flush: Raw recovery...")
                audio = AudioSegment.from_raw(audio_stream, sample_width=2, frame_rate=16000, channels=1)
            except Exception:
                audio_buffer_bytes = b""
                return

        audio_buffer_bytes = b""
        audio = audio.set_frame_rate(16000).set_channels(1)
        temp_file = f"temp_flush_{sid}.wav"
        audio.export(temp_file, format="wav")

        # --- SYNCED ENGINES START ---
        # 1. Transcription
        text, source = router.get_transcription(temp_file)
        
        # 2. Diarization
        current_speaker = "Speaker"
        if audio.duration_seconds > 1.5:
            try:
                speaker_info = diarizer.process_file(temp_file)
                if speaker_info:
                    current_speaker = speaker_info[-1]['speaker']
            except: pass

        # 3. Emotion Detection (The missing piece!)
        current_emotion = "Neutral"
        try:
            current_emotion = emotion_model.detect_emotion(temp_file)
        except Exception as e:
            print(f"Flush Emotion failed: {e}")
        # --- SYNCED ENGINES END ---

        if text and text.strip():
            # Translation logic
            session = await sio.get_session(sid)
            target_lang_code = session.get('language', 'en')
            translated_text = text
            if target_lang_code != 'en':
                try:
                    translated_text = GoogleTranslator(source='auto', target=target_lang_code).translate(text)
                except: pass

            # FINAL EMIT: No more hardcoded "Neutral" or "Speaker"
            await sio.emit('caption_update', {
                'text': text.strip(),
                'translated_text': translated_text,
                'speaker': current_speaker,
                'emotion': current_emotion,
                'source': f"{source} (Final)"
            })
            
            # RESEARCH SYNC: Correctly passing the detected emotion and weight
            add_to_memory(text, current_emotion, 1.0)
            print(f"✅ Final Flush Success: [{current_emotion}] {text}")

        if os.path.exists(temp_file):
            os.remove(temp_file)

    except Exception as e:
        print(f"❌ Flush Error: {e}")
        audio_buffer_bytes = b""

@app.get("/ask")
async def answer_question(q: str):
    # 1. RETRIEVAL & RE-RANKING (The AW-RAG Core)
    # Get a larger pool (k=10) to allow weighted items to "climb" to the top
    results = vectorstore.similarity_search_with_score(q, k=10)
    
    if not results:
        print(f"❌ RAG Error: No context found for query: '{q}'")
        return {"answer": "I haven't processed any lecture notes yet. Please speak into the mic first!"}

    # 2. RE-RANKING LOGIC
    # FinalScore = Distance / Priority (Lower distance is better)
    reranked = []
    for doc, score in results:
        # Get priority (2.0 for emphasis, 1.0 for neutral)
        priority = doc.metadata.get("priority", 1.0)
        weighted_score = score / priority 
        reranked.append((doc, weighted_score))
    
    # Sort by the new weighted score and take top 5
    reranked.sort(key=lambda x: x[1])
    top_docs = [item[0] for item in reranked[:5]]
    
    # 3. CONTEXT ASSEMBLY
    # Join with clear separators to help the LLM distinguish between chunks
    context = "\n---\n".join([d.page_content for d in top_docs])
    print(f"🔍 AW-RAG: Found {len(top_docs)} segments. Re-ranked using Acoustic Priority.")

    # 4. REFINED ACADEMIC PROMPT
    # This prompt tells the LLM to prioritize emphasized concepts and provide detail.
    prompt = f"""
    You are an Advanced Academic Research Assistant. Your goal is to help a student understand a lecture.
    Below is the context retrieved from the live transcription. 
    NOTE: Some segments have been prioritized because the lecturer emphasized them vocally.

    ### LECTURE CONTEXT:
    {context}

    ### INSTRUCTIONS:
    1. Use the provided context to provide a comprehensive and detailed explanation.
    2. If a specific term (like "Transformer" or "Self-Attention") was mentioned, explain its definition, purpose, and how it relates to modern AI based on the context.
    3. Do not give one-word answers. Provide at least 3-4 sentences of detail.
    4. If the information is not in the context, state that you are waiting for the lecturer to cover that specific detail.

    ### STUDENT QUESTION: 
    {q}

    ### DETAILED ACADEMIC ANSWER:
    """
    
    # 5. INFERENCE
    print(f"DEBUG: Querying Llama 3.2 with context length: {len(context)}")
    answer = llm.invoke(prompt)
    
    return {"answer": answer}

import os

# Create an 'uploads' directory if it doesn't exist
if not os.path.exists("uploads"):
    os.makedirs("uploads")

@sio.on('upload-audio-file')
async def handle_file_upload(sid, data):
    try:
        # data is usually a dict: {'fileName': '...', 'data': b'binary...'}
        file_name = data.get('fileName', 'uploaded_lecture.wav')
        file_bytes = data.get('data')

        # FIX: Some browsers send the buffer wrapped in a string or blob
        if not isinstance(file_bytes, bytes):
            print("⚠️ Warning: Received non-byte data, attempting conversion...")
            # If it's a list or other format, convert to bytes
            file_bytes = bytes(file_bytes)

        temp_path = os.path.join("uploads", file_name)
        
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
        
        print(f"✅ File Received: {file_name} ({len(file_bytes)} bytes)")
        
        # Trigger the STABLE demo function that uses Pydub (No EBML errors here!)
        asyncio.create_task(run_stable_file_demo(temp_path,sid,target_lang_code="en"))
        
        await sio.emit('processing-complete', {'msg': f"Processing {file_name}..."}, to=sid)
        
    except Exception as e:
        print(f"❌ Upload Error: {e}")
        await sio.emit('error', {'msg': str(e)}, to=sid)
from pydub import AudioSegment

async def run_stable_file_demo(file_path,sid,target_lang_code="en"):
    """
    Bypasses SocketIO/WebM issues by reading a local WAV file 
    and feeding it directly to the processing pipeline.
    """
    print(f"🚀 [OFFLINE DEMO MODE] Processing: {file_path}")
    # 1. Fetch the user's selected language from their session
    session = await sio.get_session(sid)
    target_lang_code = session.get('language', 'en') 
    
    print(f"🌐 Target Language from UI: {target_lang_code}")
    # 1. Load the clean file
    # Ensure your demo file is a .wav for maximum stability
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    
    # 2. Define chunk size (e.g., 10 seconds of audio)
    chunk_length_ms = 10000 
    
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        temp_chunk_name = "demo_chunk.wav"
        chunk.export(temp_chunk_name, format="wav")
        
        print(f"📦 Processing Segment {i//1000}s to {(i+chunk_length_ms)//1000}s...")

        # 3. DIRECT PIPELINE CALLS (Mirroring your handle_audio logic)
        try:
            # ASR
            text, _ = router.get_transcription(temp_chunk_name)
            # 2. Translation Logic (Using Ollama or a dedicated Translator)
            translated_text = ""
            if target_lang_code != 'en' and text.strip():
                try:
                    # If you have a translation function in your router/utils:
                    # translated_text = router.translate_text(text, target_lang="Hindi") 
                    
                    # Manual Ollama Call for Translation (Fast & Offline)
                    # LangChain's .generate expects a LIST of strings
                    res = llm.generate(prompts=[f"Translate to {target_lang_code}: {text}"])
                    translated_text = res.generations[0][0].text.strip()
                except Exception as e:
                    print(f"⚠️ Translation failed: {e}")
                    translated_text = text # Fallback to original text
            # Diarization (Who is speaking?)
            current_speaker = "Speaker"
            if audio.duration_seconds > 1.5: # Safety check for PyTorch math
                try:
                    speaker_info = diarizer.process_file(temp_chunk_name)
                    if speaker_info:
                        current_speaker = speaker_info[-1]['speaker']
                except Exception:
                    pass
            # Emotion (AW-RAG Core)
            current_emotion = emotion_model.detect_emotion(temp_chunk_name)
            
            # Priority Weight Calculation
            priority_emotions = ['Happy','happy','surprise','angry','fear','Emphasized', 'Surprise', 'Angry', 'Fear']
            weight = 2.0 if current_emotion in priority_emotions else 1.0
            
            if text.strip():
                # Process through your Sentence-Aware Buffer
                completed_sentences = sentence_buffer.process_chunk(text, weight)
                
                for item in completed_sentences:
                    add_to_memory(item['text'], current_emotion, item['weight'])
                    print(f"✅ Indexed: {item['text']} (Weight: {item['weight']})")
                    # ADD THIS LINE TO SHOW IN UI:
                    await sio.emit('caption_update', {
                        'text': item['text'],
                        'translated_text': translated_text,
                        'speaker': current_speaker,
                        'emotion': current_emotion,
                        'weight':float(item['weight']),
                        'source':_
                        
                    })
        except Exception as e:
            print(f"❌ Error in demo chunk: {e}")
            
    print("🎯 Demo Processing Complete. You can now ask questions!")


if __name__ == "__main__":
    uvicorn.run(combined_app, host="0.0.0.0", port=8000)