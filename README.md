# AW-RAG: Improving Pedagogical Saliency in Retrieval-Augmented Generation through Acoustic Weighting and Edge-Based Inference

## 📌 Project Overview
AW-RAG is a multimodal research project that transitions traditional RAG systems from text-only retrieval to **context-aware retrieval**.
By integrating **Speech Emotion Recognition (SER)**, the system captures paralinguistic signals—such as tone, pitch, and prosody—to prioritize "pedagogically salient" content (like exam tips or key definitions) that standard ASR pipelines often discard.

### **Key Results**
* **Saliency Boost:** Acoustic weighting yields a **~20% increase** in retrieval accuracy for exam-critical content.
* **Priority Ranking:** Emphasized segments (Wp = 2.0) are automatically ranked higher in the vector store than monotone generic filler.

---

## 🏗️ System Architecture
The framework utilizes a parallel processing pipeline to fuse lexical and paralinguistic data streams.



1.  **ASR Engine:** Uses **OpenAI Whisper** for high-fidelity time-stamped transcription.
2.  **Emotion Engine (SER):** A custom module using **Wav2Vec2** to detect vocal arousal and assign priority weights ($W_p$).
3.  **Diarization:** **Pyannote 3.1** ensures accurate speaker tracking and temporal alignment.
4.  **Vector Store:** **ChromaDB** stores embedded chunks with associated acoustic metadata.
5.  **LLM Inference:** **Llama 3.2 (via Ollama)** generates detailed academic responses grounded in weighted context.

---

## 🛠️ Installation & Setup

### **1. Prerequisites**
* **Python 3.10+**
* **FFmpeg:** Required for audio decoding in the `diary.py` compatibility patch.
* **Ollama:** Ensure `llama3.2:3b` is pulled and running locally.

### **2. Environment Setup**
Clone the repository and install the dependencies:
```bash
git clone [https://github.com/username/AW-RAG-Project.git](https://github.com/username/AW-RAG-Project.git)
cd AW-RAG-Project
pip install -r requirements.txt
```
### **3. Local Model Preparation (Mandatory)**
To keep the repository lightweight, large weights are excluded. You must run the following script to download the Wav2Vec2 weights before starting the server:
```bash
python emotion_model_weights.py
```
This restores the emotion_model_weights/ folder required by the EmotionEngine.

## 🚀 Running the Application
Launch the FastAPI backend server:
```bash
uvicorn main:combined_app --host 0.0.0.0 --port 8000 --reload
```

Frontend: Open index.html in your browser to access the LiveSpeak dashboard.

Port Note: Mac users may need to use --port 8001 if port 8000 is occupied by AirPlay.

👥 Research Team
Apoorva Gondane (D006): Project Lead & Data Science Batch 2025-2027.

Krishnanshu Ramjiwala (D008): Lead Developer & System Integration.

Gaurav Ranadive (D021): Research Analyst & Performance Optimization.

Mentor: Prof. Archana Lakhe

HOD: Dr. Siba Panda
