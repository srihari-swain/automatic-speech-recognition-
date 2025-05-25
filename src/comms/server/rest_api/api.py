import io
import logging
import soundfile as sf
import time
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.speech_recognizer.speech_recognizer import SpeechRecognizer
from src.configs.config_loader import read_base_config

# Load config
config = read_base_config()

# Initialize speech recognizer
speech_recognizer = SpeechRecognizer()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=config["title"],
    description=config["description"],
    version=config["version"],
    docs_url=config["docs_url"],
    redoc_url=config["redoc_url"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config["allow_origins"],
    allow_credentials=config["allow_credentials"],
    allow_methods=config["allow_methods"],
    allow_headers=config["allow_headers"]
)

class TranscriptionResponse(BaseModel):
    text: str
    confidence: Optional[float] = None
    processing_time: float
    duration: float

def validate_audio_file(file: UploadFile, audio, duration, max_duration=60.0):
    # File type check
    if not file.filename.lower().endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only .wav files are supported.")
    # Duration check
    if duration > max_duration:
        raise HTTPException(status_code=400, detail=f"Audio is too long (max {max_duration} seconds).")
    if duration < 0.1:
        raise HTTPException(status_code=400, detail="Audio is too short (min 0.1 seconds).")

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe an audio file to text.

    Accepts:
        - .wav files, any sample rate, mono or stereo, up to 60s

    Returns:
        - text: Transcribed text
        - confidence: Confidence score (if available)
        - processing_time: Time taken for processing
        - duration: Duration of the audio file

    Async compatibility:
        - This endpoint is async, but the underlying inference pipeline is synchronous (CPU-bound, ONNX/NumPy/SoundFile).
        - For true async inference, use a model/pipeline that supports asyncio or run the sync code in a threadpool.
    """
    try:
        # Read the entire file into memory
        file_content = await file.read()
        audio_bytes = io.BytesIO(file_content)

        # Validate and get audio info
        try:
            audio, sr = sf.read(audio_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read audio file: {str(e)}")
        duration = len(audio) / sr if sr else 0.0

        validate_audio_file(file, audio, duration)

        # Reset pointer for inference
        audio_bytes.seek(0)

        # Inference: SpeechRecognizer handles resampling if needed
        start_time = time.time()
        transcription, confidence, _ = speech_recognizer.transcribe(audio_bytes)
        processing_time = time.time() - start_time

        return TranscriptionResponse(
            text=transcription,
            confidence=confidence,
            processing_time=processing_time,
            duration=duration
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to transcribe audio: {str(e)}"
        )
