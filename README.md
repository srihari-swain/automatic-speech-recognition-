# Automatic Speech Recognition (ASR) API

## Author Information

- Developer: Srihari Swain
- Email: srihariswain2001@gmail.com
- GitHub: https://github.com/srihari-swain

## Project Timeline

- Development Time: Approximately 15 hours

## Overview

This project provides a robust FastAPI-based REST API for transcribing audio files to text. The system leverages ONNX Runtime for efficient speech recognition and is designed with clean architecture principles, separating concerns between audio handling, speech processing, and API endpoints.

The service allows users to:
- Upload audio files in various formats (.wav)
- Receive accurate text transcriptions
- Get confidence scores and processing metrics

---

## Model Conversion: From .nemo to ONNX

This project uses an ONNX model for fast inference. If you have a `.nemo` model (from NVIDIA NeMo), you can convert it to ONNX with the following steps:

1. **Install NeMo and dependencies:**
   ```bash
   pip install nemo_toolkit[asr] onnx onnxruntime
   ```

2. **Export the .nemo model to ONNX:**
   ```python
   import nemo.collections.asr as nemo_asr
   asr_model = nemo_asr.models.EncDecCTCModel.restore_from('stt_hi_conformer_ctc_medium.nemo')
   asr_model.export('stt_hi_conformer_ctc_medium.onnx')
   ```

3. **Move the ONNX file:**
   - Place the resulting `stt_hi_conformer_ctc_medium.onnx` in your project's `src/models/` directory (or update your config accordingly).

**References:**
- [NeMo Export Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/intro.html)

---

## Why ONNX Model Inference is Not Fully Async

While the FastAPI endpoints are async, ONNX Runtime and NumPy are CPU-bound and synchronous libraries. True async (non-blocking) inference would require ONNX Runtime to support `await` natively or for the model to run in a fully async event loop (which is not currently the case for most Python ML frameworks).

### Technical Details:
- **ONNXRuntime**: The Python API for ONNX Runtime is synchronous and releases the GIL only for heavy computation. It does not provide an `await`-able interface.
- **NumPy/SoundFile**: Preprocessing steps (e.g., reading and resampling audio) are also synchronous and CPU-bound.
- **Async in FastAPI**: To avoid blocking, the recommended pattern is to offload CPU-bound tasks to a thread pool or process pool using `run_in_executor`, allowing FastAPI to handle other requests concurrently.

### Example (in code):
```python
import asyncio
loop = asyncio.get_event_loop()
result = await loop.run_in_executor(None, speech_recognizer.transcribe, audio_bytes)
```

### Summary:
- The endpoint is async-compatible and will not block the FastAPI event loop.
- The actual model inference and preprocessing are performed synchronously in a thread pool.
- This is a standard pattern in Python for integrating synchronous ML libraries with async web frameworks.

## Project Structure

```
├── __init__.py
└── src
    ├── comms
    │   └── server
    │       └── rest_api
    │           └── api.py
    ├── configs
    │   └── config.json
    ├── main.py
    ├── models
    │   └── stt_hi_conformer_ctc_medium.onnx
    ├── speech_recognizer
    │   └── speech_recognizer.py
    └── vocab
        └── hi_vocab.txt
```

## API Endpoints

| Endpoint | Method | Description | Parameters |
| --- | --- | --- | --- |
| `/transcribe` | POST | Transcribe an audio file to text | `file` (audio file) |

## Configuration

The API loads configuration from a JSON file located at `src/configs/config.json`. This file includes important settings for the application:

```json
{
    "title": "ASR Service",
    "description": "Automatic Speech Recognition API",
    "version": "0.0.1",
    "docs_url": "/docs",
    "redoc_url": "/redoc",
    "model_path": "src/models/stt_hi_conformer_ctc_medium.onnx",
    "vocab_path": "src/vocab/hi_vocab.txt",
    "app": "src.comms.server.rest_api.api:app",
    "host": "0.0.0.0",
    "port": 8000,
    "reload": true,
    "workers": 1,
    "allow_origins": ["*"],
    "allow_credentials": true,
    "allow_methods": ["*"],
    "allow_headers": ["*"]
}
```

## Error Handling

The API includes comprehensive error handling for various scenarios:

- Invalid audio file format errors
- Audio processing failures
- Model inference errors
- Unexpected exceptions

Each error is returned with an appropriate HTTP status code and a descriptive message.

## Installation and Setup

### Prerequisites

- Python 3.10
- pip (Python package installer)
- Docker (optional, for containerized deployment)

### Setting Up the Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/srihari-swain/ASR-Service.git
   cd ASR-Service
   ```

2. Create a Python virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

#### Method 1: Direct Execution

To run the application directly:

```bash
python src/main.py
```

#### Method 2: Using Docker

1. Build the Docker image:
   ```bash
   docker build -t asr-service .
   ```

2. Run the container:
   ```bash
   docker run -d -p 8000:8000 --name asr-service asr-service
   ```

## Testing the Endpoint

### Using curl:
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/audio.wav"
```

### Using Postman:
1. Create a new POST request to `http://localhost:8000/transcribe`
2. Set the body to "form-data"
3. Add a key "file" with type "file" and select your audio file
4. Send the request

## Response Format

The endpoint returns a JSON response with:

```json
{
    "text": "transcribed text",
    "confidence": 0.99,
    "processing_time": 2.5,
    "duration": 5.0
}
```

## API Documentation

The API documentation is available at:
- Swagger UI: `/docs` 
- ReDoc: `/redoc`

## Design Considerations

1. **Performance**:
   - Async-compatible API design
   - Optimized audio processing pipeline
   - Efficient memory usage with proper cleanup

2. **Error Handling**:
   - Comprehensive validation for audio files
   - Detailed error messages
   - Graceful degradation for model failures

3. **Scalability**:
   - Thread pool for parallel processing
   - Configurable maximum file size and duration
   - Lightweight container image using Python slim

4. **Maintainability**:
   - Clean separation of concerns
   - Proper logging and error tracking
   - Well-documented API endpoints
