# ASR Service Implementation Details

## Successfully Implemented Features

1. **Model Conversion and Integration**:
   - Successfully converted .nemo model to ONNX format for efficient inference
   - Extracted vocabulary file from the original model
   - Adapted code to handle model input and output specifications
   - Implemented proper beam search decoding with the vocabulary
   - Modified code to accommodate specific vocabulary size requirements

2. **Audio Processing Pipeline**:
   - Implemented comprehensive audio preprocessing including:
     - Stereo to mono conversion
     - Resampling to 16kHz
     - Log-mel spectrogram computation with appropriate parameters
     - Audio normalization for improved model performance
   - Added validation for various audio file formats and properties
   - Ensured proper handling of BytesIO objects and file pointers

3. **FastAPI Application**:
   - Built a robust REST API with `/transcribe` endpoint
   - Implemented input validation for WAV files
   - Added proper error handling with appropriate HTTP status codes
   - Created async-compatible endpoints while handling sync operations
   - Included model response with confidence scores and timing metrics

4. **Containerization**:
   - Created a lightweight Docker image using Python slim base
   - Configured Docker Compose for easy deployment
   - Set up proper volume mounting for model files
   - Ensured correct Python path configuration in containers
   - Implemented proper restart policies

5. **Configuration Management**:
- Implemented a centralized configuration system using JSON files for all I/O operations
- All model paths, preprocessing parameters, and API settings are externalized
- Easy modification of parameters without code changes, enhancing maintainability
- Supports different environments (development, testing, production) through config switching
- Includes validation of configuration parameters at application startup to catch issues early

## Issues Encountered During Development

1. **ONNX Model Integration Challenges**:
   - Initially encountered difficulty loading the ONNX model correctly
   - Faced issues with input tensor shapes and preprocessing requirements
   - Had to research extensively in NeMo documentation and repositories
   - Required trial and error to determine the correct normalization parameters
   - Needed to properly extract and format the vocabulary file

2. **Audio Processing Complexities**:
   - Encountered issues with BytesIO objects and file pointer positions
   - Had to carefully manage memory when processing large audio files
   - Required special handling for stereo vs. mono audio inputs
   - Needed to implement proper audio duration checking

3. **Async Compatibility Issues**:
   - ONNX Runtime and NumPy libraries are inherently synchronous
   - Implemented workarounds to integrate with FastAPI's async architecture
   - The endpoint is async, but the underlying inference pipeline is synchronous (CPU-bound, ONNX/NumPy/SoundFile)
   - For true async inference, would need to use a model/pipeline that supports asyncio or run the sync code in a threadpool
   - Required careful thread management and resource utilization

4. **Docker Configuration Challenges**:
   - Initially faced module import errors due to incorrect Python path
   - Had to properly configure environment variables and volumes
   - Required precise coordination between Dockerfile and docker-compose.yml

## How I Overcame These Challenges

1. **For Model Integration**:
   - Leveraged multiple AI tools and documentation resources
   - Studied the NeMo repository for correct model export procedures
   - Created a systematic approach to test input and output formats
   - Implemented proper error handling to diagnose issues

2. **For Audio Processing**:
   - Implemented robust file handling with proper position resets
   - Added comprehensive validation steps for audio files
   - Used explicit type checking and conversion for BytesIO objects
   - Removed debugging print statements for cleaner production code

3. **For Docker Configuration**:
   - Used environment variables to manage Python path
   - Implemented proper volume mounting for persistence
   - Added appropriate container restart policies
   - Used docker-compose for streamlined deployment

## Limitations and Assumptions

   - Currently only supports WAV audio format
   - Maximum file size is limited (10MB default)
   - Maximum audio duration is capped (60 seconds)
   - Model is optimized for Hindi language specifically
   - Non-trivial deployment requirements (Docker, sufficient RAM/CPU)



## Future Improvements

   - Add support for more audio formats
   - Implement multi-language capabilities
   - Add punctuation and capitalization
   - Include speaker diarization


