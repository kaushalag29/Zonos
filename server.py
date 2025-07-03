import os
import sys
import platform
import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict

# --- macOS Specific Environment Setup ---
if platform.system().lower() == "darwin":
    logging.info("macOS detected. Setting LLVM environment variables for Zonos.")
    # Add llvm to path
    llvm_path = "/opt/homebrew/opt/llvm/bin"
    os.environ["PATH"] = f"{llvm_path}:{os.environ.get('PATH', '')}"
    # Set linker and compiler flags
    os.environ["LDFLAGS"] = "-L/opt/homebrew/opt/llvm/lib"
    os.environ["CPPFLAGS"] = "-I/opt/homebrew/opt/llvm/include"
    os.environ["CPLUS_INCLUDE_PATH"] = f"/opt/homebrew/opt/llvm/include/c++/v1:{os.environ.get('CPLUS_INCLUDE_PATH', '')}"

app = FastAPI(title="Zonos TTS API Server")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Detect device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
# elif platform.system().lower() == "darwin" and torch.backends.mps.is_available():
#     device = "mps"
logger.info(f"Using device: {device}")

# --- Pydantic Models for Requests ---
class TTSRequest(BaseModel):
    text: str
    reference_speaker_path: str
    output_path: str

# --- Model Loading ---
zonos_model = None
try:
    logger.info("--- Starting Zonos Model Loading ---")
    zonos_model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
    logger.info("--- Zonos Model Loaded Successfully ---")
except Exception as e:
    logger.error(f"--- FATAL: Error loading Zonos model: {e} ---", exc_info=True)
    # The server will still start, but endpoints will fail.

# --- API Endpoints ---
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if zonos_model is not None else "unhealthy",
        "model": "Zonos",
        "device": device,
        "model_loaded": zonos_model is not None
    }

@app.post("/generate-audio")
async def generate_audio(request: TTSRequest):
    """Generate audio using the Zonos TTS model."""
    if zonos_model is None:
        raise HTTPException(status_code=500, detail="Zonos model not loaded.")

    try:
        logger.info(f"Generating Zonos audio for text: '{request.text}'")
        
        if not os.path.exists(request.reference_speaker_path):
            raise HTTPException(status_code=404, detail=f"Reference speaker file not found: {request.reference_speaker_path}")
            
        wav, sampling_rate = torchaudio.load(request.reference_speaker_path)
        speaker = zonos_model.make_speaker_embedding(wav, sampling_rate)

        torch.manual_seed(421)

        cond_dict = make_cond_dict(text=request.text, speaker=speaker, language="en-us")
        conditioning = zonos_model.prepare_conditioning(cond_dict)
        codes = zonos_model.generate(conditioning)
        wavs = zonos_model.autoencoder.decode(codes).cpu()
        
        torchaudio.save(request.output_path, wavs[0], zonos_model.autoencoder.sampling_rate)

        if not os.path.exists(request.output_path) or os.path.getsize(request.output_path) == 0:
            raise HTTPException(status_code=500, detail="Zonos failed to generate or save the audio file.")

        return {"status": "success", "message": f"Zonos audio saved to {request.output_path}"}
    except Exception as e:
        logger.error(f"Error generating Zonos audio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating Zonos audio: {str(e)}")

@app.post("/shutdown")
async def shutdown():
    logger.info("Shutdown request received for Zonos server")
    os._exit(0)
    return {"status": "shutdown", "message": "Server shutting down"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011) 