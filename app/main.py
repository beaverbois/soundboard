from fastapi import FastAPI, Form, UploadFile, File, Request, HTTPException, Depends, Cookie
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer 
from pathlib import Path
import threading
import uvicorn
import json
import numpy as np
import sounddevice as sd
import platform
from pydub import AudioSegment
from pydub.silence import detect_silence
from pydub.effects import normalize
from typing import Dict, Any, Optional
import uuid
import os
from dotenv import load_dotenv
import jwt
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

app = FastAPI()

ROOT = Path(__file__).parent.resolve()
templates = Jinja2Templates(directory=str(ROOT.parent / "templates"))

# Authentication configuration
SECRET_KEY = os.getenv("SECRET_KEY")
SOUNDBOARD_PASSWORD = os.getenv("SOUNDBOARD_PASSWORD")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

# Determine if running in production (secure cookies for HTTPS)
IS_PRODUCTION = os.getenv("ENVIRONMENT") == "production"

security = HTTPBearer(auto_error=False)

if (ROOT / "static").exists():
    STATIC_DIR = ROOT / "static"
elif (ROOT.parent / "static").exists():
    STATIC_DIR = ROOT.parent / "static"
else:
    STATIC_DIR = ROOT / "static"

SOUNDS_DIR = STATIC_DIR / "sounds"
METADATA_FILE = ROOT.parent / "sounds_metadata.json"

# Global variables for low-latency audio
audio_cache: Dict[str, np.ndarray] = {}
sample_rate = 44100
device_info = None
active_streams = []
MAX_CONCURRENT_STREAMS = 32
selected_device_id = None  # User-selected audio output device
master_volume = 1.0  # Master volume multiplier (0.0 to 1.0)

# Initialize audio device
try:
    device_info = sd.query_devices()
    print(f"Available audio devices: {len(device_info)} found")
except Exception as e:
    print(f"Warning: Could not query audio devices: {e}")

def get_audio_devices():
    """Get list of available audio output devices"""
    try:
        devices = sd.query_devices()
        output_devices = []
        
        for i, device in enumerate(devices):
            if device['max_output_channels'] > 0:  # Only output devices
                output_devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_output_channels'],
                    'sample_rate': device['default_samplerate'],
                    'is_default': i == sd.default.device[1]  # Check if it's default output
                })
        
        return output_devices
    except Exception as e:
        print(f"Error getting audio devices: {e}")
        return []

def get_selected_device_id():
    """Get the currently selected device ID, fallback to default"""
    global selected_device_id
    if selected_device_id is not None:
        return selected_device_id
    
    # Try to get default output device
    try:
        return sd.default.device[1]  # Default output device
    except:
        return None

def load_metadata() -> Dict[str, Any]:
    """Load sound metadata from JSON file"""
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
    return {"sounds": []}

def save_metadata(metadata: Dict[str, Any]) -> None:
    """Save sound metadata to JSON file"""
    try:
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Error saving metadata: {e}")

def load_audio_to_cache() -> None:
    """Load all WAV files into RAM for low-latency playback"""
    global audio_cache
    metadata = load_metadata()
    
    for sound in metadata["sounds"]:
        wav_path = SOUNDS_DIR / sound["filename"]
        if wav_path.exists() and wav_path.suffix.lower() == '.wav':
            try:
                audio = AudioSegment.from_wav(str(wav_path))
                # Convert to numpy array for sounddevice
                samples = np.array(audio.get_array_of_samples())
                if audio.channels == 2:
                    samples = samples.reshape((-1, 2))
                # Normalize to float32 range [-1, 1]
                samples = samples.astype(np.float32) / 32768.0
                audio_cache[sound["filename"]] = samples
                print(f"Loaded {sound['filename']} into cache")
            except Exception as e:
                print(f"Error loading {sound['filename']}: {e}")
    
    print(f"Audio cache loaded with {len(audio_cache)} sounds")

def process_audio_file(input_path: Path, output_path: Path) -> None:
    """Convert audio file to WAV, strip silence, and normalize"""
    try:
        # Load audio file (supports many formats)
        audio = AudioSegment.from_file(str(input_path))
        
        # Convert to mono if stereo (optional, comment out to keep stereo)
        # audio = audio.set_channels(1)
        
        # Set sample rate to 44.1kHz
        audio = audio.set_frame_rate(sample_rate)
        
        # Strip silence from beginning and end more aggressively
        silence_ranges = detect_silence(audio, min_silence_len=50, silence_thresh=-45)
        
        if silence_ranges:
            # Find first non-silent segment
            start_trim = 0
            for start, end in silence_ranges:
                if start == 0:
                    start_trim = end
                else:
                    break
            
            # Find last non-silent segment
            end_trim = len(audio)
            for start, end in reversed(silence_ranges):
                if end == len(audio):
                    end_trim = start
                else:
                    break
            
            # Apply trimming
            if start_trim > 0 or end_trim < len(audio):
                audio = audio[start_trim:end_trim]
        
        # Normalize audio
        audio = normalize(audio)
        
        # Export as WAV
        audio.export(str(output_path), format="wav")
        
    except Exception as e:
        raise Exception(f"Error processing audio: {e}")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> bool:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("authenticated") is True
    except jwt.PyJWTError:
        return False

def get_current_user(auth_token: Optional[str] = Cookie(None)):
    """Dependency to check authentication"""
    if not auth_token or not verify_token(auth_token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return True

def cleanup_finished_streams():
    """Remove finished streams from active_streams list and properly close them"""
    global active_streams
    finished_streams = [stream for stream in active_streams if not stream.active]
    
    # Explicitly close finished streams to free PulseAudio connections
    for stream in finished_streams:
        try:
            if not stream.closed:
                stream.close()
        except Exception as e:
            print(f"Error closing stream: {e}")
    
    # Keep only active streams
    active_streams = [stream for stream in active_streams if stream.active]

def play_sound_async(filename: str, volume: float = 1.0) -> None:
    """Play sound from cache using sounddevice with concurrent playback and stream management"""
    global active_streams, master_volume
    
    if filename not in audio_cache:
        print(f"Sound {filename} not found in cache")
        return
    
    # Clean up finished streams
    cleanup_finished_streams()
    
    # More aggressive cleanup if approaching limits
    if len(active_streams) >= MAX_CONCURRENT_STREAMS * 0.8:  # At 80% capacity
        cleanup_finished_streams()
    
    # Check if we're at the stream limit
    if len(active_streams) >= MAX_CONCURRENT_STREAMS:
        print(f"Maximum concurrent streams ({MAX_CONCURRENT_STREAMS}) reached, ignoring new sound")
        return
    
    try:
        # Apply both individual volume and master volume
        final_volume = volume * master_volume
        samples = audio_cache[filename] * final_volume
        
        # Create a new OutputStream for each sound to enable concurrent playback
        def audio_callback(outdata, frames, time, status):
            # Get the current position in the audio
            current_frame = getattr(audio_callback, 'frame_index', 0)
            
            # Calculate how many frames we can output
            frames_to_output = min(frames, len(samples) - current_frame)
            
            if frames_to_output > 0:
                # Handle mono vs stereo
                if len(samples.shape) == 1:  # Mono
                    outdata[:frames_to_output, 0] = samples[current_frame:current_frame + frames_to_output]
                    if outdata.shape[1] > 1:  # If output is stereo, duplicate to both channels
                        outdata[:frames_to_output, 1] = samples[current_frame:current_frame + frames_to_output]
                else:  # Stereo
                    outdata[:frames_to_output] = samples[current_frame:current_frame + frames_to_output]
                
                # Clear remaining frames if any
                if frames_to_output < frames:
                    outdata[frames_to_output:] = 0
                
                # Update frame index
                audio_callback.frame_index = current_frame + frames_to_output
            else:
                # No more audio data, fill with silence
                outdata.fill(0)
                raise sd.CallbackStop
        
        # Initialize frame index
        audio_callback.frame_index = 0
        
        # Determine number of channels
        channels = 1 if len(samples.shape) == 1 else samples.shape[1]
        
        # Get selected device
        device_id = get_selected_device_id()
        
        # Create the output stream with smaller buffer for minimal latency
        # Always use default device on Linux to avoid device conflicts
        actual_device = None if platform.system() == 'Linux' else device_id
        
        stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=2,  # Force stereo to avoid channel conflicts
            callback=audio_callback,
            finished_callback=lambda: cleanup_finished_streams(),
            device=actual_device,
            blocksize=2048,
            latency='low'
        )
        
        # Add to active streams list before starting
        active_streams.append(stream)
        stream.start()
        
        print(f"Playing {filename}, active streams: {len(active_streams)}")
        
    except Exception as e:
        print(f"Error playing sound {filename}: {e}")
        # Clean up on error
        cleanup_finished_streams()

ALLOW_OVERLAP = True


# Remove old audio player detection functions as we now use sounddevice


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Load audio cache on startup
@app.on_event("startup")
async def startup_event():
    """Load all audio files into cache on server startup"""
    SOUNDS_DIR.mkdir(exist_ok=True)
    load_audio_to_cache()

@app.get("/")
async def index(auth_token: Optional[str] = Cookie(None)):
    if not auth_token or not verify_token(auth_token):
        return RedirectResponse(url="/login", status_code=302)
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/login")
async def login_page():
    return FileResponse(ROOT.parent / "templates" / "login.html")

@app.post("/api/login")
async def login(password: str = Form()):
    if password != SOUNDBOARD_PASSWORD:
        html_content = templates.get_template("status_message.html").render(
            message="Invalid password",
            color="#f66"
        )
        return HTMLResponse(html_content)
    
    # Create access token
    access_token_expires = timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    access_token = create_access_token(
        data={"authenticated": True}, expires_delta=access_token_expires
    )
    
    # Create response with redirect
    response = HTMLResponse(
        '<script>window.location.href = "/";</script>'
    )
    response.set_cookie(
        key="auth_token",
        value=access_token,
        max_age=ACCESS_TOKEN_EXPIRE_HOURS * 3600,
        httponly=True,
        secure=IS_PRODUCTION,  # True for production (HTTPS), False for development
        samesite="lax"
    )
    return response

@app.post("/api/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie(key="auth_token", secure=IS_PRODUCTION, samesite="lax")
    return response

@app.get("/api/sounds")
async def list_sounds(current_user: bool = Depends(get_current_user)):
    """Get sounds from metadata JSON store"""
    metadata = load_metadata()
    sounds = metadata.get("sounds", [])
    
    html_content = templates.get_template("sound_buttons.html").render(
        sounds=sounds,
        current_volume=100
    )
    return HTMLResponse(html_content)

@app.post("/api/play")
async def play(name: str = Form(), volume: str = Form("100"), current_user: bool = Depends(get_current_user)):
    """Play sound using low-latency sounddevice from RAM cache"""
    name = (name or "").strip()
    from pathlib import Path as _P
    name = _P(name).name
    
    try:
        volume_pct = float(volume)
    except (ValueError, TypeError):
        volume_pct = 100.0

    if not name:
        html_content = templates.get_template("status_message.html").render(
            message="Error: Missing filename",
            color="#f66"
        )
        return HTMLResponse(html_content)

    # Convert volume percentage to float (0.0 to 1.0)
    volume_float = max(0.0, min(100.0, volume_pct)) / 100.0
    
    # Play sound from cache
    def _play_sound():
        play_sound_async(name, volume_float)
    
    # Run in background thread for non-blocking playback
    threading.Thread(target=_play_sound, daemon=True).start()
    
    # Return empty response
    return HTMLResponse("")

@app.get("/upload")
async def upload_form(current_user: bool = Depends(get_current_user)):
    return FileResponse(ROOT.parent / "templates" / "upload_form.html")

@app.get("/templates/audio_settings_modal")
async def audio_settings_modal(current_user: bool = Depends(get_current_user)):
    return FileResponse(ROOT.parent / "templates" / "audio_settings_modal.html")

@app.get("/templates/sound_edit_modal")
async def sound_edit_modal(current_user: bool = Depends(get_current_user)):
    return FileResponse(ROOT.parent / "templates" / "sound_edit_modal.html")

@app.post("/api/upload")
async def upload_sound(
    sound_name: str = Form(),
    button_color: str = Form("#ff1744"),
    sound_file: UploadFile = File(),
    current_user: bool = Depends(get_current_user)
):
    """Upload and process audio file, convert to WAV, and add to metadata"""
    # Validate inputs
    if not sound_name.strip():
        html_content = templates.get_template("status_message.html").render(
            message="Error: Sound name is required",
            color="#f66"
        )
        return HTMLResponse(html_content)
    
    if not sound_file.filename:
        html_content = templates.get_template("status_message.html").render(
            message="Error: Sound file is required",
            color="#f66"
        )
        return HTMLResponse(html_content)
    
    # Create sounds directory if it doesn't exist
    SOUNDS_DIR.mkdir(exist_ok=True)
    
    # Clean the sound name for filename
    clean_name = sound_name.strip().replace(' ', '_').replace('-', '_')
    # Remove any characters that aren't alphanumeric or underscore
    clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '_')
    
    # Create temporary filename for uploaded file
    temp_filename = f"temp_{uuid.uuid4().hex}{Path(sound_file.filename).suffix.lower()}"
    temp_path = SOUNDS_DIR / temp_filename
    
    # Final WAV filename
    wav_filename = f"{clean_name}.wav"
    wav_path = SOUNDS_DIR / wav_filename
    
    # Handle duplicate filenames
    counter = 1
    while wav_path.exists():
        wav_filename = f"{clean_name}_{counter}.wav"
        wav_path = SOUNDS_DIR / wav_filename
        counter += 1
    
    try:
        # Save the uploaded file temporarily
        with open(temp_path, "wb") as buffer:
            content = await sound_file.read()
            buffer.write(content)
        
        # Process audio file (convert to WAV, strip silence, normalize)
        process_audio_file(temp_path, wav_path)
        
        # Remove temporary file
        temp_path.unlink()
        
        # Update metadata
        metadata = load_metadata()
        new_sound = {
            "filename": wav_filename,
            "nice_label": sound_name.strip(),
            "color": button_color
        }
        metadata["sounds"].append(new_sound)
        save_metadata(metadata)
        
        # Load the new sound into cache
        try:
            audio = AudioSegment.from_wav(str(wav_path))
            samples = np.array(audio.get_array_of_samples())
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
            samples = samples.astype(np.float32) / 32768.0
            audio_cache[wav_filename] = samples
            print(f"Added {wav_filename} to audio cache")
        except Exception as e:
            print(f"Error adding {wav_filename} to cache: {e}")
        
        # Return success message with redirect
        html_content = templates.get_template("status_message.html").render(
            message=f"Successfully uploaded and processed: {sound_name}",
            color="#0a0",
            redirect_delay=2
        )
        return HTMLResponse(html_content)
        
    except Exception as e:
        # Clean up temporary file if it exists
        if temp_path.exists():
            temp_path.unlink()
        
        html_content = templates.get_template("status_message.html").render(
            message=f"Error processing file: {str(e)}",
            color="#f66"
        )
        return HTMLResponse(html_content)

@app.post("/api/delete")
async def delete_sound(filename: str = Form(), current_user: bool = Depends(get_current_user)):
    """Delete a sound file and remove from metadata"""
    try:
        # Remove from metadata
        metadata = load_metadata()
        metadata["sounds"] = [s for s in metadata["sounds"] if s["filename"] != filename]
        save_metadata(metadata)
        
        # Remove from audio cache
        if filename in audio_cache:
            del audio_cache[filename]
        
        # Remove file from disk
        file_path = SOUNDS_DIR / filename
        if file_path.exists():
            file_path.unlink()
        
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/api/update")
async def update_sound(request: Request, current_user: bool = Depends(get_current_user)):
    """Update sound name and color"""
    try:
        data = await request.json()
        filename = data.get("filename")
        name = data.get("name")
        color = data.get("color")
        
        if not filename or not name or not color:
            return JSONResponse({"success": False, "error": "Missing required fields"}, status_code=400)
            
        metadata = load_metadata()
        for sound in metadata["sounds"]:
            if sound["filename"] == filename:
                sound["nice_label"] = name
                sound["color"] = color
                break
        else:
            return JSONResponse({"success": False, "error": "Sound not found"}, status_code=404)
            
        save_metadata(metadata)
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/api/edit")
async def edit_sound(filename: str = Form(), name: str = Form(), color: str = Form(), current_user: bool = Depends(get_current_user)):
    """Edit sound name and color"""
    try:
        metadata = load_metadata()
        for sound in metadata["sounds"]:
            if sound["filename"] == filename:
                sound["nice_label"] = name
                sound["color"] = color
                break
        save_metadata(metadata)
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/api/reorder")
async def reorder_sounds(request: Request, current_user: bool = Depends(get_current_user)):
    """Reorder sounds based on provided order"""
    try:
        data = await request.json()
        order = data.get("order", [])
        
        metadata = load_metadata()
        sounds_dict = {s["filename"]: s for s in metadata["sounds"]}
        
        # Reorder sounds based on the provided order
        reordered_sounds = []
        for filename in order:
            if filename in sounds_dict:
                reordered_sounds.append(sounds_dict[filename])
        
        metadata["sounds"] = reordered_sounds
        save_metadata(metadata)
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/audio-devices")
async def get_devices(current_user: bool = Depends(get_current_user)):
    """Get list of available audio output devices"""
    devices = get_audio_devices()
    current_device = get_selected_device_id()
    volume_percent = int(master_volume * 100)
    
    return JSONResponse({
        "devices": devices,
        "current_device": current_device,
        "volume": volume_percent
    })

@app.post("/api/select-device")
async def select_device(request: Request, current_user: bool = Depends(get_current_user)):
    """Select audio output device"""
    global selected_device_id
    
    try:
        data = await request.json()
        device_id = data.get("device_id")
        
        if device_id is not None:
            # Validate device exists
            devices = get_audio_devices()
            valid_device_ids = [d["id"] for d in devices]
            
            if device_id in valid_device_ids:
                selected_device_id = device_id
                return JSONResponse({"success": True, "device_id": device_id})
            else:
                return JSONResponse({"success": False, "error": "Invalid device ID"}, status_code=400)
        else:
            return JSONResponse({"success": False, "error": "Device ID required"}, status_code=400)
            
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/api/set-volume")
async def set_volume(request: Request, current_user: bool = Depends(get_current_user)):
    """Set master volume level"""
    global master_volume
    
    try:
        data = await request.json()
        volume = data.get("volume")
        
        if volume is not None:
            # Convert percentage (0-100) to float (0.0-1.0)
            volume_float = max(0.0, min(100.0, float(volume))) / 100.0
            master_volume = volume_float
            return JSONResponse({"success": True, "volume": volume})
        else:
            return JSONResponse({"success": False, "error": "Volume required"}, status_code=400)
            
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/api/get-volume")
async def get_volume(current_user: bool = Depends(get_current_user)):
    """Get current master volume level"""
    global master_volume
    # Convert float (0.0-1.0) to percentage (0-100)
    volume_percent = int(master_volume * 100)
    return JSONResponse({"volume": volume_percent})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
