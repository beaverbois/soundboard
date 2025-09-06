from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import threading
import uvicorn
import json
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from pydub.silence import detect_silence
from pydub.effects import normalize
from typing import Dict, Any
import uuid

app = FastAPI()
ROOT = Path(__file__).parent.resolve()
templates = Jinja2Templates(directory=str(ROOT.parent / "templates"))

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
MAX_CONCURRENT_STREAMS = 16

# Initialize audio device
try:
    device_info = sd.query_devices()
    print(f"Available audio devices: {len(device_info)} found")
except Exception as e:
    print(f"Warning: Could not query audio devices: {e}")

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

def cleanup_finished_streams():
    """Remove finished streams from active_streams list"""
    global active_streams
    active_streams = [stream for stream in active_streams if stream.active]

def play_sound_async(filename: str, volume: float = 1.0) -> None:
    """Play sound from cache using sounddevice with concurrent playback and stream management"""
    global active_streams
    
    if filename not in audio_cache:
        print(f"Sound {filename} not found in cache")
        return
    
    # Clean up finished streams
    cleanup_finished_streams()
    
    # Check if we're at the stream limit
    if len(active_streams) >= MAX_CONCURRENT_STREAMS:
        print(f"Maximum concurrent streams ({MAX_CONCURRENT_STREAMS}) reached, ignoring new sound")
        return
    
    try:
        samples = audio_cache[filename] * volume
        
        # Create a new OutputStream for each sound to enable concurrent playback
        def audio_callback(outdata, frames, time, status):
            if status:
                print(f"Audio callback status: {status}")
            
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
        
        # Create the output stream
        stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=max(channels, 2),  # Ensure at least stereo output
            callback=audio_callback,
            finished_callback=lambda: cleanup_finished_streams()
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
async def index():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/sounds")
async def list_sounds():
    """Get sounds from metadata JSON store"""
    metadata = load_metadata()
    sounds = metadata.get("sounds", [])
    
    html_content = templates.get_template("sound_buttons.html").render(
        sounds=sounds,
        current_volume=100
    )
    return HTMLResponse(html_content)

@app.post("/api/play")
async def play(name: str = Form(), volume: str = Form("100")):
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
async def upload_form():
    return FileResponse(ROOT.parent / "templates" / "upload_form.html")

@app.post("/api/upload")
async def upload_sound(
    sound_name: str = Form(),
    button_color: str = Form("#ff1744"),
    sound_file: UploadFile = File()
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
