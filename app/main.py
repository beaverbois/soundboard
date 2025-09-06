from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import subprocess, shutil, threading
import uvicorn
import math
from typing import Optional

app = FastAPI()
ROOT = Path(__file__).parent.resolve()

if (ROOT / "static").exists():
    STATIC_DIR = ROOT / "static"
elif (ROOT.parent / "static").exists():
    STATIC_DIR = ROOT.parent / "static"
else:
    STATIC_DIR = ROOT / "static"

SOUNDS_DIR = STATIC_DIR / "sounds"

def find_player() -> Optional[str]:
    for candidate in ("mpg123", "mpg321", "mpv"):
        p = shutil.which(candidate)
        if p:
            return p
    return None

PLAYER = find_player()  # None when not found
KILLALL = shutil.which("killall")  # from psmisc; optional

AUDIO_DEVICE = None
ALLOW_OVERLAP = True


def suggest_install_cmd() -> str:

    if shutil.which("apt-get"):
        return "sudo apt-get install mpg123"
    if shutil.which("dnf"):
        return "sudo dnf install mpg123"
    if shutil.which("pacman"):
        return "sudo pacman -S mpg123"
    if shutil.which("apk"):
        return "sudo apk add mpg123"
    return "Install mpg123 using your distro package manager (e.g. apt, dnf, pacman)"


def suggest_ffmpeg_install_cmd() -> str:
    if shutil.which("apt-get"):
        return "sudo apt-get install ffmpeg"
    if shutil.which("dnf"):
        return "sudo dnf install ffmpeg"
    if shutil.which("pacman"):
        return "sudo pacman -S ffmpeg"
    if shutil.which("apk"):
        return "sudo apk add ffmpeg"
    return "Install ffmpeg (which provides ffprobe) using your distro package manager"


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def log_proc_output(proc):
    try:
        for line in proc.stdout:
            print("[mpg123]", line.rstrip(), flush=True)
    except Exception as e:
        print("[mpg123] logger stopped:", e, flush=True)

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/sounds")
async def list_sounds():
    base = SOUNDS_DIR.resolve()
    files = sorted([p.name for p in base.iterdir() if p.is_file() and p.suffix.lower()==".mp3"]) if base.exists() else []
    return JSONResponse(files)

@app.post("/api/play")
async def play(request: Request):
    data = await request.json()
    name = (data.get("name") or "").strip()
    from pathlib import Path as _P
    name = _P(name).name
    volume_pct = data.get("volume")

    if not name:
        raise HTTPException(status_code=400, detail="Missing filename")

    target = (SOUNDS_DIR / name).resolve()
    base = SOUNDS_DIR.resolve()
    if not str(target).startswith(str(base)):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {name}")

    try:
        v = float(volume_pct) if volume_pct is not None else 100.0
    except Exception:
        v = 100.0
    v = max(0.0, min(100.0, v))
    scale = int(round(v * 327.68)) 

    if not ALLOW_OVERLAP and KILLALL:
        subprocess.run([KILLALL, "-q", "mpg123"], check=False)
    if not PLAYER:
        try:
            from pydub import AudioSegment
            from pydub.playback import play as pydub_play
        except Exception:
            hint = suggest_install_cmd()
            raise HTTPException(status_code=503, detail=(
                "No external audio player found (mpg123). "
                f"Install it: {hint} or install the Python package 'pydub' and ffmpeg to enable a software fallback."
            ))
        if not (shutil.which("ffprobe") or shutil.which("ffmpeg")):
            ff_hint = suggest_ffmpeg_install_cmd()
            raise HTTPException(status_code=503, detail=(
                "pydub requires ffmpeg/ffprobe but none were found on PATH. "
                f"Install ffmpeg: {ff_hint}"
            ))

        def _play_pydub():
            # load and apply volume; run in background so request returns quickly
            audio = AudioSegment.from_file(str(target))
            if v <= 0:
                audio = AudioSegment.silent(duration=len(audio))
            elif v < 100.0:
                # convert linear percent to decibels: gain_db = 20 * log10(fraction)
                try:
                    gain_db = 20 * math.log10(v / 100.0)
                    audio = audio.apply_gain(gain_db)
                except ValueError:
                    # guard against v == 0 etc.
                    audio = AudioSegment.silent(duration=len(audio))
            pydub_play(audio)

        threading.Thread(target=_play_pydub, daemon=True).start()
        return JSONResponse({"ok": True, "playing": name, "volume": v, "player": "pydub"})

    # Build command for known external players
    player_bin = Path(PLAYER).name
    if player_bin in ("mpg123", "mpg321"):
        cmd = [PLAYER]
        if AUDIO_DEVICE:
            cmd += ["-a", AUDIO_DEVICE]
        cmd += ["-f", str(scale), str(target)]
    elif player_bin == "mpv":
        cmd = [PLAYER, "--no-terminal", "--really-quiet", "--volume", str(int(v))]
        if AUDIO_DEVICE:
            cmd += ["--audio-device=alsa/" + AUDIO_DEVICE]
        cmd += [str(target)]
    else:
        cmd = [PLAYER, str(target)]

    print(f"[play] {cmd}", flush=True)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    threading.Thread(target=log_proc_output, args=(proc,), daemon=True).start()

    return JSONResponse({"ok": True, "playing": name, "volume": v, "player": player_bin})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
