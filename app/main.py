from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import subprocess, shutil, threading
import uvicorn
import math
from typing import Optional

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
    
    colors = [
        {'bg':'#ff1744', 'shadow':'#c20029'},
        {'bg':'#ff6d00', 'shadow':'#c25500'},
        {'bg':'#ffea00', 'shadow':'#c2b700'},
        {'bg':'#00e676', 'shadow':'#00b85d'},
        {'bg':'#00e5ff', 'shadow':'#00b6cc'},
        {'bg':'#2979ff', 'shadow':'#1f5fcc'},
        {'bg':'#651fff', 'shadow':'#4e18c2'},
        {'bg':'#d500f9', 'shadow':'#a600c6'},
        {'bg':'#ff4081', 'shadow':'#c23365'},
        {'bg':'#7c4dff', 'shadow':'#5e3ac2'},
        {'bg':'#1de9b6', 'shadow':'#17b68f'},
        {'bg':'#ffa000', 'shadow':'#c27c00'}
    ]
    
    sounds = []
    for i, filename in enumerate(files):
        nice_label = filename.replace('.mp3', '').replace('_', ' ').replace('-', ' ').strip()
        sounds.append({
            'filename': filename,
            'nice_label': nice_label,
            'color': colors[i % len(colors)]
        })
    
    html_content = templates.get_template("sound_buttons.html").render(
        sounds=sounds,
        current_volume=100
    )
    return HTMLResponse(html_content)

@app.post("/api/play")
async def play(name: str = Form(), volume: str = Form("100")):
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

    target = (SOUNDS_DIR / name).resolve()
    base = SOUNDS_DIR.resolve()
    if not str(target).startswith(str(base)):
        html_content = templates.get_template("status_message.html").render(
            message="Error: Invalid path",
            color="#f66"
        )
        return HTMLResponse(html_content)
    if not target.exists():
        html_content = templates.get_template("status_message.html").render(
            message=f"Error: File not found: {name}",
            color="#f66"
        )
        return HTMLResponse(html_content)

    v = max(0.0, min(100.0, volume_pct))
    scale = int(round(v * 327.68)) 

    if not ALLOW_OVERLAP and KILLALL:
        subprocess.run([KILLALL, "-q", "mpg123"], check=False)
    if not PLAYER:
        try:
            from pydub import AudioSegment
            from pydub.playback import play as pydub_play
        except Exception:
            hint = suggest_install_cmd()
            html_content = templates.get_template("status_message.html").render(
                message=f"No external audio player found (mpg123). Install it: {hint} or install the Python package 'pydub' and ffmpeg to enable a software fallback.",
                color="#f66"
            )
            return HTMLResponse(html_content)
        if not (shutil.which("ffprobe") or shutil.which("ffmpeg")):
            ff_hint = suggest_ffmpeg_install_cmd()
            html_content = templates.get_template("status_message.html").render(
                message=f"pydub requires ffmpeg/ffprobe but none were found on PATH. Install ffmpeg: {ff_hint}",
                color="#f66"
            )
            return HTMLResponse(html_content)

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
        html_content = templates.get_template("status_message.html").render(
            message=f"Now playing: {name} ({int(v)}%) via pydub",
            color="#aaa"
        )
        return HTMLResponse(html_content)

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

    html_content = templates.get_template("status_message.html").render(
        message=f"Now playing: {name} ({int(v)}%) via {player_bin}",
        color="#aaa"
    )
    return HTMLResponse(html_content)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
