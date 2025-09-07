# Soundboard

A simple, lightweight soundboard web server built with FastAPI and HTMX that can be deployed on a Linux machine like a Raspberry Pi.

Sounds can be uploaded through the web UI, they are processed to remove any leading/trailing silence and converted to uncompressed WAV. When the server starts, all of the sounds are loaded into RAM to enable low latency playback. The server also supports a high number of concurrent streams, which can be increased by increasing the audio buffer size.

## Installation

Make sure you have UV installed. See https://docs.astral.sh/uv/ for installation instructions for UV.

To start the server, run `start.sh`. This will create a `.env` file if it doesn't exist and start the server. You should update the `.env` file with your desired values, which will be used the next time you start the server.

> Note, you'll have to set up a reverse proxy (_e.g. Nginx, Apache_) to forward requests to the server.

For the best audio performance, you should install the PulseAudio driver using your favorite package manager.

