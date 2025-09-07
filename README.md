# Soundboard

A simple, lightweight soundboard web server built with FastAPI and HTMX that can be deployed on a Linux machine like a Raspberry Pi.

Sounds can be uploaded through the web UI, they are processed to remove any leading/trailing silence and converted to uncompressed WAV. When the server starts, all of the sounds are loaded into RAM to enable low latency playback. The server also supports a high number of concurrent streams, which can be increased by increasing the audio buffer size.

## Installation

Make sure you have UV installed. See https://docs.astral.sh/uv/ for installation instructions for UV.

To start the server, run `./start.sh`. This will install all dependencies, create a `.env` file if it doesn't exist, and start the server. You should update the `.env` file with your desired values, which will be used the next time you start the server. The server is by default password protected by the key in `.env`.

For the best audio performance on Linux, you should install the PulseAudio driver using your favorite package manager.

## Deployment

> Note, you'll have to set up a reverse proxy (_e.g. Nginx, Apache_) to forward requests to your production app.

To start the production server, run `./prod.sh`, which assumes you are using HTTPS for the server, and will only send an authentication cookie if your reverse proxy is set up to use HTTPS. The development script, `start.sh`, allows for cookies to be sent over HTTP, but should only be used for development.


