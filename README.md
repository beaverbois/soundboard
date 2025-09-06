# Soundboard

A simple, lightweight soundboard application built with FastAPI and HTMX that can be deployed on a Linux machine like a Raspberry Pi and triggered from the web.

## Installation

Make sure you have UV installed. See https://docs.astral.sh/uv/ for installation instructions for UV.

To start the server, run `start.sh`. This will create a `.env` file if it doesn't exist and start the server. You should update the `.env` file with your desired values, which will be used the next time you start the server.

> Note, you'll have to set up a reverse proxy (_e.g. Nginx, Apache_) to forward requests to the server.
