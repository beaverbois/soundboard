#!/bin/bash
set -e

cd "$(dirname "$0")"

cp --update=none .env.example .env || true

exec uv run fastapi start --host 127.0.0.1 --port 8000