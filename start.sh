#!/bin/bash
set -e

cd "$(dirname "$0")"

cp --update=none .env.example .env || true

if ! command -v uv >/dev/null 2>&1; then
    echo "UV not found. Please install it."
    exit 1
fi

export ENVIRONMENT=development
exec uv run fastapi dev --reload --host 0.0.0.0 --port 8000