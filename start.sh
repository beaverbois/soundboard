#!/bin/bash

set -e

cd "$(dirname "$0")"

cp -n .env.example .env || true

if ! command -v uv >/dev/null 2>&1; then
    echo "no uv found"
    exit 1
else
    uv run fastapi dev --reload &
fi

uv run fastapi start --host
