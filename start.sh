#!/bin/bash

set -e

cd "$(dirname "$0")"
pwd

if [ ! command -v uv >/dev/null 2>&1 ]; then
	echo "no uv found"
else
	uv run fastapi dev --reload
fi


uv run fastapi start --host