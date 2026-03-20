#!/bin/bash
# Start the mabool API server in development mode.
# Run from anywhere — the script resolves its own directory.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV="$REPO_ROOT/.venv/bin/gunicorn"

cd "$SCRIPT_DIR"
APP_CONFIG_ENV=dev "$VENV" \
    -k uvicorn.workers.UvicornWorker \
    --workers 1 \
    --timeout 0 \
    --bind 0.0.0.0:8000 \
    --enable-stdio-inheritance \
    --access-logfile - \
    --reload \
    'mabool.api.app:create_app()'
