"""Command-line entry point for the local API server."""

from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("atlas.web.app:create_app", factory=True, host="127.0.0.1", port=8000, reload=False)
