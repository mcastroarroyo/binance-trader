# main.py
import os
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app

# Cloud Run will set PORT; default to 8080 locally
PORT = int(os.environ.get("PORT", 8080))

# Serve the Dev UI too (toggle to False if you want API-only)
SERVE_WEB_INTERFACE = True

# Agents live in the current directory (which contains trade_agent/)
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))

app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    session_service_uri="sqlite:///./sessions.db",  # simple built-in session store
    allow_origins=["*"],
    web=SERVE_WEB_INTERFACE,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
