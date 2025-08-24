cat > main.py << 'PY'
import os
from fastapi import FastAPI

PORT = int(os.environ.get("PORT", 8080))
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()

# Health endpoint for Cloud Run
@app.get("/healthz")
def health():
    return {"ok": True}

# Try to create the ADK app; fall back to a helpful message if it fails
try:
    from google.adk.cli.fast_api import get_fast_api_app
    adk_app = get_fast_api_app(
        agents_dir=AGENT_DIR,
        session_service_uri="sqlite:///./sessions.db",
        allow_origins=["*"],
        web=True,  # Dev UI
    )
    # mount ADK app at root
    app.mount("/", adk_app)
except Exception as e:
    @app.get("/")
    def adk_boot_error():
        # Show exactly why ADK couldn't boot
        return {
            "status": "ADK failed to initialize",
            "hint": "Check GOOGLE_API_KEY / code imports / trade_agent package",
            "error": str(e),
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
PY
