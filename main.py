import os
from fastapi import FastAPI

app = FastAPI()

@app.get("/healthz")
def health():
    return {"ok": True}

@app.get("/")
def home():
    return {"status": "up", "message": "hello from Cloud Run"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
PY

