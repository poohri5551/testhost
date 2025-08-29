from typing import Annotated, Optional
from pathlib import Path

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .logic import run_pipeline, simulate_event

app = FastAPI(title="Quake PGA Web", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))

# --- แก้ตรงนี้: /api/run รองรับ simulate ด้วย ---
@app.post("/api/run")
def api_run(body: Optional[dict] = Body(default=None)):
    try:
        if body and body.get("mode") == "simulate":
            lat   = float(body["lat"])
            lon   = float(body["lon"])
            depth = float(body["depth"])
            mag   = float(body["mag"])
            data = simulate_event(lat=lat, lon=lon, depth_km=depth, mag=mag)
            return JSONResponse(data)
        # ปกติ: ดึงเหตุการณ์ล่าสุดจาก TMD
        data = run_pipeline()
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# --- มี /api/simulate ไว้ด้วย เผื่อ frontend เรียกเส้นนี้ ---
class SimRequest(BaseModel):
    lat:   Annotated[float, Field(ge=-90,  le=90)]
    lon:   Annotated[float, Field(ge=-180, le=180)]
    depth: Annotated[float, Field(ge=0,    le=700)]
    mag:   Annotated[float, Field(ge=1,    le=10)]

@app.post("/api/simulate")
def api_simulate(req: SimRequest):
    try:
        data = simulate_event(
            lat=float(req.lat),
            lon=float(req.lon),
            depth_km=float(req.depth),
            mag=float(req.mag),
        )
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/health")
def health():
    return {"status": "ok"} 