"""API skeleton. A validated model backend will be added after experiments."""

from fastapi import FastAPI, HTTPException

from app.schemas import PredictRequest, PredictResponse

app = FastAPI(title="DomainTune", version="0.1.0")


@app.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/ready")
def ready() -> None:
    raise HTTPException(status_code=503, detail="No validated model loaded")


@app.get("/model/info")
def model_info() -> None:
    raise HTTPException(status_code=503, detail="No validated model loaded")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    raise HTTPException(status_code=503, detail="No validated model loaded")
