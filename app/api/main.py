import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predictor import load_model, predict_rul

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rul-api")

app = FastAPI(title="API RUL Motor Induksi", version="1.0.0")
model = load_model()  # dimuat sekali saat startup


class SensorMotor(BaseModel):
    arus_ampere: float = Field(ge=0, le=100, description="Arus (A)")
    getaran_mm_s: float = Field(ge=0, le=50, description="Getaran (mm/s)")
    suhu_body_c: float = Field(ge=0, le=200, description="Suhu body (°C)")
    tegangan_volt: float = Field(ge=0, le=500, description="Tegangan (V)")


class BatchSensor(BaseModel):
    items: list[SensorMotor] = Field(min_length=1, max_length=1000)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "fitur_input": list(SensorMotor.model_fields.keys()),
    }


@app.post("/predict")
def predict(data: SensorMotor):
    try:
        hasil = predict_rul(model, data.model_dump())
    except Exception:
        log.exception("prediksi gagal")
        raise HTTPException(status_code=500, detail="Prediksi gagal")
    log.info("input=%s output=%s", data.model_dump(), hasil)
    return hasil


@app.post("/predict/batch")
def predict_batch(data: BatchSensor):
    try:
        hasil = [predict_rul(model, item.model_dump()) for item in data.items]
    except Exception:
        log.exception("prediksi batch gagal")
        raise HTTPException(status_code=500, detail="Prediksi batch gagal")
    log.info("batch n=%d", len(hasil))
    return {"jumlah": len(hasil), "hasil": hasil}
