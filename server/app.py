"""API dự đoán giá nhà và phục vụ giao diện web.

Chạy từ thư mục gốc của dự án:
    uvicorn server.app:app --reload
rồi mở http://127.0.0.1:8000

Biến môi trường MODEL chọn mô hình (mặc định hgb = Gradient Boosting; có thể là ridge, rf).
"""
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                      # để joblib import được src.features
from src.features import RAW_COLS, TARGET          # noqa: E402
from src.models import LABELS                      # noqa: E402
from src.split import load_splits                  # noqa: E402

MODELS = ROOT / "server" / "models"
MODEL_NAME = os.environ.get("MODEL", "hgb")
INTERVAL = (0.10, 0.90)                            # khoảng dự đoán ~80%
state = {}


class House(BaseModel):
    """Dữ liệu một căn nhà. Chỉ tỉnh, quận/huyện và diện tích là bắt buộc."""
    tinh: str
    quan_huyen: str
    phuong_xa: Optional[str] = None
    dien_tich: float = Field(ge=10, le=300, description="m²")
    so_phong_ngu: Optional[int] = Field(None, ge=1, le=9)
    so_tang: Optional[int] = Field(None, ge=1, le=12)
    so_phong_tam: Optional[int] = Field(None, ge=1, le=9)
    mat_tien: Optional[float] = Field(None, ge=1, le=30, description="m")
    duong_vao: Optional[float] = Field(None, ge=1, le=40, description="m")
    phap_ly: Optional[str] = None
    noi_that: Optional[str] = None
    huong_nha: Optional[str] = None
    la_du_an: int = Field(0, ge=0, le=1)


@asynccontextmanager
async def lifespan(_app):
    path = MODELS / f"{MODEL_NAME}.joblib"
    if not path.exists():
        raise RuntimeError(f"Không thấy {path}. Hãy chạy `python -m src.train` trước.")
    model = joblib.load(path)
    # Khoảng dự đoán: phân vị của tỉ lệ (giá thật / giá dự đoán) trên tập val (không dùng test)
    _, val, _ = load_splits()
    ratio = val[TARGET].to_numpy() / model.predict(val[RAW_COLS])
    state.update(model=model, label=LABELS.get(MODEL_NAME, MODEL_NAME),
                 q_lo=float(np.quantile(ratio, INTERVAL[0])), q_hi=float(np.quantile(ratio, INTERVAL[1])))
    yield


app = FastAPI(title="Dự đoán giá nhà Việt Nam", lifespan=lifespan)


@app.get("/api/health")
def health():
    return {"status": "ok", "model": state["label"]}


@app.post("/api/predict")
def predict(h: House):
    row = pd.DataFrame([h.model_dump()]).reindex(columns=RAW_COLS)
    try:
        price = float(state["model"].predict(row)[0])
    except Exception as e:                          # lỗi bất ngờ của pipeline -> 500 có thông điệp rõ
        raise HTTPException(500, f"Không dự đoán được: {e}")
    return {"gia_m2": round(price, 2), "model": state["label"],
            "low": round(price * state["q_lo"], 2), "high": round(price * state["q_hi"], 2),
            "tong_gia_ty": round(price * h.dien_tich / 1000, 2)}


@app.get("/api/models")
def models():
    """Bảng so sánh mô hình (điểm test nếu đã có, ngược lại điểm val)."""
    m = json.loads((MODELS / "metrics.json").read_text(encoding="utf-8"))
    rows = []
    for k, v in m.items():
        split = "test" if "test" in v else "val"
        rows.append({"id": k, "name": v["name"], "split": split, "in_use": k == MODEL_NAME,
                     **{x: v[split][x] for x in ("mae", "rmse", "r2", "mape")}})
    return rows


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")   # phải đặt sau các route /api
