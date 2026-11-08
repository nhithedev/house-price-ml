"""Phân tích mô hình tốt nhất trên tập val: độ quan trọng đặc trưng và loại bỏ từng nhóm đặc trưng.

    python -m src.analyze

Ghi results/analysis.json. Chỉ dùng train và val (không dùng test).
"""
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from src.evaluate import metrics
from src.features import RAW_COLS, TARGET, make_model
from src.models import LABELS, get_models
from src.split import load_splits

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "server" / "models"
RESULTS = ROOT / "results"

DETAILS = ["mat_tien", "duong_vao", "so_tang", "so_phong_ngu", "so_phong_tam", "phap_ly", "noi_that",
           "huong_nha", "la_du_an"]
ABLATIONS = {
    "Đầy đủ đặc trưng": [],
    "Bỏ phường/xã": ["phuong_xa"],
    "Bỏ quận/huyện và phường/xã (chỉ tỉnh)": ["quan_huyen", "phuong_xa"],
    "Bỏ thông tin chi tiết căn nhà": DETAILS,
    "Chỉ tỉnh + quận/huyện + diện tích": ["phuong_xa"] + DETAILS,
    "Chỉ tỉnh + diện tích": ["quan_huyen", "phuong_xa"] + DETAILS,
}
NAMES = {"tinh": "Tỉnh/thành", "quan_huyen": "Quận/huyện", "phuong_xa": "Phường/xã", "la_du_an": "Thuộc dự án",
         "dien_tich": "Diện tích", "mat_tien": "Mặt tiền", "duong_vao": "Đường vào", "huong_nha": "Hướng nhà",
         "so_tang": "Số tầng", "so_phong_ngu": "Số phòng ngủ", "so_phong_tam": "Số phòng tắm",
         "phap_ly": "Pháp lý", "noi_that": "Nội thất"}


def blank(df, cols):
    df = df[RAW_COLS].copy()
    for c in cols:
        df[c] = np.nan
    return df


def main():
    train, val, _ = load_splits()
    m = json.loads((MODELS / "metrics.json").read_text(encoding="utf-8"))
    have = [k for k in m if k != "median" and (MODELS / f"{k}.joblib").exists()]
    best = min(have, key=lambda k: m[k]["val"]["mae"])
    spec = get_models()[best]
    print("Mô hình tốt nhất trên val:", LABELS[best], m[best]["params"])

    # 1) độ quan trọng hoán vị: hoán các giá trị của một cột thô rồi đo MAE tăng bao nhiêu
    import joblib
    model = joblib.load(MODELS / f"{best}.joblib")
    pi = permutation_importance(model, val[RAW_COLS], val[TARGET], scoring="neg_mean_absolute_error",
                                n_repeats=5, random_state=0, n_jobs=1)
    imp = sorted(({"feature": NAMES[c], "mae_increase": float(mu), "std": float(sd)}
                  for c, mu, sd in zip(RAW_COLS, pi.importances_mean, pi.importances_std)),
                 key=lambda r: -r["mae_increase"])
    for r in imp:
        print(f"  {r['feature']:<14} +{r['mae_increase']:.2f} ± {r['std']:.2f}")

    # 2) loại bỏ nhóm đặc trưng: huấn luyện lại với cùng tham số
    abl = []
    for name, cols in ABLATIONS.items():
        mdl = make_model(clone(spec["est"]).set_params(**m[best]["params"]), spec["kind"])
        mdl.fit(blank(train, cols), train[TARGET])
        r = metrics(val[TARGET], mdl.predict(blank(val, cols)))
        abl.append({"setting": name, **{k: r[k] for k in ("mae", "rmse", "r2", "mape")}})
        print(f"  {name:<42} MAE={r['mae']:.2f} R2={r['r2']:.3f}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "analysis.json").write_text(json.dumps(
        {"model": LABELS[best], "params": m[best]["params"], "permutation_importance": imp, "ablation": abl},
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
