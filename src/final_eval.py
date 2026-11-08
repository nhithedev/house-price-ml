"""Đánh giá cuối cùng trên tập TEST. Chỉ chạy một lần, sau khi đã chốt mô hình và tham số bằng val.

    python -m src.final_eval

Ghi "test" vào server/models/metrics.json và results/test_results.json.
Không chọn lại mô hình hay tham số dựa trên kết quả ở đây.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluate import by_group, metrics
from src.features import RAW_COLS, TARGET
from src.models import LABELS
from src.split import load_splits

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "server" / "models"
RESULTS = ROOT / "results"
N_BOOT = 1000


def bootstrap_ci(y, pred, seed=0):
    """Khoảng tin cậy 95% bằng bootstrap lại các dòng test."""
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    idx = rng.integers(0, len(y), (N_BOOT, len(y)))
    yb, pb = y[idx], pred[idx]
    mae = np.abs(yb - pb).mean(1)
    r2 = 1 - ((yb - pb) ** 2).sum(1) / ((yb - yb.mean(1, keepdims=True)) ** 2).sum(1)
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"mae": q(mae), "r2": q(r2)}


def breakdown(df, pred, col, bins, labels):
    g = pd.cut(df[col], bins, labels=labels, right=False).reset_index(drop=True)
    y, pred = df[TARGET].reset_index(drop=True), np.asarray(pred)
    rows = {str(k): metrics(y[g == k], pred[(g == k).values]) for k in labels if (g == k).sum() >= 20}
    return pd.DataFrame(rows).T[["n", "mae", "mape"]].round(2).to_dict("index")


def main():
    _, _, test = load_splits()
    path = MODELS / "metrics.json"
    m = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for name in list(m):
        if not (MODELS / f"{name}.joblib").exists():
            print(f"Bỏ qua {name}: chưa có file mô hình (hãy chạy python -m src.train)")
            continue
        model = joblib.load(MODELS / f"{name}.joblib")
        pred = model.predict(test[RAW_COLS])
        r = metrics(test[TARGET], pred)
        r["ci95"] = bootstrap_ci(test[TARGET], pred)
        m[name]["test"] = r
        m[name]["test_by_group"] = by_group(test, pred).to_dict("index")
        out[name] = {"name": LABELS[name], "test": r, "by_group": m[name]["test_by_group"]}
        print(f"{name:<7} test: MAE={r['mae']:.2f} [{r['ci95']['mae'][0]:.2f}; {r['ci95']['mae'][1]:.2f}] "
              f"RMSE={r['rmse']:.2f} R2={r['r2']:.3f} [{r['ci95']['r2'][0]:.3f}; {r['ci95']['r2'][1]:.3f}] "
              f"MAPE={r['mape']:.1f}%")

    have = [k for k in m if k != "median" and (MODELS / f"{k}.joblib").exists()]
    best = min(have, key=lambda k: m[k]["val"]["mae"])   # chọn theo val, không theo test
    pred = joblib.load(MODELS / f"{best}.joblib").predict(test[RAW_COLS])
    out["_best_by_val"] = best
    out["_breakdown"] = {
        "gia_m2": breakdown(test, pred, TARGET, [0, 60, 100, 150, 200, 1000],
                            ["<60", "60-100", "100-150", "150-200", ">=200"]),
        "dien_tich": breakdown(test, pred, "dien_tich", [0, 30, 50, 80, 1000],
                               ["<30 m²", "30-50 m²", "50-80 m²", ">=80 m²"]),
    }
    bias = float(np.mean(pred - test[TARGET]))
    out["_bias_mean_error"] = bias
    worst = test.assign(du_doan=pred, sai_so=pred - test[TARGET]).reindex(
        columns=["tinh", "quan_huyen", "dien_tich", TARGET, "du_doan", "sai_so"])
    out["_worst"] = worst.loc[worst["sai_so"].abs().nlargest(8).index].round(1).to_dict("records")
    print("Mô hình tốt nhất theo val:", best, "| sai số trung bình có dấu (thiên lệch):", round(bias, 2))
    print(json.dumps(out["_breakdown"], ensure_ascii=False, indent=1))

    path.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "test_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
