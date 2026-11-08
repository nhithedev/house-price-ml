"""Huấn luyện các mô hình, duyệt lưới tham số và chọn theo MAE trên tập val (KHÔNG dùng test).

    python -m src.train                  # tất cả mô hình
    python -m src.train --only hgb ridge

Lưu: server/models/<tên>.joblib và server/models/metrics.json (phần "val").
Test chỉ được dùng ở src/final_eval.py.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

from src.evaluate import by_group, metrics
from src.features import RAW_COLS, TARGET, make_model
from src.models import LABELS, get_models
from src.split import load_splits

OUT = Path(__file__).resolve().parent.parent / "server" / "models"
METRICS = OUT / "metrics.json"


def tune(name, spec, train, val):
    """Duyệt toàn bộ lưới; trả về (model tốt nhất, kết quả val, bảng mọi cấu hình)."""
    runs, best = [], None
    for params in (list(ParameterGrid(spec["grid"])) or [{}]):
        t0 = time.time()
        model = make_model(clone(spec["est"]).set_params(**params), spec["kind"]).fit(train[RAW_COLS], train[TARGET])
        r = metrics(val[TARGET], model.predict(val[RAW_COLS]))
        r.update(params=params, seconds=round(time.time() - t0, 1))
        runs.append(r)
        print(f"  {name} {params} val MAE={r['mae']:.2f} R2={r['r2']:.3f} ({r['seconds']}s)", flush=True)
        if best is None or r["mae"] < best[1]["mae"]:
            best = (model, r)
    return best[0], best[1], runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+")
    a = ap.parse_args()
    train, val, _test = load_splits()          # test chưa dùng ở đây
    OUT.mkdir(parents=True, exist_ok=True)
    results = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() and a.only else {}
    for name, spec in get_models().items():
        if a.only and name not in a.only:
            continue
        model, best, runs = tune(name, spec, train, val)
        pred = model.predict(val[RAW_COLS])
        results[name] = {"name": LABELS[name], "val": {k: best[k] for k in ("mae", "rmse", "r2", "mape", "n")},
                         "params": best["params"], "n_configs": len(runs),
                         "train_seconds": best["seconds"], "grid_runs": runs,
                         "val_by_group": by_group(val, pred).to_dict("index")}
        joblib.dump(model, OUT / f"{name}.joblib")
        print(f"==> {name}: tham số tốt nhất {best['params']} | val MAE={best['mae']:.2f} RMSE={best['rmse']:.2f} "
              f"R2={best['r2']:.3f} MAPE={best['mape']:.1f}%", flush=True)
    METRICS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
