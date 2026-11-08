"""Sinh dữ liệu tĩnh cho giao diện web.

    python -m src.make_options

- web/options.json : danh sách tỉnh -> quận/huyện -> phường/xã (kèm số tin và giá/m² trung vị),
                     các lựa chọn pháp lý / nội thất / hướng nhà, và khoảng giá trị số.
- web/metrics.json : bảng so sánh mô hình (từ server/models/metrics.json) để giao diện hiển thị
                     kể cả khi chưa chạy server.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"


def stat(g):
    return {"count": int(len(g)), "median": round(float(g["gia_m2"].median()), 2)}


def main():
    d = pd.read_csv(ROOT / "data" / "interim" / "05_features.csv", encoding="utf-8-sig")
    provinces = {}
    for t, gt in d.groupby("tinh"):
        districts = {}
        for q, gq in gt.groupby("quan_huyen"):
            wards = {w: stat(gw) for w, gw in gq.groupby("phuong_xa")}
            districts[q] = {**stat(gq), "wards": dict(sorted(wards.items(), key=lambda kv: -kv[1]["count"]))}
        provinces[t] = {**stat(gt), "districts": dict(sorted(districts.items(), key=lambda kv: -kv[1]["count"]))}
    opts = {
        "provinces": dict(sorted(provinces.items(), key=lambda kv: -kv[1]["count"])),
        "legal": sorted(d["phap_ly"].dropna().unique().tolist()),
        "furniture": sorted(d["noi_that"].dropna().unique().tolist()),
        "directions": sorted(d["huong_nha"].dropna().unique().tolist()),
        "area": {"min": 10, "max": 300, "median": float(d["dien_tich"].median())},
        "price_cap_ty": float(d["gia_ty"].max()),     # dữ liệu chỉ có nhà không vượt mức này
        "overall_median": round(float(d["gia_m2"].median()), 2),
    }
    (WEB / "options.json").write_text(json.dumps(opts, ensure_ascii=False), encoding="utf-8")
    print("web/options.json:", len(provinces), "tỉnh,", sum(len(p["districts"]) for p in provinces.values()), "quận/huyện")

    mpath = ROOT / "server" / "models" / "metrics.json"
    if mpath.exists():
        m = json.loads(mpath.read_text(encoding="utf-8"))
        rows = []
        for v in m.values():
            split = "test" if "test" in v else "val"          # ưu tiên điểm test cuối cùng nếu đã có
            r = v[split]
            rows.append({"name": v["name"], "mae": r["mae"], "rmse": r["rmse"], "r2": r["r2"],
                         "mape": r["mape"], "split": split})
        (WEB / "metrics.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        print("web/metrics.json:", len(rows), "mô hình")


if __name__ == "__main__":
    main()
