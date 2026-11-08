"""Chia train / val / test từ data/interim/05_features.csv (bước 6 của tiền xử lý).

    python -m src.split

Chia phân tầng theo tỉnh/thành (tỉnh < 100 dòng gộp thành "Khác" chỉ để phân tầng) VÀ theo nhóm:
các dòng cùng (địa chỉ, diện tích, giá) luôn nằm chung một tập, để căn giống nhau (thường cùng dự án)
không bị tách sang train và test (tránh rò rỉ).
Đầu ra: data/splits/{train,val,test}.csv và cập nhật data/interim/preprocessing_log.json.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data" / "interim"
SPLIT_DIR = ROOT / "data" / "splits"
SEED = 42
VAL_SIZE, TEST_SIZE = 0.15, 0.15
RARE_PROVINCE_MIN = 100
TARGET = "gia_m2"


N_FOLDS = 20     # 20 phần: 3 phần test (15%), 3 phần val (15%), 14 phần train (70%)


def split(df):
    cnt = df["tinh"].value_counts()
    strat = df["tinh"].where(df["tinh"].map(cnt) >= RARE_PROVINCE_MIN, "Khác")
    groups = df.groupby(["dia_chi", "dien_tich", "gia_ty"], dropna=False).ngroup()
    fold = pd.Series(-1, index=df.index)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for i, (_, idx) in enumerate(sgkf.split(df, strat, groups)):
        fold.iloc[idx] = i
    n_test, n_val = round(N_FOLDS * TEST_SIZE), round(N_FOLDS * VAL_SIZE)
    return (df[fold >= n_test + n_val], df[(fold >= n_test) & (fold < n_test + n_val)], df[fold < n_test])


def load_splits():
    """Đọc 3 tập đã lưu: trả về (train, val, test)."""
    return tuple(pd.read_csv(SPLIT_DIR / f"{n}.csv", encoding="utf-8-sig")
                 for n in ("train", "val", "test"))


def main():
    df = pd.read_csv(INTERIM / "05_features.csv", encoding="utf-8-sig")
    train, val, test = split(df)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(SPLIT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    top = df["tinh"].value_counts().head(5).index
    share = {n: (p["tinh"].value_counts(normalize=True).reindex(top).round(3).to_dict())
             for n, p in (("train", train), ("val", val), ("test", test))}
    notes = {"train/val/test": [len(train), len(val), len(test)], "seed": SEED,
             "giá/m² trung vị (train/val/test)": [round(float(p[TARGET].median()), 1) for p in (train, val, test)],
             "tỷ lệ 5 tỉnh lớn nhất": share}
    log_path = INTERIM / "preprocessing_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8"))
    log["6"] = {"name": "Chia train/val/test", "rows_in": len(df), "rows_out": len(df), "removed": 0, "notes": notes}
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(len(df), "->", len(train), len(val), len(test))
    for k, v in notes.items():
        print(k, v)


if __name__ == "__main__":
    main()
