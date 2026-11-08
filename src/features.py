"""Đặc trưng và pipeline mô hình.

Mô hình nhận DataFrame thô (các cột như trong data/splits/*.csv, hoặc dữ liệu form của web) và
tự tính khoá địa lý + đặc trưng dẫn xuất bên trong pipeline, nên train và phục vụ dùng chung một đường.
Mọi bước học từ dữ liệu (điền thiếu, mã hoá, chuẩn hoá) chỉ fit trên train.

    model = make_model(Ridge(alpha=10), kind="linear")   # kind="tree" bỏ chuẩn hoá
    model.fit(train[RAW_COLS], train[TARGET]); model.predict(df[RAW_COLS])   # triệu/m²
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "gia_m2"
UNKNOWN = "Không rõ"
RARE_MIN = 20          # giá trị phân loại < 20 lần trong train -> gộp vào nhóm "hiếm"

# Cột thô mà mô hình cần (cột thiếu sẽ được coi là không rõ). Baseline không dùng huong_ban_cong
# (thiếu 83%) và duong (tên đường, quá nhiều giá trị).
RAW_COLS = ["tinh", "quan_huyen", "phuong_xa", "la_du_an", "dien_tich", "mat_tien", "duong_vao",
            "huong_nha", "so_tang", "so_phong_ngu", "so_phong_tam", "phap_ly", "noi_that"]
NUMERIC_RAW = ["la_du_an", "dien_tich", "mat_tien", "duong_vao", "so_tang", "so_phong_ngu", "so_phong_tam"]

CAT_COLS = ["tinh", "quan_key", "phuong_key", "phap_ly", "noi_that", "huong_nha"]
NUM_COLS = ["log_dien_tich", "mat_tien", "duong_vao", "so_tang", "so_phong_ngu", "so_phong_tam",
            "dien_tich_moi_tang", "chieu_sau_uoc_tinh", "so_phong_tam_tren_phong_ngu", "la_du_an"]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Đặc trưng dẫn xuất bằng công thức cố định (cũng được dùng ở bước 5 tiền xử lý)."""
    df = df.copy()
    df["dien_tich_moi_tang"] = (df["dien_tich"] / df["so_tang"]).round(2)
    df["chieu_sau_uoc_tinh"] = (df["dien_tich"] / df["mat_tien"]).round(2)   # ≈ chiều sâu lô đất
    df["so_phong_tam_tren_phong_ngu"] = (df["so_phong_tam"] / df["so_phong_ngu"]).round(2)
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Từ cột thô -> bảng đặc trưng (CAT_COLS + NUM_COLS)."""
    df = df.reindex(columns=RAW_COLS)
    df[NUMERIC_RAW] = df[NUMERIC_RAW].apply(pd.to_numeric, errors="coerce").astype(float)
    df = add_derived(df)
    df["log_dien_tich"] = np.log1p(df["dien_tich"])
    # quận/huyện và phường/xã trùng tên giữa các tỉnh -> ghép khoá theo cấp trên
    df["quan_key"] = df["tinh"].astype(str) + "|" + df["quan_huyen"].astype(str)
    df["phuong_key"] = df["quan_key"] + "|" + df["phuong_xa"].astype(str)
    df.loc[df["quan_huyen"].isna(), ["quan_key", "phuong_key"]] = np.nan
    df.loc[df["phuong_xa"].isna(), "phuong_key"] = np.nan
    out = df[CAT_COLS + NUM_COLS].copy()
    out[CAT_COLS] = out[CAT_COLS].astype(object).where(out[CAT_COLS].notna(), UNKNOWN)
    return out


def build_preprocessor(kind: str = "tree") -> ColumnTransformer:
    cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=RARE_MIN, sparse_output=False)
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]   # + cờ "đang thiếu"
    if kind == "linear":
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([("cat", cat, CAT_COLS), ("num", Pipeline(num_steps), NUM_COLS)])


def make_model(estimator, kind: str = "tree"):
    """Pipeline đầy đủ; học trên log(giá/m²), dự đoán trả về triệu/m²."""
    steps = [("prepare", FunctionTransformer(prepare)),
             ("pre", build_preprocessor(kind)),
             ("est", estimator)]
    return TransformedTargetRegressor(regressor=Pipeline(steps), func=np.log1p, inverse_func=np.expm1)
