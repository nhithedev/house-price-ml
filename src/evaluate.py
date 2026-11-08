"""Chỉ số đánh giá (đơn vị: triệu/m²) và báo cáo theo nhóm tỉnh."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metrics(y, pred) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return {"mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "r2": float(r2_score(y, pred)),
            "mape": float(np.mean(np.abs(pred - y) / y) * 100),
            "n": int(len(y))}


def province_group(tinh: pd.Series) -> pd.Series:
    return tinh.where(tinh.isin(["Hồ Chí Minh", "Hà Nội"]), "Tỉnh/thành khác")


def by_group(df: pd.DataFrame, pred, target: str = "gia_m2") -> pd.DataFrame:
    """Chỉ số riêng cho Hồ Chí Minh, Hà Nội và các tỉnh còn lại (để hai thành phố lớn không che kết quả)."""
    g = province_group(df["tinh"]).reset_index(drop=True)
    y = df[target].reset_index(drop=True)
    pred = np.asarray(pred)
    rows = {k: metrics(y[g == k], pred[(g == k).values]) for k in g.unique()}
    return pd.DataFrame(rows).T[["n", "mae", "rmse", "r2", "mape"]].round(3)
