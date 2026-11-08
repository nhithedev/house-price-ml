"""Danh sách mô hình: tên -> {estimator, kind, grid}. Thêm mô hình mới chỉ cần thêm vào đây.

`grid` là lưới tham số được duyệt toàn bộ trên tập val (chọn theo MAE).
"""
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

SEED = 0

LABELS = {
    "median": "Trung vị (mốc tham chiếu)",
    "ridge": "Ridge Regression",
    "rf": "Random Forest",
    "hgb": "Gradient Boosting",
}


def get_models() -> dict:
    return {
        "median": dict(est=DummyRegressor(strategy="median"), kind="linear", grid={}),
        "ridge": dict(est=Ridge(), kind="linear",
                      grid={"alpha": [0.3, 1, 3, 10, 30, 100]}),
        "rf": dict(est=RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED), kind="tree",
                   grid={"min_samples_leaf": [1, 3, 5], "max_features": [0.3, 0.6, 1.0]}),
        "hgb": dict(est=HistGradientBoostingRegressor(max_iter=500, early_stopping=False, random_state=SEED),
                    kind="tree",
                    grid={"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31, 63, 127, 255],
                          "l2_regularization": [0.0, 1.0]}),
    }
