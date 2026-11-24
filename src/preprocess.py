"""Tiền xử lý dữ liệu theo từng bước; mỗi bước đọc file của bước trước và ghi file riêng.

    original-dataset/vietnam_housing_dataset.csv
      -> 01_standardized.csv        chuẩn hoá tên cột, tách địa chỉ, chuẩn hoá tỉnh/thành, tính giá/m²
      -> 02_deduplicated.csv        loại bản ghi trùng gần (giữ bản đầy đủ nhất)
      -> 03_filtered.csv            loại bản ghi ngoại lai / vô lý
      -> 04_values_fixed.csv        giá trị đo đạc bất thường -> thiếu (NaN)
    (rồi src/split.py chia train/val/test)

Chạy:  python -m src.preprocess            # chạy cả 4 bước
       python -m src.preprocess --only 3   # chỉ chạy bước 3 (cần có file bước 2)
Đầu ra nằm ở data/interim/, kèm preprocessing_log.json và các file *_removed.csv để kiểm tra.
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "original-dataset" / "vietnam_housing_dataset.csv"
INTERIM = ROOT / "data" / "interim"
LOG = INTERIM / "preprocessing_log.json"

# ---- Tham số (lý do chọn: xem docs/preprocessing.md) ----
GIA_M2_RANGE = (10, 400)       # triệu/m²
AREA_RANGE = (10, 300)         # m²
MIN_AREA_PER_BEDROOM = 4       # m² / phòng ngủ
MAX_FRONTAGE = 30              # m
MAX_ROAD = 40                  # m
MAX_BATH_OVER_BED = 2          # phòng tắm nhiều hơn phòng ngủ quá mức này -> vô lý
MAX_BED_PER_FLOOR = 3          # phòng ngủ > 3 × số tầng + 3 -> vô lý
MIN_DEPTH = 3                  # m: chiều sâu ước tính (diện tích / mặt tiền) nhỏ hơn mức này -> vô lý

COLS = {"Address": "dia_chi", "Area": "dien_tich", "Frontage": "mat_tien",
        "Access Road": "duong_vao", "House direction": "huong_nha",
        "Balcony direction": "huong_ban_cong", "Floors": "so_tang",
        "Bedrooms": "so_phong_ngu", "Bathrooms": "so_phong_tam",
        "Legal status": "phap_ly", "Furniture state": "noi_that", "Price": "gia_ty"}
LEGAL = {"Have certificate": "Đã có sổ", "Sale contract": "Hợp đồng mua bán"}
FURNITURE = {"Full": "Đầy đủ", "Basic": "Cơ bản"}

PROVINCES = """An Giang|Bà Rịa Vũng Tàu|Bạc Liêu|Bắc Giang|Bắc Kạn|Bắc Ninh|Bến Tre|Bình Dương|Bình Định|
Bình Phước|Bình Thuận|Cà Mau|Cao Bằng|Cần Thơ|Đà Nẵng|Đắk Lắk|Đắk Nông|Điện Biên|Đồng Nai|Đồng Tháp|
Gia Lai|Hà Giang|Hà Nam|Hà Nội|Hà Tĩnh|Hải Dương|Hải Phòng|Hậu Giang|Hòa Bình|Hồ Chí Minh|Hưng Yên|
Khánh Hòa|Kiên Giang|Kon Tum|Lai Châu|Lạng Sơn|Lào Cai|Lâm Đồng|Long An|Nam Định|Nghệ An|Ninh Bình|
Ninh Thuận|Phú Thọ|Phú Yên|Quảng Bình|Quảng Nam|Quảng Ngãi|Quảng Ninh|Quảng Trị|Sóc Trăng|Sơn La|
Tây Ninh|Thái Bình|Thái Nguyên|Thanh Hóa|Thừa Thiên Huế|Tiền Giang|Trà Vinh|Tuyên Quang|Vĩnh Long|
Vĩnh Phúc|Yên Bái""".replace("\n", "").split("|")
ALIASES = {"tphcm": "Hồ Chí Minh", "hcm": "Hồ Chí Minh", "tp hcm": "Hồ Chí Minh", "hn": "Hà Nội",
           "huế": "Thừa Thiên Huế", "hue": "Thừa Thiên Huế"}


def _key(s: str) -> str:
    """Bỏ dấu, chữ thường, chỉ giữ chữ-số: 'Bà Rịa - Vũng Tàu.' -> 'bariavungtau'."""
    s = unicodedata.normalize("NFD", str(s).lower().replace("đ", "d"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]", "", s)


_PROV = {_key(p): p for p in PROVINCES} | {_key(k): v for k, v in ALIASES.items()}
_PROV_KEYS = sorted(_PROV, key=len, reverse=True)


def canon_province(s):
    """Chuẩn hoá tên tỉnh/thành; khớp tiền tố để cứu chuỗi lẫn tạp ('Bình Dương (gần cafe...)')."""
    k = _key(s)
    for pre in ("thanhpho", "tinh", "tp"):          # 'TP Hồ Chí Minh', 'Thành phố Hà Nội', 'Tỉnh Bình Dương'
        if k.startswith(pre) and k[len(pre):] in _PROV:
            k = k[len(pre):]
    if k in _PROV:
        return _PROV[k]
    for p in _PROV_KEYS:
        if len(p) >= 5 and k.startswith(p):
            return _PROV[p]
    return None


_UNIT_PREFIX = re.compile(r"^(thành phố|thanh pho|tp\.?|quận|quan|q\.?|huyện|huyen|h\.?|thị xã|thi xa|tx\.?|"
                          r"phường|phuong|p\.?|xã|xa|thị trấn|thi tran|tt\.?)\s+")


def _clean_text(s):
    """Chuẩn hoá Unicode (NFC), gộp khoảng trắng thừa."""
    if pd.isna(s):
        return s
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s))).strip()


def _unit_key(s):
    """Khoá so sánh tên quận/phường: chữ thường, bỏ tiền tố hành chính ('Thành phố Dĩ An' -> 'dĩ an')."""
    if pd.isna(s):
        return None
    return _UNIT_PREFIX.sub("", _clean_text(s).lower())


def unify_unit_names(df):
    """Hợp nhất các cách viết khác nhau của cùng một quận/huyện hoặc phường/xã trong cùng tỉnh (cùng quận).

    Chỉ khác tiền tố, hoa/thường, khoảng trắng hoặc mã hoá Unicode thì coi là một nơi; chọn cách viết phổ biến nhất
    (hoà thì theo thứ tự chữ cái) làm tên chuẩn. Không bịa tên mới. Trả về (df, bảng các phép gộp)."""
    df = df.copy()
    merges = []
    for col, scope in (("quan_huyen", ["tinh"]), ("phuong_xa", ["tinh", "quan_huyen"])):
        df[col] = df[col].map(_clean_text)
        key = df[col].map(_unit_key)
        t = df[scope + [col]].assign(_k=key).dropna(subset=[col, "_k"])
        cnt = t.groupby(scope + ["_k", col]).size().reset_index(name="n")
        cnt = cnt.sort_values(["n", col], ascending=[False, True], kind="stable")
        canon = cnt.drop_duplicates(scope + ["_k"]).rename(columns={col: "_canon"}).drop(columns="n")
        m = df[scope + [col]].assign(_k=key).merge(canon, on=scope + ["_k"], how="left")
        m.index = df.index
        new = m["_canon"].where(m["_canon"].notna(), df[col])
        changed = new != df[col]
        if changed.any():
            ch = pd.DataFrame({"cot": col, "tinh": df.loc[changed, "tinh"], "quan_huyen": df.loc[changed, "quan_huyen"],
                               "ten_goc": df.loc[changed, col], "ten_chuan": new[changed]})
            merges.append(ch.groupby(["cot", "tinh", "quan_huyen", "ten_goc", "ten_chuan"], dropna=False)
                          .size().reset_index(name="so_dong"))
        df[col] = new
    out = pd.concat(merges, ignore_index=True) if merges else pd.DataFrame()
    return df, out


# ---------------------------------------------------------------- helpers
def _read(name):
    return pd.read_csv(INTERIM / name, encoding="utf-8-sig")


def _write(df, name):
    INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_csv(INTERIM / name, index=False, encoding="utf-8-sig")


def _log(step, name, rows_in, rows_out, notes):
    log = json.loads(LOG.read_text(encoding="utf-8")) if LOG.exists() else {}
    log[str(step)] = {"name": name, "rows_in": int(rows_in), "rows_out": int(rows_out),
                      "removed": int(rows_in - rows_out), "notes": notes}
    LOG.write_text(json.dumps(dict(sorted(log.items())), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{step}] {name}: {rows_in} -> {rows_out} (bỏ {rows_in - rows_out})")
    for k, v in notes.items():
        print(f"      {k}: {v}")


# ---------------------------------------------------------------- bước 1
def step1_standardize():
    """Đổi tên cột, tách địa chỉ thành tỉnh/quận/phường/đường, chuẩn hoá tỉnh, tính giá/m²."""
    df = pd.read_csv(RAW, encoding="utf-8-sig").rename(columns=COLS)
    n0 = len(df)
    parts = df["dia_chi"].astype(str).str.split(",").apply(lambda p: [x.strip() for x in p if x.strip()])
    n = parts.str.len()

    df["tinh"] = parts.str[-1].map(canon_province)
    bad_prov = df["tinh"].isna()
    too_short = n < 3                       # không đủ cấp quận/huyện
    drop = bad_prov | too_short

    df["quan_huyen"] = parts.str[-2]
    # n>=4: [.., phường/xã, quận, tỉnh]; n==3: phần đầu là phường/xã hoặc tên đường
    first_is_ward = parts.str[0].str.match(r"^(Phường|Xã|Thị trấn)\b", na=False)
    df["phuong_xa"] = np.where(n >= 4, parts.str[-3], np.where(first_is_ward, parts.str[0], None))
    df["duong"] = [", ".join(p[:-3]) if len(p) >= 4 else (None if w else p[0])
                   for p, w in zip(parts, first_is_ward)]
    df["la_du_an"] = df["dia_chi"].astype(str).str.startswith("Dự án").astype(int)

    df["phap_ly"] = df["phap_ly"].map(LEGAL)
    df["noi_that"] = df["noi_that"].map(FURNITURE)
    df["gia_m2"] = (df["gia_ty"] * 1000 / df["dien_tich"]).round(2)   # triệu/m²

    removed = df[drop]
    df = df[~drop]
    INTERIM.mkdir(parents=True, exist_ok=True)
    removed.to_csv(INTERIM / "01_removed.csv", index=False, encoding="utf-8-sig")
    n_q0, n_w0 = df.groupby("tinh")["quan_huyen"].nunique().sum(), df.groupby(["tinh", "quan_huyen"])["phuong_xa"].nunique().sum()
    df, merges = unify_unit_names(df)
    merges.to_csv(INTERIM / "01_unit_merges.csv", index=False, encoding="utf-8-sig")
    n_q1, n_w1 = df.groupby("tinh")["quan_huyen"].nunique().sum(), df.groupby(["tinh", "quan_huyen"])["phuong_xa"].nunique().sum()
    cols = ["dia_chi", "tinh", "quan_huyen", "phuong_xa", "duong", "la_du_an", "dien_tich", "mat_tien",
            "duong_vao", "huong_nha", "huong_ban_cong", "so_tang", "so_phong_ngu", "so_phong_tam",
            "phap_ly", "noi_that", "gia_ty", "gia_m2"]
    _write(df[cols], "01_standardized.csv")
    _log(1, "Chuẩn hoá cột, tách địa chỉ, chuẩn hoá tỉnh/thành", n0, len(df), {
        "bỏ vì tỉnh/thành không nhận diện được": int(bad_prov.sum()),
        "bỏ vì địa chỉ thiếu cấp quận/huyện": int((too_short & ~bad_prov).sum()),
        "số tỉnh/thành sau chuẩn hoá": int(df["tinh"].nunique()),
        "quận/huyện (theo tỉnh) trước -> sau khi hợp nhất cách viết": [int(n_q0), int(n_q1)],
        "phường/xã (theo tỉnh, quận) trước -> sau khi hợp nhất cách viết": [int(n_w0), int(n_w1)],
        "số dòng có tên quận/phường được hợp nhất": int(merges["so_dong"].sum()) if len(merges) else 0,
        "dòng có tên tỉnh được sửa so với bản gốc": int((df["dia_chi"].str.split(",").str[-1].str.strip()
                                                         != df["tinh"]).sum()),
    })


# ---------------------------------------------------------------- bước 2
DUP_KEY = ["dia_chi", "dien_tich", "gia_ty", "so_tang", "so_phong_ngu", "so_phong_tam", "mat_tien", "duong_vao"]
OPTIONAL = ["huong_nha", "huong_ban_cong", "phap_ly", "noi_that"]


def step2_deduplicate():
    """Loại trùng gần: giống nhau ở địa chỉ + diện tích + giá + tầng/phòng/mặt tiền/đường vào,
    chỉ khác ở các trường phụ (hướng, pháp lý, nội thất) -> coi là cùng tin đăng, giữ bản đầy đủ nhất.
    Không dùng khoá lỏng (địa chỉ + diện tích + giá): các căn trong cùng dự án có thể giống nhau thật."""
    df = _read("01_standardized.csv")
    n0 = len(df)
    exact = int(df.duplicated().sum())
    loose = int(df.duplicated(["dia_chi", "dien_tich", "gia_ty"]).sum())
    df["_filled"] = df[OPTIONAL + ["huong_ban_cong"]].notna().sum(axis=1)
    order = df.sort_values("_filled", ascending=False, kind="stable")
    dup = order.duplicated(DUP_KEY, keep="first")
    removed = order[dup].drop(columns="_filled")
    kept = df.loc[order[~dup].index].sort_index().drop(columns="_filled")
    removed.to_csv(INTERIM / "02_removed.csv", index=False, encoding="utf-8-sig")
    _write(kept, "02_deduplicated.csv")
    _log(2, "Loại bản ghi trùng gần", n0, len(kept), {
        "trùng tuyệt đối trong dữ liệu gốc": exact,
        "trùng theo khoá lỏng (địa chỉ+diện tích+giá), KHÔNG xoá": loose,
        "trùng theo khoá chặt (bị xoá)": int(dup.sum()),
    })


# ---------------------------------------------------------------- bước 3
def step3_filter():
    """Loại bản ghi ngoại lai hoặc vô lý (áp dụng lần lượt, mỗi dòng tính vào luật đầu tiên nó vi phạm)."""
    df = _read("02_deduplicated.csv")
    n0 = len(df)
    rules = [
        (f"giá/m² ngoài {GIA_M2_RANGE[0]}-{GIA_M2_RANGE[1]} triệu", ~df["gia_m2"].between(*GIA_M2_RANGE)),
        (f"diện tích ngoài {AREA_RANGE[0]}-{AREA_RANGE[1]} m²", ~df["dien_tich"].between(*AREA_RANGE)),
        (f"dưới {MIN_AREA_PER_BEDROOM} m² mỗi phòng ngủ", df["dien_tich"] / df["so_phong_ngu"] < MIN_AREA_PER_BEDROOM),
        (f"phòng tắm nhiều hơn phòng ngủ quá {MAX_BATH_OVER_BED}",
         df["so_phong_tam"] > df["so_phong_ngu"] + MAX_BATH_OVER_BED),
        (f"phòng ngủ > {MAX_BED_PER_FLOOR} × số tầng + {MAX_BED_PER_FLOOR}",
         df["so_phong_ngu"] > MAX_BED_PER_FLOOR * df["so_tang"] + MAX_BED_PER_FLOOR),
    ]
    reason = pd.Series("", index=df.index)
    notes = {}
    for name, bad in rules:
        hit = bad.fillna(False) & (reason == "")
        reason[hit] = name
        notes["bỏ vì " + name] = int(hit.sum())
    df[reason != ""].assign(ly_do=reason[reason != ""]).to_csv(
        INTERIM / "03_removed.csv", index=False, encoding="utf-8-sig")
    out = df[reason == ""]
    _write(out, "03_filtered.csv")
    _log(3, "Loại ngoại lai / bản ghi vô lý", n0, len(out), notes)


# ---------------------------------------------------------------- bước 4
def step4_fix_values():
    """Số đo bất thường (không thể là thật) đặt thành thiếu thay vì xoá cả dòng."""
    df = _read("03_filtered.csv")
    notes = {}
    bad = (df["mat_tien"] > MAX_FRONTAGE) | (df["mat_tien"] > df["dien_tich"] / 2)
    notes[f"mặt tiền > {MAX_FRONTAGE} m hoặc > nửa diện tích -> thiếu"] = int(bad.sum())
    df.loc[bad, "mat_tien"] = np.nan
    bad = df["duong_vao"] > MAX_ROAD
    notes[f"đường vào > {MAX_ROAD} m -> thiếu"] = int(bad.sum())
    df.loc[bad, "duong_vao"] = np.nan
    n0 = len(df)
    shallow = (df["dien_tich"] / df["mat_tien"]) < MIN_DEPTH     # mặt tiền còn lại nhưng lô đất nông vô lý
    notes[f"bỏ vì chiều sâu ước tính < {MIN_DEPTH} m (diện tích / mặt tiền)"] = int(shallow.sum())
    df[shallow].to_csv(INTERIM / "04_removed.csv", index=False, encoding="utf-8-sig")
    df = df[~shallow]
    _write(df, "04_values_fixed.csv")
    _log(4, "Sửa số đo bất thường thành thiếu; bỏ chiều sâu vô lý", n0, len(df), notes)


STEPS = {1: step1_standardize, 2: step2_deduplicate, 3: step3_filter, 4: step4_fix_values}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, choices=STEPS)
    a = ap.parse_args()
    if a.only is None and LOG.exists():
        LOG.unlink()
    for i in ([a.only] if a.only else STEPS):
        STEPS[i]()
