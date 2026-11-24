# Dự đoán giá nhà Việt Nam

Dự án Machine Learning: nhập thông tin một căn nhà (tỉnh, quận, phường, diện tích, số tầng...) và mô hình ước tính **giá bán theo triệu đồng/m²**. Có kèm một trang web để thử trực tiếp.

**Kết quả tóm tắt:** mô hình tốt nhất (Gradient Boosting) dự đoán đúng trong khoảng ±17% so với giá thật, và giải thích được khoảng 80% sự biến động của giá (R² = 0,80) trên dữ liệu chưa từng thấy.

## Mục lục

1. [Chạy thử nhanh](#1-chạy-thử-nhanh)
2. [Dữ liệu](#2-dữ-liệu)
3. [Tiền xử lý dữ liệu](#3-tiền-xử-lý-dữ-liệu)
4. [Các mô hình](#4-các-mô-hình)
5. [Kết quả và nằm ở đâu](#5-kết-quả-và-nằm-ở-đâu)
6. [Cấu trúc thư mục](#6-cấu-trúc-thư-mục)
7. [Chạy lại toàn bộ từ đầu](#7-chạy-lại-toàn-bộ-từ-đầu)
8. [Hạn chế cần biết](#8-hạn-chế-cần-biết)

---

## 1. Chạy thử nhanh

Yêu cầu: Python 3.11 trở lên (dự án được làm trên 3.14).

```bash
git clone <địa-chỉ-repo>
cd house-price-ml

python -m venv .venv
.venv\Scripts\activate            # Windows. macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**Lưu ý quan trọng:** repo **không chứa file mô hình** (`*.joblib`, vì có file nặng 545 MB). Lần đầu bạn cần tạo chúng:

```bash
python -m src.preprocess          # làm sạch dữ liệu (vài giây)
python -m src.split               # chia train/val/test (vài giây)
python -m src.train --only ridge hgb   # huấn luyện (khoảng 10-15 phút, chủ yếu là Gradient Boosting)
```

Sau đó chạy web:

```bash
uvicorn server.app:app --reload
```

Mở http://127.0.0.1:8000, chọn tỉnh/quận/phường, nhập diện tích rồi bấm **Dự đoán giá**. Trang cũng hiển thị bảng so sánh các mô hình. Tài liệu API tự sinh có ở http://127.0.0.1:8000/docs.

Muốn dùng mô hình khác: đặt biến môi trường `MODEL` là `hgb` (mặc định, Gradient Boosting), `rf` (Random Forest) hoặc `ridge`.
Ví dụ Windows: `set MODEL=ridge` rồi chạy `uvicorn`.

> Nếu không chạy server, giao diện vẫn mở được (ví dụ `python -m http.server 8000 -d web`) nhưng nút dự đoán chỉ cho ước tính tạm theo giá trung vị khu vực, có ghi rõ trên màn hình.

---

## 2. Dữ liệu

- **File dùng:** `original-dataset/vietnam_housing_dataset.csv` — 30.229 tin rao bán nhà trên cả nước (chủ yếu Hồ Chí Minh và Hà Nội).
- **Các cột:** địa chỉ, diện tích, mặt tiền, đường vào, hướng nhà, hướng ban công, số tầng, số phòng ngủ, số phòng tắm, pháp lý, nội thất và **giá (tỷ đồng, cho cả căn)**.
- **Dự đoán:** giá trên mỗi m² = giá × 1000 / diện tích (triệu đồng/m²). Dùng giá/m² để so sánh công bằng giữa nhà to và nhà nhỏ.
- **Điểm cần biết:** dữ liệu chỉ có nhà giá tối đa khoảng **11,5 tỷ** (99% dưới 10 tỷ). Mô hình vì vậy không dự đoán được biệt thự hay nhà cao cấp.

---

## 3. Tiền xử lý dữ liệu

Dữ liệu thô có nhiều lỗi: địa chỉ viết lộn xộn, tin đăng trùng, số liệu vô lý. Việc làm sạch được chia thành **5 bước** (4 bước làm sạch ghi vào `data/interim/`, bước 5 là chia tập). 

| Bước | Làm gì | File ra | Số dòng |
|---|---|---|---|
| (gốc) | | `vietnam_housing_dataset.csv` | 30.229 |
| 1. Chuẩn hoá | Đổi tên cột; tách địa chỉ thành tỉnh / quận / phường / đường; sửa tên tỉnh viết sai ("Hà Nội.", "TP Hồ Chí Minh"...); tính giá/m². Bỏ 11 dòng địa chỉ không xác định được tỉnh | `01_standardized.csv` | 30.218 |
| 2. Loại trùng | Bỏ các tin đăng lại (giống nhau ở địa chỉ, diện tích, giá, số tầng, số phòng, mặt tiền, đường vào); giữ bản đầy đủ nhất | `02_deduplicated.csv` | 29.976 |
| 3. Loại ngoại lai | Bỏ giá/m² ngoài 10–400 triệu, diện tích ngoài 10–300 m², nhà có dưới 4 m² mỗi phòng ngủ | `03_filtered.csv` | 29.650 |
| 4. Sửa số đo | Mặt tiền > 30 m hoặc đường vào > 40 m là không thể đúng, chuyển thành "thiếu" thay vì xoá cả dòng | `04_values_fixed.csv` | 29.650 |
| 5. Chia tập | Chia 70% **train** (học) / 15% **val** (chọn mô hình) / 15% **test** (chấm điểm cuối), phân tầng theo tỉnh và chia theo nhóm tin đăng giống nhau để tránh rò rỉ | `data/splits/` | 20.752 / 4.449 / 4.449 |


**Phần xử lý còn lại nằm trong mô hình** (`src/features.py`):
- diện tích đi vào mô hình dưới dạng `log(1 + diện tích)`;
- điền giá trị thiếu bằng trung vị và thêm cột đánh dấu "ô này vốn thiếu";
- mã hoá chữ thành số (one-hot) cho tỉnh, quận, phường, pháp lý, nội thất, hướng nhà; giá trị quá hiếm (dưới 20 lần) gộp chung;
- học trên `log(giá/m²)` vì giá lệch về phía cao.

---

## 4. Các mô hình

Cả bốn dùng chung một pipeline đặc trưng, chỉ khác bước dự đoán cuối.

| Mô hình | Giải thích đơn giản | Vai trò |
|---|---|---|
| **Trung vị** | Luôn đoán giá/m² ở mức giữa của tập train | Mốc để biết mô hình có thật sự học được gì không |
| **Ridge Regression** | Đường thẳng đa chiều: giá = tổng các yếu tố nhân hệ số | Mô hình đơn giản, dễ giải thích |
| **Random Forest** | Hàng trăm cây quyết định học độc lập rồi lấy trung bình | Bắt được quan hệ phức tạp |
| **Gradient Boosting** | Các cây nhỏ nối tiếp nhau, cây sau sửa lỗi của cây trước | Thường mạnh nhất với dữ liệu dạng bảng |

**Cách chọn tham số:** mỗi mô hình thử nhiều cấu hình (lưới tham số) và chọn cấu hình có sai số thấp nhất trên tập **val**. Tập **test** chỉ được chạy **một lần** ở cuối, để điểm cuối cùng khách quan. Cấu hình được chọn:

| Mô hình | Số cấu hình thử | Tham số chọn |
|---|---|---|
| Ridge | 6 | `alpha` = 1 |
| Random Forest | 9 | 300 cây, `min_samples_leaf` = 1, `max_features` = 0,6 |
| Gradient Boosting | 20 | `learning_rate` = 0,05, `max_leaf_nodes` = 255, `l2_regularization` = 1 |

Lưu ý: `max_leaf_nodes` = 255 là giá trị lớn nhất trong lưới, nên chưa biết giá trị lớn hơn có tốt hơn không (mức giảm sai số giữa các mức đã nhỏ dần: MAE val 16,65 ở 127 và 16,57 ở 255).


---

## 5. Kết quả

### Điểm trên tập test 

| Mô hình | MAE (triệu/m²) | R² | MAPE |
|---|---|---|---|
| Trung vị (mốc) | 44,9 | −0,01 | 65,3% |
| Ridge | 18,8 | 0,766 | 19,5% |
| Random Forest | 16,9 | 0,796 | 17,6% |
| **Gradient Boosting** | **16,8** | **0,801** | **16,9%** |

Gradient Boosting hơn Ridge có ý nghĩa thống kê. So với Random Forest, chênh lệch rất nhỏ (MAE 0,12 triệu/m², khoảng tin cậy chồng lên nhau) nên chưa kết luận được mô hình nào tốt hơn. Gradient Boosting được chọn vì thắng trên val và file mô hình nhẹ hơn nhiều (14 MB so với 545 MB).

Quy đổi sang tổng giá cả căn (nhân với diện tích), trên cùng tập test:

| Mô hình | MAE (tỷ đồng) | RMSE (tỷ đồng) | R² |
|---|---|---|---|
| Mốc: trung vị giá cả căn | 1,85 | 2,22 | 0,000 |
| Ridge | 1,04 | 1,40 | 0,599 |
| Random Forest | 0,93 | 1,28 | 0,665 |
| **Gradient Boosting** | **0,92** | **1,27** | **0,673** |

MAPE giữ nguyên như bảng trên (diện tích triệt tiêu trong tỉ số sai số). R² thấp hơn ở thang tỷ đồng vì giá cả căn bị chặn trong khoảng hẹp (1–11,5 tỷ), nên sai số còn lại chiếm tỉ lệ lớn hơn so với phương sai của nó.

Chú thích:
- **MAE**: trung bình mô hình đoán lệch bao nhiêu triệu/m² (giá trung vị của dữ liệu khoảng 103).
- **MAPE**: lệch trung bình bao nhiêu phần trăm so với giá thật.
- **R²**: mô hình giải thích được bao nhiêu phần sự khác nhau về giá giữa các căn. 0 là không hơn đoán trung vị, 1 là hoàn hảo.


---

## 6. Cấu trúc thư mục

```
original-dataset/   dữ liệu thô (chỉ vietnam_housing_dataset.csv được dùng)
data/
  interim/          kết quả từng bước tiền xử lý (01-04) + log + các dòng bị loại
  splits/           train.csv, val.csv, test.csv
src/
  preprocess.py     tiền xử lý, bước 1-4
  split.py          bước 5: chia train/val/test
  features.py       pipeline đặc trưng dùng chung cho mọi mô hình
  models.py         danh sách mô hình và lưới tham số
  evaluate.py       các chỉ số đánh giá (MAE, RMSE, R², MAPE)
  train.py          huấn luyện + chọn tham số trên val
  analyze.py        độ quan trọng đặc trưng, thử bỏ từng nhóm đặc trưng
  final_eval.py     chấm điểm cuối trên test
  make_options.py   sinh dữ liệu cho giao diện web
server/
  app.py            API (FastAPI), đồng thời phục vụ giao diện web
  models/           mô hình đã huấn luyện (*.joblib, không có trong git) và metrics.json
web/                giao diện (HTML + CSS + JS thuần, không cần build)
results/            kết quả phân tích và đánh giá test (JSON)
  audit/            kiểm tra độc lập: tái lập điểm test, kết quả ở thang tỷ đồng, biểu đồ thực tế vs dự đoán và phần dư
```

---

## 7. Chạy lại toàn bộ từ đầu

Chạy theo thứ tự (từ thư mục gốc, đã kích hoạt `.venv`):

```bash
python -m src.preprocess     # 1. tiền xử lý, bước 1-4 -> data/interim/
python -m src.split          # 2. chia train/val/test (bước 5) -> data/splits/
python -m src.train          # 3. huấn luyện 4 mô hình, chọn tham số trên val (khoảng 15-20 phút)
python -m src.analyze        # 4. phân tích đặc trưng (val) -> results/analysis.json
python -m src.final_eval     # 5. chấm điểm cuối trên test -> results/test_results.json
python -m src.make_options   # 6. cập nhật dữ liệu cho giao diện web
```

- Bước 3 có thể rút gọn: `python -m src.train --only ridge hgb` (bỏ Random Forest, nhanh hơn và không tạo file 545 MB).
- Các hạt giống ngẫu nhiên đã cố định nên kết quả chạy lại gần như trùng.
- **Sau khi huấn luyện lại, hãy chạy lại bước 5** để điểm test được cập nhật cho mô hình mới. Tập test nên được coi là chỉ chấm một lần cho mỗi lần chốt mô hình; đừng chỉnh tham số dựa trên điểm test.

---
