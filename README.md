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
- **Cái ta dự đoán:** giá trên mỗi m² = giá × 1000 / diện tích (triệu đồng/m²). Dùng giá/m² để so sánh công bằng giữa nhà to và nhà nhỏ.
- Hai file khác trong `original-dataset/` (`VN_housing_dataset.csv`, `VN-housing-dataset-refined.csv`) là bộ dữ liệu cũ, **không còn được dùng**.
- **Điểm cần biết:** dữ liệu chỉ có nhà giá tối đa khoảng **11,5 tỷ** (99% dưới 10 tỷ). Mô hình vì vậy không dự đoán được biệt thự hay nhà cao cấp.

---

## 3. Tiền xử lý dữ liệu

Dữ liệu thô có nhiều lỗi: địa chỉ viết lộn xộn, tin đăng trùng, số liệu vô lý. Việc làm sạch được chia thành **6 bước**, mỗi bước đọc file của bước trước và ghi ra một file riêng trong `data/interim/` để bạn mở ra xem được từng bước. Các dòng bị loại ở mỗi bước được lưu trong `*_removed.csv`, và thống kê nằm trong `data/interim/preprocessing_log.json`.

| Bước | Làm gì | File ra | Số dòng |
|---|---|---|---|
| (gốc) | | `vietnam_housing_dataset.csv` | 30.229 |
| 1. Chuẩn hoá | Đổi tên cột; tách địa chỉ thành tỉnh / quận / phường / đường; sửa tên tỉnh viết sai ("Hà Nội.", "TP Hồ Chí Minh"...); tính giá/m². Bỏ 11 dòng địa chỉ không xác định được tỉnh | `01_standardized.csv` | 30.218 |
| 2. Loại trùng | Bỏ các tin đăng lại (giống nhau ở địa chỉ, diện tích, giá, số tầng, số phòng, mặt tiền, đường vào); giữ bản đầy đủ nhất | `02_deduplicated.csv` | 29.976 |
| 3. Loại ngoại lai | Bỏ giá/m² ngoài 10–400 triệu, diện tích ngoài 10–300 m², nhà có dưới 4 m² mỗi phòng ngủ | `03_filtered.csv` | 29.650 |
| 4. Sửa số đo | Mặt tiền > 30 m hoặc đường vào > 40 m là không thể đúng, chuyển thành "thiếu" thay vì xoá cả dòng | `04_values_fixed.csv` | 29.650 |
| 5. Đặc trưng mới | Thêm vài cột tính bằng công thức: diện tích mỗi tầng, chiều sâu ước tính (diện tích ÷ mặt tiền), tỉ lệ phòng tắm trên phòng ngủ | `05_features.csv` | 29.650 |
| 6. Chia tập | Chia 70% **train** (học) / 15% **val** (chọn mô hình) / 15% **test** (chấm điểm cuối) | `data/splits/` | 20.752 / 4.449 / 4.449 |

Hai điểm đáng nói về bước loại trùng và chia tập:

- **Không xoá mọi dòng "giống nhau".** Trong cùng một dự án chung cư có nhiều căn khác nhau nhưng giá niêm yết như nhau. Chỉ xoá khi cả cấu hình căn (tầng, phòng, mặt tiền, đường vào) cũng giống hệt.
- **Chia tập có chống rò rỉ.** Các tin cùng địa chỉ + diện tích + giá luôn nằm chung một tập, nếu không mô hình "thấy" căn đó lúc học rồi gặp lại lúc thi, điểm sẽ cao giả. Ngoài ra, mỗi tập có tỉ lệ các tỉnh giống nhau.

**Phần xử lý còn lại nằm trong mô hình** (`src/features.py`), vì chúng phải "học" từ tập train rồi áp lên val/test cho công bằng:
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
| Gradient Boosting | 20 | `learning_rate` = 0,05, `max_leaf_nodes` = 127, `l2_regularization` = 1 |

Lưới tham số khai báo ở `src/models.py`; muốn thêm mô hình chỉ cần thêm vào đó.

---

## 5. Kết quả và nằm ở đâu

### Điểm trên tập test (chưa từng dùng để chọn gì)

| Mô hình | MAE (triệu/m²) | R² | MAPE |
|---|---|---|---|
| Trung vị (mốc) | 44,9 | −0,01 | 65,3% |
| Ridge | 18,7 | 0,766 | 19,4% |
| Random Forest | 17,2 | 0,790 | 17,8% |
| **Gradient Boosting** | **16,7** | **0,804** | **17,0%** |

Cách đọc nhanh:
- **MAE**: trung bình mô hình đoán lệch bao nhiêu triệu/m² (giá trung vị của dữ liệu khoảng 103). Càng thấp càng tốt.
- **MAPE**: lệch trung bình bao nhiêu phần trăm so với giá thật. Càng thấp càng tốt.
- **R²**: mô hình giải thích được bao nhiêu phần sự khác nhau về giá giữa các căn. 0 là không hơn đoán trung vị, 1 là hoàn hảo.

### Điều rút ra

- Ba mô hình học được đều **vượt xa** việc đoán trung vị. Gradient Boosting hơn Ridge rõ ràng; so với Random Forest thì chênh lệch quá nhỏ để kết luận chắc.
- **Yếu tố quyết định giá nhất: diện tích, tỉnh, quận.** Chỉ với tỉnh + quận + diện tích đã đạt R² ≈ 0,71. Số tầng và đường vào có ích; hướng nhà, nội thất, pháp lý gần như không đáng kể.
- Mô hình đoán tốt nhất với nhà giá trung bình, kém hơn với nhà giá rất thấp và nhà rất nhỏ ở trung tâm Hà Nội (thường bị đoán thấp hơn giá thật).
- R² riêng từng nhóm thấp hơn con số tổng (Hồ Chí Minh 0,75; Hà Nội 0,60; các tỉnh khác 0,74), vì một phần lớn chênh lệch giá nằm giữa các tỉnh với nhau.

### File kết quả

| Muốn xem | Mở file |
|---|---|
| Điểm số mọi mô hình (val và test, cả theo nhóm tỉnh, mọi cấu hình đã thử) | `server/models/metrics.json` |
| Điểm test chi tiết: khoảng tin cậy 95%, sai số theo khoảng giá và diện tích, 8 ca sai nặng nhất | `results/test_results.json` |
| Độ quan trọng từng đặc trưng; điểm khi bỏ từng nhóm đặc trưng | `results/analysis.json` |
| Thống kê số dòng bị loại ở mỗi bước tiền xử lý | `data/interim/preprocessing_log.json` |
| Các dòng bị loại ở bước 1, 2, 3 | `data/interim/01_removed.csv`, `02_removed.csv`, `03_removed.csv` |
| Bảng so sánh trên trang web | chạy web, mục "So sánh mô hình" |

---

## 6. Cấu trúc thư mục

```
original-dataset/   dữ liệu thô (chỉ vietnam_housing_dataset.csv được dùng)
data/
  interim/          kết quả từng bước tiền xử lý + log + các dòng bị loại
  splits/           train.csv, val.csv, test.csv
src/
  preprocess.py     tiền xử lý, bước 1-5
  split.py          bước 6: chia train/val/test
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
```

---

## 7. Chạy lại toàn bộ từ đầu

Chạy theo thứ tự (từ thư mục gốc, đã kích hoạt `.venv`):

```bash
python -m src.preprocess     # 1. tiền xử lý, bước 1-5 -> data/interim/
python -m src.split          # 2. chia train/val/test -> data/splits/
python -m src.train          # 3. huấn luyện 4 mô hình, chọn tham số trên val (khoảng 15-20 phút)
python -m src.analyze        # 4. phân tích đặc trưng (val) -> results/analysis.json
python -m src.final_eval     # 5. chấm điểm cuối trên test -> results/test_results.json
python -m src.make_options   # 6. cập nhật dữ liệu cho giao diện web
```

- Bước 3 có thể rút gọn: `python -m src.train --only ridge hgb` (bỏ Random Forest, nhanh hơn và không tạo file 545 MB).
- Các hạt giống ngẫu nhiên đã cố định nên kết quả chạy lại gần như trùng.
- **Sau khi huấn luyện lại, hãy chạy lại bước 5** để điểm test được cập nhật cho mô hình mới. Tập test nên được coi là chỉ chấm một lần cho mỗi lần chốt mô hình; đừng chỉnh tham số dựa trên điểm test.

---

## 8. Hạn chế cần biết

- Dữ liệu là **giá rao bán**, không phải giá giao dịch thực.
- Dữ liệu **giới hạn ở khoảng 11,5 tỷ**, nên kết quả vượt mức này không đáng tin (giao diện có cảnh báo).
- **Thiếu thông tin về loại nhà** (mặt phố, trong ngõ, biệt thự). Đây có vẻ là lý do chính mô hình đoán thấp với nhà nhỏ ở khu trung tâm Hà Nội.
- Hồ Chí Minh và Hà Nội chiếm khoảng 75% dữ liệu; **43 trên 58 tỉnh có dưới 100 tin**, nên kết quả ở các tỉnh nhỏ chỉ mang tính tham khảo.
- Dữ liệu không có ngày đăng nên chưa kiểm tra được mô hình theo thời gian (giá nhà thay đổi theo năm).
