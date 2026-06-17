# Suspicious Review Detection on Tiki E-Commerce Platform

> **Data Collection and Preprocessing for Suspicious Review Detection on E-Commerce Platforms**  
> Môn học: DS108 – Data Science | Trường Đại học Công nghệ Thông tin – ĐHQG TP.HCM

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Dữ liệu](#dữ-liệu)
- [Cấu trúc repository](#cấu-trúc-repository)
- [Cài đặt môi trường](#cài-đặt-môi-trường)
- [Pipeline](#pipeline)
  - [Notebook 1 – Data Collection](#notebook-1--data-collection)
  - [Notebook 2 – EDA for Preprocessing](#notebook-2--eda-for-preprocessing)
  - [Notebook 3 – Data Cleaning & Validation](#notebook-3--data-cleaning--validation)
  - [Notebook 4 – Feature Engineering & EDA](#notebook-4--feature-engineering--eda)
  - [Notebook 5 – Merge](#notebook-5--merge)
  - [Notebook 6 – Labelling & Validation](#notebook-6--labelling--validation)
- [Kết quả](#kết-quả)
- [Nhóm thực hiện](#nhóm-thực-hiện)

---

## Tổng quan

Dự án xây dựng pipeline thu thập, tiền xử lý và gán nhãn giả (pseudo-labeling) cho tập dữ liệu đánh giá sản phẩm từ **Tiki** nhằm phục vụ bài toán phát hiện review đáng ngờ. Vì không có nhãn ground-truth, nhóm đề xuất một **two-regime semi-supervised labeling pipeline** tách biệt hai loại hành vi:

- **NC (No-Content):** Review chỉ có sao, không có nội dung – đặc trưng click-farm. Chấm điểm qua Behavioral (B) và Rating-pattern (C) features.  
- **HC (Has-Content):** Review có nội dung văn bản – đặc trưng astroturfing. Chấm điểm qua B, C và Textual (A) features kết hợp Isolation Forest.

Đầu ra cuối cùng là tập dữ liệu **584,005 records** với pseudo-label nhị phân, confidence score, consensus vote, temporal IPW weight và sample weight, sẵn sàng cho huấn luyện mô hình phân loại.

---

## Dữ liệu

> ⚠️ **Lưu ý quan trọng về dữ liệu**
>
> Do giới hạn dung lượng GitHub, thư mục `data/` trong repository này **chỉ chứa sample nhỏ** dùng để minh hoạ luồng chạy.  
> **Toàn bộ dữ liệu gốc và dữ liệu đã xử lý** được lưu trữ trên Google Drive:
>
> 📁 **[Truy cập Google Drive](https://drive.google.com/your-link-here)**  
>
> | Thư mục Drive | Nội dung |
> |---|---|
> | `raw/` | Dữ liệu thô crawl từ Tiki (`data2.1.csv`, ~719k records) |
> | `cleaned/` | `train_clean.csv`, `val_clean.csv`, `test_clean.csv` (584k records) |
> | `features/` | `train_merged_features.csv`, `val_merged_features.csv`, `test_merged_features.csv` |
> | `labelled/` | `labelled_dataset.csv` – file đầu ra cuối cùng (54 columns) |

Sau khi tải về từ Drive, đặt các file vào đúng thư mục theo hướng dẫn ở từng notebook.

---

## Cấu trúc repository

```
.
├── data/
│   ├── tiki_pt_review_sample.csv          # ← SAMPLE nhỏ (GitHub only)
│   └── data2.1_sample.csv                 # ← SAMPLE nhỏ (GitHub only)
│
├── outputs/                               # Output của Notebook 3
│   ├── train_clean.csv
│   ├── val_clean.csv
│   └── test_clean.csv
│
├── output/                                # Output của Notebook 5
│   ├── train_merged_features.csv
│   ├── val_merged_features.csv
│   └── test_merged_features.csv
│
├── 1._Data_Collection.ipynb
├── 2._EDA_For_Preprocessing.ipynb
├── 3._Data_Cleaning___Validation.ipynb
├── 4._Feature_Engineering___EDA.ipynb
├── 5._Merge.ipynb
├── 6._Labelling___Validation.ipynb
│
├── secrets.json.example                   # Template xác thực Tiki (không commit secrets thật)
├── requirements.txt
└── README.md
```

---

## Cài đặt môi trường

**Yêu cầu:** Python ≥ 3.9

```bash
pip install -r requirements.txt
```

Các thư viện chính:

```
pandas
numpy
scikit-learn
scipy
matplotlib
seaborn
requests
tqdm
```

---

## Pipeline

Chạy các notebook **theo đúng thứ tự** từ 1 đến 6. Mỗi notebook là một bước độc lập, đọc output của bước trước.

```
Tiki API
   │
   ▼
[1] Data Collection  ──►  raw CSV (tiki_pt_review_*.csv)
   │
   ▼
[2] EDA for Preprocessing  ──►  phân tích, không sửa dữ liệu
   │
   ▼
[3] Data Cleaning & Validation  ──►  outputs/{train,val,test}_clean.csv
   │
   ▼
[4] Feature Engineering & EDA  ──►  {user,review}_features_{train,val,test}.csv
   │
   ▼
[5] Merge  ──►  output/{train,val,test}_merged_features.csv
   │
   ▼
[6] Labelling & Validation  ──►  labelled_dataset.csv
```

---

### Notebook 1 – Data Collection

**File:** `1._Data_Collection.ipynb`  
**Input:** `secrets.json` (cookie + x-guest-token từ browser), từ khoá tìm kiếm  
**Output:** `data/tiki_pt_review_<keyword>.csv`

Crawl dữ liệu review từ internal JSON API của Tiki theo hai bước:

1. **Crawl Product ID** – Gọi Search API (`/api/v2/products`), phân trang tối đa 50 trang/keyword, lưu `product_id` ra CSV.
2. **Crawl Review** – Với mỗi `product_id`, gọi Review API phân trang đến trang trống; trích xuất `review_id`, `rating`, `content`, `timestamp`, `seller_id`, `customer_joined_time`, `customer_total_review`, `avatar`. Xử lý theo batch 200 sản phẩm để tránh mất data khi Colab timeout.

> **Lưu ý bảo mật:** Không commit `secrets.json`. Dùng file `secrets.json.example` làm template.  
> **Anti-crawling:** Dùng randomized delay giữa các request. Nếu nhận response rỗng, refresh cookie từ browser DevTools.

---

### Notebook 2 – EDA for Preprocessing

**File:** `2._EDA_For_Preprocessing.ipynb`  
**Input:** `data/data2.1.csv` (raw dataset)  
**Output:** Không tạo file mới – kết quả phân tích dùng để thiết kế Notebook 3

Phân tích thám hiểm dữ liệu bao gồm:

| Mục phân tích | Phát hiện chính |
|---|---|
| Schema & fill rate | 18 cột; `content` fill rate chỉ 36.4% (cấu trúc, không phải missing thông thường) |
| Missing values (MAR/MCAR/MNAR) | Pearson correlation với missingness indicator; tất cả |corr| < 0.10 |
| Outlier & skewness | `customer_total_review`: median=14, mean=42.5, max=2709 – right-skewed nặng |
| Duplication | 114,851 duplicate `review_id`; 81,915 duplicate (customer, product, content) |
| Text field parseability | `explain` parseability 99.96%; `customer_joined_time` 35 unique values (dạng categorical) |
| Temporal distribution | Volume tăng từ cuối 2020, đỉnh 2021–2022 |
| Content regime | No-content rate: 66.8% với 5-sao, 2.1% với 1-sao |
| Entity-level | Top 1% seller chiếm 87.2% tổng review |

---

### Notebook 3 – Data Cleaning & Validation

**File:** `3._Data_Cleaning___Validation.ipynb`  
**Input:** `data/data2.1.csv`  
**Output:** `outputs/train_clean.csv`, `outputs/val_clean.csv`, `outputs/test_clean.csv`

Pipeline gồm 5 class chính, **tất cả stateful transformation chỉ fit trên train set**:

```
SchemaValidator  →  DataCleaner  →  DataSplitter  →  OutlierHandler  →  MissingValueHandler
```

**DataCleaner** (row-wise, không dùng global statistics):
- `_cast_dtypes()` – ép kiểu từ object
- `_parse_dates()` – parse `review_created_date`, `delivery_date`
- `_filter_reviews_before_2020()` – loại 17,318 records cũ
- `_parse_explain_to_hours()` / `_parse_joined_time_to_months()` – parse text time fields
- `_drop_duplicates` (3 cấp: review_id → customer+product+content → normalized content)
- `_remove_invalid_ratings()`, `_nullify_impossible_counts()`, `_clean_text_content()`

**DataSplitter:** Customer-based split (70/15/15) – toàn bộ review của 1 customer nằm trong cùng 1 split, tránh user-level leakage.

**OutlierHandler:** `customer_total_review` dùng percentile-based (upper=457); `num_images`, `num_vote_agree` dùng domain bound. Mode REMOVE (ratio 0.25% < ngưỡng 8%).

**MissingValueHandler:** Phân loại MNAR/MAR/MCAR per-column; fill value khác nhau (MNAR numeric → -999, MNAR categorical → "Missing", MAR → median/mode).

**Post-cleaning Validation** (cùng notebook, phần 2):  
40 checks chia 4 phase: Row-wise invariants → Split integrity → Outlier bounds & missing sentinels → Cross-split distribution drift. **Kết quả: 40/40 PASS, 0 warning.**

---

### Notebook 4 – Feature Engineering & EDA

**File:** `4._Feature_Engineering___EDA.ipynb`  
**Input:** `{train,val,test}_clean.csv`  
**Output:** `user_features_{train,val,test}.csv`, `review_features_{train,val,test}.csv`

Ba nhóm feature được tính trên cả 3 split:

**B – Behavioral features** (tất cả review):

| Feature | Mô tả |
|---|---|
| B1 `reviews_per_day` | Số review trung bình/ngày của customer |
| B2 `time_gap_std` | Độ lệch chuẩn khoảng thời gian giữa các review |
| B3 `burst_flag` / `max_reviews_in_one_day` | Flag khi post ≥ N review/ngày |
| B4 `account_age_days` | Tuổi tài khoản (ngày) |
| B5 `customer_total_review` | Tổng số review của tài khoản |
| B6 `avatar_default` | Indicator dùng avatar mặc định |

**C – Rating-pattern features** (tất cả review):

| Feature | Mô tả |
|---|---|
| C1 `rating_variance` | Phương sai rating (nghịch đảo – uniform → suspicious) |
| C2 `percent_extreme_rating` | Tỉ lệ rating 1-sao hoặc 5-sao |
| C3 `single_brand_focus_ratio` | Tỉ lệ review tập trung vào 1 seller |
| C4 `review_length_variance` | Phương sai độ dài review (nghịch đảo) |

**A – Textual features** (chỉ HC reviews):

| Feature | Mô tả |
|---|---|
| A1 `promo_keyword_ratio` | Tỉ lệ keyword quảng cáo trong nội dung |
| A2 `external_link_flag` | Có chứa URL ngoài |
| A3 `contact_info_flag` | Có số điện thoại / Zalo / Telegram |
| A4 `excessive_punctuation` | Tỉ lệ dấu câu bất thường |
| A5 `capitalization_ratio` | Tỉ lệ chữ hoa |
| A6 `emoji_density` | Mật độ emoji trong văn bản |
| A7 `word_repetition_ratio` | Tỉ lệ lặp từ (lexical diversity thấp) |

---

### Notebook 5 – Merge

**File:** `5._Merge.ipynb`  
**Input:** `{split}_clean.csv` + `user_features_{split}.csv` + `review_features_{split}.csv`  
**Output:** `output/{train,val,test}_merged_features.csv`

Gộp bảng review đã làm sạch với user-level features và review-level features theo `customer_id`. Thực hiện độc lập cho từng split (train → test → val) để không lẫn thông tin.

---

### Notebook 6 – Labelling & Validation

**File:** `6._Labelling___Validation.ipynb`  
**Input:** `{train,val,test}_merged_features.csv`  
**Output:** `train_dataset.csv`, `val_dataset.csv`, `test_dataset.csv`, `labelled_dataset.csv`

**Two-regime pseudo-labeling pipeline:**

```
Features (B, C, A)
       │
       ├─── NC regime (65.1% train) ──► SNC-pre = 0.6·B̃ + 0.4·C̃
       │         │                            │
       │    Quantile-cut pseudo-labels   LR scoring → SNC = σ(β₀ + βB·B + βC·C)
       │    (contamination=15%)          βB=11.09, βC=65.39, β₀=−29.95
       │
       └─── HC regime (34.9% train) ──► Isolation Forest([B, A], contamination=15%)
                 │                            │
            IF pseudo-labels            LR scoring → SHC = σ(β₀ + βB·B + βC·C + βA·A)
                                        βB=2.21, βC=2.98, βA=39.87, β₀=−4.19
```

**Threshold calibration** (`calibrate_threshold_combined()`):  
Chấp nhận threshold khi `prec_vs_pseudo ≥ 0.90` và `near-threshold rate < 15%`.  
→ `τNC = 0.1242` (prec=0.90, recall=1.00) | `τHC = 0.4846` (prec=0.90, recall=0.57)

**Bổ sung outputs:**
- `label_confidence` – khoảng cách normalised từ ngưỡng → [0, 1]
- `soft_label` – vote_count/5 từ consensus sweep (contamination ∈ {0.10, 0.15, 0.20, 0.25, 0.30})
- `temporal_weight` – IPW correction năm (ptarget=0.1168 từ 2020–2022)
- `uncertainty_weight` – consensus confidence
- `sample_weight` = temporal × uncertainty (ESS = 90.2%)

> ⚠️ **Quan trọng:** Các cột `B_score`, `C_score`, `A_score`, `suspicious_score` là output của quá trình gán nhãn. **Không dùng các cột này làm input cho mô hình** – sẽ gây label-feature circularity và inflate performance giả tạo.

---

## Kết quả

### Thống kê dataset cuối

| Split | Records | Label=1 (Suspicious) | NC label=1 | HC label=1 |
|---|---|---|---|---|
| Train | 409,333 | 14.1% | 16.7% | 9.4% |
| Validation | 87,526 | 14.5% | 17.1% | 9.5% |
| Test | 87,146 | 13.9% | 16.4% | 9.2% |
| **Total** | **584,005** | **~14%** | — | — |

### Cấu trúc file `labelled_dataset.csv`

54 cột, bao gồm:
- **Identifiers:** `review_id`, `customer_id`, `product_id`, `seller_id`, `split`  
- **Features thô:** `rating`, `content`, `content_length`, `num_images`, `num_vote_agree`, ...  
- **Features kỹ thuật:** B1–B6, C1–C4, A1–A7 (17 features)  
- **Labelling outputs:** `label`, `regime`, `suspicious_score`, `label_confidence`, `vote_count`, `soft_label`, `temporal_weight`, `uncertainty_weight`, `sample_weight`

### Validation pseudo-labels

| Kiểm tra | Kết quả |
|---|---|
| Threshold stability (±0.05) | Flip rate < 1% |
| Near-threshold rate | 1.2% (target < 15%) |
| Burst users suspicious rate | 60.5% (RR = 4.28×) |
| High RPD users suspicious rate | 72.2% (RR = 5.11×) |
| IF seed robustness (Fleiss' κ) | 0.942 |
| HC Jaccard mean (seed sweep) | 0.956 |
| Contamination sensitivity (Jaccard) | Mean 0.604 ← remaining limitation |
| Effective Sample Size | 90.2% |

---

## Nhóm thực hiện

| Họ và tên | MSSV | Email |
|---|---|---|
| Phạm Minh Tuấn | 24521938 | 24521938@gm.uit.edu.vn |
| Đặng Nguyễn Tú Trinh | 24521858 | 24521858@gm.uit.edu.vn |

**Giảng viên hướng dẫn:**  
- TS. Nguyễn Gia Tuấn Anh – anhngt@uit.edu.vn  
- Trần Quốc Khánh, BSc – khanhtq@uit.edu.vn

---

## Tài liệu tham khảo

1. M. Ott et al., "Finding deceptive opinion spam by any stretch of the imagination," *ACL 2011*
2. A. Mukherjee et al., "What yelp fake review filter might be doing?" *ICWSM 2013*
3. L. Akoglu et al., "Opinion fraud detection in online reviews by network effects," *ICWSM 2013*
4. G. Fei et al., "Exploiting burstiness in reviews for review spammer detection," *ICWSM 2013*
5. F. Li et al., "Learning to identify review spam," *IJCAI 2011*
6. S. Feng et al., "Distributional footprints of deceptive product reviews," *ICWSM 2012*
7. Z. Hadi et al., "Detect fake reviews using random forest and support vector machine," *Sinkron 2023*
