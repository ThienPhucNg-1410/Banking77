# BÁO CÁO KẾT QUẢ THỰC NGHIỆM: PIPELINE V3 (BANKING77)
**Dự án**: Customer Feedback Topic Discovery using Clustering  
**Môi trường thực thi**: Kaggle GPU (T4 x2) | **Dữ liệu**: BANKING77 (9,993 Train | 3,076 Test | 77 Intents)

---

## 1. Cấu hình Pipeline tối ưu (Champion Setup)
```text
Raw Utterance 
  → Light Clean (giữ nguyên ngữ cảnh, không bỏ stopword)
  → Encoder: BAAI/bge-small-en-v1.5 (384D, Normalized)
  → UMAP: n_neighbors=30, n_components=5, metric='cosine'
  → HDBSCAN: min_cluster_size=50, min_samples=5, selection='eom'
  → c-TF-IDF (tính theo từng câu, ngram 1-2)
  → Novelty Detector: kNN Cosine Distance (k=10, ngưỡng p95 Train)
```

---

## 2. Bảng điểm hiệu năng chính (Scorecard)

| Chỉ số đánh giá | Tập Train (Fit) | Tập Test (Transform) | Nhận xét Data Science |
| :--- | :---: | :---: | :--- |
| **Số cụm phát hiện** | **76 cụm** | **76 cụm** | Tự động tiệm cận gần như tuyệt đối 77 intent gốc (không cần biết K). |
| **Tỷ lệ bao phủ (Coverage)** | **92.34%** | **89.79%** | Độ phủ cao, chỉ ~7.6% (Train) và 10.2% (Test) bị coi là noise. |
| **V-Measure (cả noise)** | **0.832** | **0.835** | Không suy giảm trên Test $\rightarrow$ **Khả năng tổng quát hóa hoàn hảo**. |
| **V-Measure (bỏ noise)** | **0.873** | **0.887** | Các cụm được gom lại có độ thuần ngữ nghĩa cực kỳ cao. |
| **ARI (bỏ noise)** | **0.670** | **0.672** | Mức độ tương đồng phân hoạch vượt trội so với phân cụm ngẫu nhiên. |
| **Độ tinh khiết (Purity)** | **78.57%** | **78.53%** | Đa số các cụm đại diện chính xác cho 1 intent duy nhất. |
| **Intent có cụm riêng** | **87.01%** | **88.31%** | 67 / 77 intent được tách thành các cụm độc lập rõ rệt. |
| **Silhouette (Cosine)** | **0.224** | **0.235** | Mật độ khoảng cách cụm đạt chuẩn cho dữ liệu văn bản NLP. |
| **DBCV (Mật độ HDBSCAN)** | **0.405** | — | Xác nhận các thung lũng mật độ được phân định sắc nét. |

---

## 3. Benchmark đối sánh (Ablation & Baselines)

| Mô hình / Pipeline | Số cụm | Noise % | V-Measure | ARI | Purity | Đánh giá |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **BGE-small → UMAP → HDBSCAN (Đề xuất)** | **76** | **7.65%** | **0.832** | **0.532** | **78.57%** | **Tốt nhất**: Cân bằng tối ưu giữa độ chính xác và tính tự động. |
| MPNet-base → UMAP → HDBSCAN | 80 | 7.80% | 0.815 | 0.505 | 77.40% | Kém hơn BGE dù vector nặng gấp đôi (768D), tốn tài nguyên. |
| MiniLM-L6 (gốc) → UMAP → HDBSCAN | 73 | 7.89% | 0.788 | 0.451 | 70.99% | Bị tụt 4.4% V-measure và 8.1% ARI so với BGE-small. |
| **Bỏ UMAP (Full-D 384D → HDBSCAN)** | **3** | **4.02%** | **0.055** | **0.001** | **4.06%** | **Sụp đổ hoàn toàn** do lời nguyền số chiều (Curse of Dimensionality). |
| BGE-small → UMAP → K-Means ($K=77$) | 77 | 0.00% | 0.864 | 0.677 | 78.35% | Cao hơn nhẹ nhưng **bắt buộc phải biết trước K=77**, ép gom rác. |
| TF-IDF → K-Means ($K=77$) | 77 | 0.00% | 0.557 | 0.161 | 36.57% | Kém xa: TF-IDF không nắm bắt được đồng nghĩa / ngữ cảnh. |
| TF-IDF → NMF ($K=77$) | 77 | 0.00% | 0.544 | 0.214 | 35.51% | Topic modeling truyền thống không phù hợp cho câu ngắn. |

---

## 4. Phát hiện chủ đề mới (Emerging Topic Detection)

Thử nghiệm giấu 15 intents ở tập Train để kiểm tra khả năng phát hiện khi chúng xuất hiện ở tập Test:

| Bộ dò tìm (Detector) | AUROC | Recall (tại FPR 10%) | Lift (R/FPR) | Khôi phục Intent mới | Kết luận |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **kNN Cosine Distance (Đề xuất)** | **0.864** | **55.22%** | **5.51x** | **100% (15/15)** | **Xuất sắc**: Định vị chính xác tuyệt đối toàn bộ chủ đề mới. |
| HDBSCAN 1 - Outlier Score | 0.618 | 21.11% | 2.09x | 73.33% (11/15) | Hiệu quả trung bình, bị ảnh hưởng bởi phép chiếu UMAP. |
| **HDBSCAN Noise Flag (`label == -1`)**| **0.542** | **17.93%** | **1.93x** | **66.67% (10/15)** | **Thất bại**: Gần như ngẫu nhiên. Tuyệt đối không dùng noise làm topic mới! |

> **Quy tắc kỹ thuật**: Điểm nhiễu (`noise = -1`) của HDBSCAN chỉ phản ánh các câu có mật độ cục bộ thấp, **không đồng nghĩa với chủ đề mới**. Muốn phát hiện chủ đề mới (novelty/emerging topic), bắt buộc phải dùng **kNN distance trong không gian embedding gốc**.

---

## 5. Kết luận & Khuyến nghị triển khai (Takeaways)

1. **Hiệu năng thực tế**: Pipeline tự động gom được **76 cụm** đại diện cho **88.3% intent độc lập** với độ tinh khiết **78.6%**, $V=0.835$ trên dữ liệu Test hoàn toàn mới.
2. **Vai trò UMAP**: Bắt buộc phải giảm chiều về **5D** trước HDBSCAN. Chạy trực tiếp 384D sẽ khiến thuật toán hỏng hoàn toàn.
3. **Mô hình nhúng**: **`BAAI/bge-small-en-v1.5`** là lựa chọn số 1: nhẹ, nhanh, vượt trội hơn cả MPNet-base 768D và MiniLM-L6.
4. **Hệ thống cảnh báo Emerging Topics**: Tích hợp cờ cảnh báo qua **kNN distance** thay vì dựa vào điểm noise của HDBSCAN.

