# 📊 PHÂN KHÚC KHÁCH HÀNG TỪ DỮ LIỆU GIAO DỊCH QUY MÔ LỚN BẰNG APACHE SPARK

Dự án nghiên cứu và xây dựng ứng dụng phân khúc người dùng dựa trên dữ liệu mua hàng Amazon (Amazon Purchases). Ứng dụng kết hợp sức mạnh xử lý dữ liệu lớn của **Apache Spark (PySpark)**, thuật toán phân cụm **K-Means**, và giao diện tương tác trực quan bằng **Streamlit**.

---

## 🚀 Demo Ứng Dụng (Live Streamlit App)
👉 **Truy cập ứng dụng tại: https://rfm-clustering-app-9mmlyzggermygpdyqnewge.streamlit.app/insights_han_che

---

## 📌 Tính Năng Chính & Nội Dung Dự Án

- **Xử lý & Khám phá Dữ liệu lớn (Apache Spark):**
  - Kiểm tra chất lượng dữ liệu, xử lý giá trị khuyết thiếu và dữ liệu ngoại lệ (outliers).
  - Trích xuất các chỉ số **RFM** (Recency, Frequency, Monetary) cùng các đặc trưng hành vi người dùng (như số lượng đơn vị mua trung bình `Avg_Quantity`).
- **Mô Hình Phân Cụm (Machine Learning):**
  - Thuật toán **K-Means** chia nhóm khách hàng.
  - So sánh mô hình RFM truyền thống (Model A) và RFM kết hợp đặc trưng hành vi (Model B).
  - Đánh giá chất lượng phân cụm bằng chỉ số **Silhouette Score** và **ARI (Adjusted Rand Index)**.
- **Giao Diện Trực Quan (Streamlit Dashboard):**
  - **Tổng quan dự án:** Báo cáo tổng thể và cấu trúc bộ dữ liệu.
  - **Khám phá phân khúc:** Biểu đồ tương tác mô tả đặc trưng từng nhóm khách hàng.
  - **Tra cứu người dùng:** Tìm kiếm phân nhóm chi tiết theo từng `Survey_ResponseID`.
  - **Insights & Hạn chế:** Đưa ra khuyến nghị kinh doanh và các điểm cần lưu ý của mô hình.

---

## 📂 Cấu Trúc Repository

```text
.
├── 01_spark_data_audit.ipynb   # Notebook xử lý dữ liệu, trích xuất đặc trưng & huấn luyện Spark K-Means
├── app.py                      # File khởi tạo chính cho ứng dụng Streamlit
├── pages/                      # Các trang giao diện chi tiết của ứng dụng
│   ├── 1_Tổng_quan.py
│   ├── 2_Kỹ_thuật_Spark.py
│   └── ...
├── utils.py                    # Các hàm phụ trợ (helper functions)
├── requirements.txt            # Danh sách thư viện Python cần thiết
├── Dulieulon/                  # Thư mục lưu trữ dữ liệu kết quả phân tích
└── assets/                     # Hình ảnh và tài nguyên giao diện
