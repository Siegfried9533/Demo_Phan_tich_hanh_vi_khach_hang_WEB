# 🚀 Demo Seminar: AI-Driven Web Analytics (Phân cụm Khách hàng)

Đây là dự án demo cho seminar chủ đề "Ứng dụng AI cho Web Analytics". Mục tiêu của dự án là xây dựng một "Data App" (ứng dụng dữ liệu) bằng Streamlit để so sánh trực quan giữa:

1.  **Analytics Truyền thống:** Thống kê mô tả (Doanh thu, Top sản phẩm).
2.  **AI-Driven Analytics:** Tự động phân cụm khách hàng (sử dụng mô hình RFM & K-Means) để tìm ra các nhóm khách hàng (VIP, Sắp mất...) và đề xuất hành động.

---

## 🛠️ Công nghệ sử dụng

* **Ngôn ngữ:** Python 3.10+
* **Giao diện Dashboard:** Streamlit
* **Xử lý dữ liệu:** Pandas
* **Mô hình AI:** Scikit-learn (K-Means)
* **Trực quan hóa:** Plotly Express
* **Quản lý code:** Git & GitHub

---

## 🏃 Hướng dẫn Cài đặt & Chạy

1.  **Clone dự án:**
    ```bash
    git clone [URL-repository-cua-ban]
    cd [ten-thu-muc-du-an]
    ```

2.  **Tạo môi trường ảo** (Rất khuyến khích):
    ```bash
    python -m venv venv
    ```
    * Trên macOS/Linux: `source venv/bin/activate`
    * Trên Windows: `venv\Scripts\activate`

3.  **Cài đặt thư viện:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Chuẩn bị Dữ liệu:**
    * Tải file dữ liệu (`online_retail.csv`) từ Kaggle.
    * đặt nó vào thư mục `data/`. (File này đã được thêm vào `.gitignore` để không push dữ liệu thô lên repo).

5.  **Chạy ứng dụng:**
    ```bash
    streamlit run app.py
    ```

---

## 🗂️ Cấu trúc Thư mục
```
Demo/
├── data/
│   └── online_retail.csv        # Dữ liệu thô (KHÔNG commit)
├── notebooks/
│   └── model_dev.ipynb          # File thử nghiệm mô hình (Người 1)
├── app.py                       # Giao diện Streamlit (Người 2 & 3)
├── utils.py                     # Xử lý data & AI logic (Người 1)
├── requirements.txt             # Danh sách thư viện
└── README.md                    # Tài liệu hướng dẫn
```

---

## 👥 Phân công Vai trò & Nhiệm vụ

### 👤 Người 1: [Tên Người 1] - Data Engineer (Lõi Data & AI)
* **Nhánh Git:** `feature/data-logic`
* **Khu vực làm việc:** `notebooks/model_dev.ipynb` (để nháp) và `utils.py` (code sạch).
* **Nhiệm vụ:**
    1.  Làm sạch dữ liệu thô từ file `.csv`.
    2.  Xây dựng logic tính toán **RFM (Recency, Frequency, Monetary)**.
    3.  Xây dựng hàm huấn luyện mô hình **K-Means**.
    4.  "Đóng gói" tất cả logic trên thành các **hàm (functions)** sạch sẽ trong `utils.py` để Người 2 và 3 có thể gọi và sử dụng.

### 👤 Người 2: [Tên Người 2] - BI Analyst (Tab Truyền thống)
* **Nhánh Git:** `feature/traditional-tab`
* **Khu vực làm việc:** `app.py` (chỉ làm việc trong Tab 1).
* **Nhiệm vụ:**
    1.  Gọi hàm tải/làm sạch dữ liệu từ `utils.py`.
    2.  Dựng bố cục Tab 1: "📊 Phân tích Truyền thống".
    3.  Vẽ các biểu đồ mô tả (Doanh thu theo thời gian, Top sản phẩm bán chạy...).
    4.  Thêm các bình luận, diễn giải về hạn chế của phương pháp truyền thống.

### 👤 Người 3: [Tên Người 3] - AI Analyst (Tab AI & Insights)
* **Nhánh Git:** `feature/ai-tab`
* **Khu vực làm việc:** `app.py` (chỉ làm việc trong Tab 2).
* **Nhiệm vụ:**
    1.  Dựng bố cục Tab 2: "🤖 Phân tích AI".
    2.  Tạo **thanh trượt (Slider)** cho phép người dùng chọn số cụm (K).
    3.  Gọi các hàm RFM và K-Means từ `utils.py` (dựa trên giá trị K từ slider).
    4.  Vẽ **biểu đồ phân cụm** (dùng Plotly) để trực quan hóa các nhóm.
    5.  **QUAN TRỌNG:** Diễn giải ý nghĩa nghiệp vụ của từng cụm (ví dụ: Cụm 0 là "VIP", Cụm 1 là "Sắp mất"...).

---
