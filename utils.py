# utils.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# =========================
# 1. Load & Clean dữ liệu
# =========================

def load_raw_data(path: str) -> pd.DataFrame:
    """
    Đọc file CSV Online Retail II từ Kaggle.
    """
    df = pd.read_csv(path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def clean_retail_data(
    df: pd.DataFrame,
    drop_missing_customer: bool = True,
    filter_positive_quantity: bool = True,
    filter_positive_price: bool = True,
    drop_duplicates: bool = True
) -> pd.DataFrame:
    """
    Làm sạch dữ liệu cho bài toán RFM + K-Means.
    - Bỏ Customer ID bị thiếu.
    - Chỉ giữ Quantity > 0 (loại đơn trả hàng/hủy).
    - (Tuỳ chọn) Chỉ giữ Price > 0.
    - Xoá trùng hoàn toàn (nếu có).
    - Tạo TotalPrice = Quantity * Price.
    """
    df_clean = df.copy()

    if not np.issubdtype(df_clean["InvoiceDate"].dtype, np.datetime64):
        df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])

    if drop_missing_customer:
        df_clean = df_clean[df_clean["Customer ID"].notna()].copy()

    if filter_positive_quantity:
        df_clean = df_clean[df_clean["Quantity"] > 0].copy()

    if filter_positive_price:
        df_clean = df_clean[df_clean["Price"] > 0].copy()

    if drop_duplicates:
        df_clean = df_clean.drop_duplicates().copy()

    df_clean["Customer ID"] = df_clean["Customer ID"].astype(int)
    df_clean["TotalPrice"] = df_clean["Quantity"] * df_clean["Price"]

    return df_clean


# =========================
# 2. RFM
# =========================

def compute_rfm(df_clean: pd.DataFrame, snapshot_date: pd.Timestamp | None = None):
    """
    Tính Recency, Frequency, Monetary cho mỗi khách hàng.
    """
    if snapshot_date is None:
        snapshot_date = df_clean["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df_clean
        .groupby("Customer ID")
        .agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("Invoice", "nunique"),  # đếm số hóa đơn, không đếm số dòng
            Monetary=("TotalPrice", "sum"),
        )
        .reset_index()
    )

    return rfm, snapshot_date


# =========================
# 3. Chuẩn hóa & K-Means
# =========================

def scale_rfm(rfm: pd.DataFrame, features=("Recency", "Frequency", "Monetary")):
    """
    Chuẩn hóa các cột R, F, M bằng StandardScaler.
    """
    scaler = StandardScaler()
    X = rfm[list(features)].values
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def train_kmeans(X_scaled, n_clusters: int = 4, random_state: int = 42):
    """
    Huấn luyện K-Means trên dữ liệu RFM đã chuẩn hóa.
    """
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    model.fit(X_scaled)
    return model


# =========================
# 4. Pipeline tiện dụng
# =========================

def run_rfm_kmeans_pipeline(
    path_to_csv: str,
    n_clusters: int = 4,
):
    """
    Chạy full pipeline:
        - load_raw_data
        - clean_retail_data
        - compute_rfm
        - scale_rfm
        - train_kmeans
    Trả về dict chứa các thành phần để Người 2 & 3 có thể dùng tiếp.
    """
    df_raw = load_raw_data(path_to_csv)
    df_clean = clean_retail_data(df_raw)

    rfm, snapshot_date = compute_rfm(df_clean)
    X_scaled, scaler = scale_rfm(rfm)
    kmeans_model = train_kmeans(X_scaled, n_clusters=n_clusters)

    # Gán nhãn cụm vào RFM
    rfm = rfm.copy()
    rfm["Cluster"] = kmeans_model.labels_

    return {
        "df_raw": df_raw,
        "df_clean": df_clean,
        "rfm": rfm,
        "snapshot_date": snapshot_date,
        "scaler": scaler,
        "kmeans_model": kmeans_model,
    }

    """
    Khi chạy chỉ cần chạy đoạn sau:
    
    from utils import run_rfm_kmeans_pipeline

    result = run_rfm_kmeans_pipeline("data/online_retail_II.csv", n_clusters=4)  

    rfm = result["rfm"]   # có cột Cluster
    """