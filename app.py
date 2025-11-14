import streamlit as st
import pandas as pd
import plotly.express as px
from utils import (
    run_rfm_kmeans_pipeline,
    clean_retail_data,
    compute_rfm,
    scale_rfm,
    train_kmeans,
)

st.set_page_config(page_title="RFM Clustering Demo", layout="wide")

traditional_tab, ai_tab = st.tabs(["Phân tích Truyền thống", "Phân tích AI"])

def _name_and_actions_for_cluster(r: float, f: float, m: float, r_med: float, f_med: float, m_med: float):
    """
    Gán tên cụm + đề xuất hành động dựa trên trung vị toàn cục (median).
    Quy ước:
      - R thấp = mua gần đây (tốt), R cao = đã lâu không mua (xấu)
      - F cao = mua thường xuyên
      - M cao = chi tiêu nhiều
    """
    r_is_low = r <= r_med
    f_is_high = f >= f_med
    m_is_high = m >= m_med

    if r_is_low and f_is_high and m_is_high:
        return (
            "Khách hàng VIP",
            [
                "Duy trì ưu đãi độc quyền và chăm sóc cá nhân hoá.",
                "Chương trình tích điểm/tier cao, early-access sản phẩm mới.",
            ],
        )
    if (not r_is_low) and (not f_is_high) and (not m_is_high):
        return (
            "Khách hàng Sắp mất",
            [
                "Kích hoạt lại bằng voucher mạnh/ưu đãi quay lại.",
                "Gửi email/SMS nhắc nhớ + ưu đãi thời hạn ngắn.",
            ],
        )
    if (not r_is_low) and (f_is_high or m_is_high):
        return (
            "Ngủ đông giá trị",
            [
                "Chiến dịch win-back nhắm đến nhóm có giá trị từng cao.",
                "Ưu đãi cá nhân hoá theo lịch sử sản phẩm đã mua.",
            ],
        )
    if r_is_low and (f_is_high or m_is_high):
        return (
            "Khách hàng Tiềm năng/Vừa quay lại",
            [
                "Upsell/cross-sell theo danh mục liên quan.",
                "Tăng tần suất bằng combo/bundle hoặc free shipping.",
            ],
        )
    if r_is_low and (not f_is_high) and (not m_is_high):
        return (
            "Khách hàng Mới",
            [
                "Onboarding: hướng dẫn, đề xuất danh mục, ưu đãi mua tiếp.",
                "Khuyến khích lần mua 2 bằng mã giảm nhẹ.",
            ],
        )
    return (
        "Nhóm Cần kích hoạt",
        [
            "A/B test thông điệp và ưu đãi khác nhau.",
            "Remarketing đa kênh (email/Zalo/advertising).",
        ],
    )

with ai_tab:
    st.header("Phân tích AI: Phân cụm khách hàng theo RFM")
    st.caption("Tải dữ liệu, chọn số cụm K, xem phân tán theo cụm, bảng trung bình R-F-M và diễn giải.")

    col_left, col_right = st.columns([2, 1])
    with col_left:
        uploaded_file = st.file_uploader("Tải file CSV (Online Retail II)", type=["csv"])
        csv_path = st.text_input("Hoặc nhập đường dẫn tới CSV trên máy", value="", placeholder="VD: D:\\data\\online_retail_II.csv")
    with col_right:
        k = st.slider("Chọn số cụm (K)", min_value=2, max_value=6, value=4, step=1)

    if uploaded_file is None and not csv_path:
        st.warning("Vui lòng tải CSV hoặc nhập đường dẫn để bắt đầu phân tích.")
        st.stop()

    try:
        if uploaded_file is not None:
            df_raw = pd.read_csv(uploaded_file)
            df_clean = clean_retail_data(df_raw)
            rfm, snapshot_date = compute_rfm(df_clean)
            X_scaled, scaler = scale_rfm(rfm)
            kmeans_model = train_kmeans(X_scaled, n_clusters=k)
            rfm = rfm.copy()
            rfm["Cluster"] = kmeans_model.labels_
        else:
            result = run_rfm_kmeans_pipeline(csv_path, n_clusters=k)
            rfm = result["rfm"]

        st.subheader("Biểu đồ phân tán theo cụm")
        fig = px.scatter(
            rfm,
            x="Frequency",
            y="Monetary",
            color="Cluster",
            color_discrete_sequence=px.colors.qualitative.Set1,
            hover_data=["Customer ID", "Recency", "Frequency", "Monetary"],
            title="Phân tán khách hàng theo cụm (trục: Frequency vs Monetary)"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("R-F-M trung bình theo cụm")
        cluster_means = (
            rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
            .mean()
            .round(2)
            .sort_index()
        )
        st.dataframe(cluster_means, use_container_width=True)

        r_med = rfm["Recency"].median()
        f_med = rfm["Frequency"].median()
        m_med = rfm["Monetary"].median()

        st.subheader("Diễn giải & Đề xuất hành động")
        for cluster_id, row in cluster_means.iterrows():
            name, actions = _name_and_actions_for_cluster(
                r=row["Recency"], f=row["Frequency"], m=row["Monetary"],
                r_med=r_med, f_med=f_med, m_med=m_med
            )
            with st.expander(f"Cụm {cluster_id}: {name}"):
                cols = st.columns(3)
                cols[0].metric("Recency (↓ tốt)", value=row["Recency"])
                cols[1].metric("Frequency (↑ tốt)", value=row["Frequency"])
                cols[2].metric("Monetary (↑ tốt)", value=row["Monetary"])

                st.markdown("**Đề xuất hành động:**")
                for act in actions:
                    st.write(f"- {act}")

    except Exception as e:
        st.error(f"Lỗi khi xử lý dữ liệu: {e}")
