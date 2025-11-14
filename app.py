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

with traditional_tab:
    st.header("Phân tích truyền thống")
    st.caption("Tải dữ liệu")
    # file uploader and path input for the traditional tab (unique keys)
    uploaded_file_tradition = st.file_uploader(
        "Tải file CSV (Online Retail II)", type=["csv"], key="file_uploader_tradition"
    )
    csv_path_tradition = st.text_input(
        "Hoặc nhập đường dẫn tới CSV trên máy",
        value="",
        placeholder="VD: D:\\data\\online_retail_II.csv",
        key="csv_path_tradition",
    )

    if uploaded_file_tradition is None and not csv_path_tradition:
        st.warning("Vui lòng tải CSV hoặc nhập đường dẫn để bắt đầu phân tích.")
    try:
        if uploaded_file_tradition is not None:
            df_raw = pd.read_csv(uploaded_file_tradition)
        else:
            df_raw = pd.read_csv(csv_path_tradition)


        # Clean the raw data to remove returns / negative quantities before aggregations
        df_clean = clean_retail_data(df_raw)

        st.subheader("Doanh thu")
        df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
        # snapshot_date used to compute recency (use one day after last invoice in data)
        snapshot_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
        df_clean['Month'] = df_clean['InvoiceDate'].dt.strftime('%Y-%m')
        # TotalPrice already created in clean_retail_data; use cleaned frame for revenue
        revenue_per_month = (
            df_clean.groupby('Month')['TotalPrice'].sum().reset_index()
        )
        fig_revenue = px.line(
            revenue_per_month,
            x='Month',
            y='TotalPrice',
            title='Doanh thu theo tháng (không tính đơn trả/hủy)',
            labels={'TotalPrice': 'Doanh thu', 'Month': 'Tháng'},
            template='plotly_white',
        )
        fig_revenue.update_traces(line_color='#1f77b4')
        st.plotly_chart(fig_revenue, use_container_width=True)

        st.subheader("Sản phẩm bán chạy")
        top_products = (
            df_clean.groupby('Description')['Quantity'].sum().reset_index()
            .sort_values(by='Quantity', ascending=False)
            .head(5)
        )
        fig_top_products = px.bar(
            top_products,
            x='Description',
            y='Quantity',
            title='Top 5 sản phẩm bán chạy (không tính đơn trả/hủy)',
            labels={'Description': 'Sản phẩm', 'Quantity': 'Số lượng bán'},
            template='plotly_white',
            color='Description',
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_top_products.update_layout(showlegend=False)
        st.plotly_chart(fig_top_products, use_container_width=True)

        st.subheader("Khách hàng")

        # Nhóm theo Customer ID và tính R, F, M (dùng dữ liệu đã làm sạch để loại đơn trả)
        rfm_df = df_clean.groupby('Customer ID').agg(
            R_Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
            F_Frequency=('Invoice', 'nunique'),
            M_Monetary=('TotalPrice', 'sum')
        ).reset_index() # Chuyển customer_id từ index thành cột
        
        # Filter controls for RFM
        col1, col2 = st.columns(2)
        with col1:
            min_monetary = st.number_input(
                "Monetary ≥ (Tối thiểu)",
                value=0.0,
                step=10.0,
                key="min_monetary_filter"
            )
        with col2:
            max_recency = st.number_input(
                "Recency ≤ (Tối đa ngày)",
                value=999,
                step=10,
                key="max_recency_filter"
            )
            
        run_filter = st.button("Lọc", key="run_rfm_filter")
        
        # Apply filters only when button is clicked
        if run_filter:
            filtered_rfm = rfm_df[
                (rfm_df['M_Monetary'] >= min_monetary) & 
                (rfm_df['R_Recency'] <= max_recency)
            ]
            
            st.caption(f"Hiển thị {len(filtered_rfm)} / {len(rfm_df)} khách hàng")
            st.dataframe(filtered_rfm, use_container_width=True)
        else:
            st.caption(f"Nhấn nút 'Lọc' để tìm kiếm khách hàng")
            st.dataframe(rfm_df, use_container_width=True)

    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu: {e}")


with ai_tab:
    st.header("Phân tích AI: Phân cụm khách hàng theo RFM")
    st.caption("Tải dữ liệu, chọn số cụm K, xem phân tán theo cụm, bảng trung bình R-F-M và diễn giải.")

    col_left, col_right = st.columns([2, 1])
    with col_left:
        # file uploader and path input for the AI tab (unique keys)
        uploaded_file_ai = st.file_uploader(
            "Tải file CSV (Online Retail II)", type=["csv"], key="file_uploader_ai"
        )
        csv_path_ai = st.text_input(
            "Hoặc nhập đường dẫn tới CSV trên máy",
            value="",
            placeholder="VD: D:\\data\\online_retail_II.csv",
            key="csv_path_ai",
        )
    with col_right:
        k = st.slider("Chọn số cụm (K)", min_value=2, max_value=6, value=4, step=1)

    if uploaded_file_ai is None and not csv_path_ai:
        st.warning("Vui lòng tải CSV hoặc nhập đường dẫn để bắt đầu phân tích.")
        st.stop()

    try:
        if uploaded_file_ai is not None:
            df_raw = pd.read_csv(uploaded_file_ai)
            df_clean = clean_retail_data(df_raw)
            rfm, snapshot_date = compute_rfm(df_clean)
            X_scaled, scaler = scale_rfm(rfm)
            kmeans_model = train_kmeans(X_scaled, n_clusters=k)
            rfm = rfm.copy()
            rfm["Cluster"] = kmeans_model.labels_
        else:
            result = run_rfm_kmeans_pipeline(csv_path_ai, n_clusters=k)
            rfm = result["rfm"]

        st.subheader("Biểu đồ phân tán theo cụm")
        # Three scatter plots: R-F, R-M, F-M
        col_a, col_b = st.columns(2)
        with col_a:
            fig_rf = px.scatter(
                rfm,
                x="Recency",
                y="Frequency",
                color="Cluster",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hover_data=["Customer ID", "Recency", "Frequency", "Monetary"],
                title="Recency vs Frequency",
                template='plotly_white',
            )
            st.plotly_chart(fig_rf, use_container_width=True)
        with col_b:
            fig_rm = px.scatter(
                rfm,
                x="Recency",
                y="Monetary",
                color="Cluster",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hover_data=["Customer ID", "Recency", "Frequency", "Monetary"],
                title="Recency vs Monetary",
                template='plotly_white',
            )
            st.plotly_chart(fig_rm, use_container_width=True)

        fig_fm = px.scatter(
            rfm,
            x="Frequency",
            y="Monetary",
            color="Cluster",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hover_data=["Customer ID", "Recency", "Frequency", "Monetary"],
            title="Frequency vs Monetary",
            template='plotly_white',
        )
        st.plotly_chart(fig_fm, use_container_width=True)

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

        # Quantiles for more differentiated actions
        r_q = {"low": rfm["Recency"].quantile(0.33), "high": rfm["Recency"].quantile(0.66)}
        f_q = {"low": rfm["Frequency"].quantile(0.33), "high": rfm["Frequency"].quantile(0.66)}
        m_q = {"low": rfm["Monetary"].quantile(0.33), "high": rfm["Monetary"].quantile(0.66)}

        def suggest_actions(r: float, f: float, m: float):
            actions: list[str] = []
            # Recency: lower is better
            if r <= r_q["low"]:
                actions.append("Upsell/cross-sell theo lịch sử sản phẩm vừa mua.")
            elif r >= r_q["high"]:
                actions.append("Kích hoạt lại bằng email/SMS kèm ưu đãi quay lại thời hạn ngắn.")
            else:
                actions.append("Nhắc nhớ nhẹ nhàng và gợi ý sản phẩm liên quan.")
            # Frequency: higher is better
            if f >= f_q["high"]:
                actions.append("Tăng quyền lợi loyalty/tier; chương trình dành riêng.")
            elif f <= f_q["low"]:
                actions.append("Khuyến khích mua lại bằng voucher nhỏ hoặc freeship ngưỡng thấp.")
            else:
                actions.append("Bundle/combo để tăng tần suất.")
            # Monetary: higher is better
            if m >= m_q["high"]:
                actions.append("Đề xuất sản phẩm cao cấp/độc quyền; chăm sóc ưu tiên.")
            elif m <= m_q["low"]:
                actions.append("Đề xuất sản phẩm giá hợp lý; tối ưu chi phí vận chuyển.")
            else:
                actions.append("Gợi ý nâng giá trị giỏ bằng phụ kiện/phụ trợ.")
            return actions

        # Rank clusters into customer tiers (1 = best)
        r_min, r_max = cluster_means['Recency'].min(), cluster_means['Recency'].max()
        f_min, f_max = cluster_means['Frequency'].min(), cluster_means['Frequency'].max()
        m_min, m_max = cluster_means['Monetary'].min(), cluster_means['Monetary'].max()
        def norm(val: float, vmin: float, vmax: float) -> float:
            return 0.0 if vmax == vmin else (val - vmin) / (vmax - vmin)

        rank_df = []
        for cid, row in cluster_means.iterrows():
            score = (1 - norm(row['Recency'], r_min, r_max)) + \
                    norm(row['Frequency'], f_min, f_max) + \
                    norm(row['Monetary'], m_min, m_max)
            rank_df.append({'Cluster': cid, 'Score': score})
        rank_df = pd.DataFrame(rank_df).sort_values('Score', ascending=False).reset_index(drop=True)
        rank_df['Rank'] = rank_df.index + 1
        cluster_to_rank = dict(zip(rank_df['Cluster'], rank_df['Rank']))

        # Build tiered names and actions per cluster
        cluster_info = []
        for cid, row in cluster_means.iterrows():
            rank = cluster_to_rank[cid]
            name = f"Khách hạng {rank}"
            actions = suggest_actions(row['Recency'], row['Frequency'], row['Monetary'])
            cluster_info.append({"Cluster": cid, "Hạng": rank, "Tên nhóm": name, "Actions": actions})

        st.caption(f"Có {k} loại khách hàng: từ Khách hạng 1 đến {k}.")
        st.dataframe(
            pd.DataFrame([{ "Hạng": c["Hạng"], "Cluster": c["Cluster"] } for c in cluster_info]).sort_values("Hạng"),
            use_container_width=True,
        )

        st.subheader("Diễn giải & Đề xuất hành động")
        for info in sorted(cluster_info, key=lambda x: x['Hạng']):
            rank = info["Hạng"]
            cid = info["Cluster"]
            actions = info["Actions"]
            row = cluster_means.loc[cid]
            with st.expander(f"Hạng {rank} (Cluster {cid})"):
                cols = st.columns(3)
                cols[0].metric("Recency (↓ tốt)", value=row["Recency"])
                cols[1].metric("Frequency (↑ tốt)", value=row["Frequency"])
                cols[2].metric("Monetary (↑ tốt)", value=row["Monetary"])

                st.markdown("**Đề xuất hành động:**")
                for act in actions:
                    st.write(f"- {act}")

    except Exception as e:
        st.error(f"Lỗi khi xử lý dữ liệu: {e}")
