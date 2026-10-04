import textwrap

import plotly.graph_objects as go
import streamlit as st

from utils import (CLUSTER_COLORS, FIXED, callout, chart_title, download_csv, has_cols, hero, kpi,
                   read_csv, show, source, static_figures, table, vn_int, vn_pct, warn_missing)

hero(
    "Phân khúc người dùng từ dữ liệu giao dịch quy mô lớn bằng Apache Spark",
    "So sánh RFM với RFM cộng đặc trưng hành vi bằng K-Means, trên dữ liệu Amazon Purchases. "
    "Ứng dụng chỉ đọc kết quả đã lưu từ pipeline Spark, không chạy lại mô hình.",
)

steps = read_csv("row_filter")
users = read_csv("user_features")

raw_rows, clean_rows, n_users = FIXED["raw_rows"], FIXED["clean_rows"], FIXED["users"]
if has_cols(steps, ["step", "rows"]) and len(steps) >= 2:
    raw_rows, clean_rows = int(steps["rows"].iloc[0]), int(steps["rows"].iloc[-1])
if users is not None:
    n_users = len(users)

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi("Dòng giao dịch thô", vn_int(raw_rows), "Tệp Amazon Purchases ban đầu")
with c2:
    kpi("Dòng sau làm sạch", vn_int(clean_rows), f"Loại {vn_int(raw_rows - clean_rows)} dòng qua các bước lọc")
with c3:
    kpi("Người dùng", vn_int(n_users), f"{FIXED['n_features']} đặc trưng cho mỗi người dùng")
with c4:
    kpi("Số cụm K", str(FIXED["k"]), f"Tốt nhất trong {FIXED['n_seeds']} seed (WSSSE)")

st.markdown("### Pipeline xử lý")
st.markdown(
    '<div class="pipe">'
    '<div class="step"><div class="t">Spark</div><div class="s">Đọc CSV, kiểm tra chất lượng, lọc ID và cửa sổ thời gian</div></div>'
    '<div class="step"><div class="t">Parquet</div><div class="s">Lưu bảng giao dịch sạch <code>clean_transactions</code></div></div>'
    '<div class="step"><div class="t">groupBy</div><div class="s">Gộp theo Survey_ResponseID thành 7 đặc trưng</div></div>'
    '<div class="step"><div class="t">MLlib K-Means</div><div class="s">Log, winsorize, chuẩn hóa rồi phân cụm với K = 4</div></div>'
    '<div class="step"><div class="t">Validation</div><div class="s">Silhouette, DB, CH, stability, robustness</div></div>'
    '<div class="step"><div class="t">Profiling</div><div class="s">Hồ sơ cụm, đặt tên cụm, so sánh nhóm hàng</div></div>'
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("### Từ dữ liệu thô đến người dùng")
left, right = st.columns([3, 2], gap="large")

with left:
    if warn_missing("row_filter") or not has_cols(steps, ["step", "rows"]):
        labels = ["Dòng thô", "ID hợp lệ", "Trong cửa sổ 2018-2022"]
        values = [FIXED["raw_rows"], FIXED["valid_id_rows"], FIXED["clean_rows"]]
    else:
        labels, values = steps["step"].astype(str).tolist(), steps["rows"].astype(int).tolist()
    removed_pct = (1 - values[-1] / values[0]) * 100
    chart_title(
        f"Sau lọc còn {vn_pct(100 - removed_pct)} số dòng thô",
        "Mỗi thanh là số dòng còn lại sau từng bước làm sạch",
    )
    fig = go.Figure(go.Funnel(
        y=["<br>".join(textwrap.wrap(l, 24)) for l in labels], x=values, textposition="inside",
        texttemplate="%{value:,.0f}<br>%{percentInitial:.1%}",
        marker=dict(color=[CLUSTER_COLORS[0], CLUSTER_COLORS[1], CLUSTER_COLORS[2], CLUSTER_COLORS[3]][: len(values)]),
        connector=dict(line=dict(color="rgba(128,128,128,0.4)", width=1)),
        hovertemplate="%{y}<br>%{x:,.0f} dòng<extra></extra>",
    ))
    fig.update_yaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    show(fig, key="funnel", height=360, legend=False)
    source()

with right:
    callout(
        "insight",
        "Dữ liệu thô có <b>39.479</b> dòng mang Survey_ResponseID sai do Spark đọc CSV với cấu hình escape mặc định. "
        "Các dòng này bị loại ở bước kiểm tra ID dạng <code>R_xxx</code>. Xem chi tiết ở trang Kỹ thuật Spark.",
        title="Điều đáng chú ý",
    )
    callout(
        "info",
        "<b>Survey_ResponseID</b> là người dùng ở cấp dữ liệu nghiên cứu, không phải khách hàng Amazon nói chung. "
        "<b>Frequency</b> là số ngày có phát sinh mua hàng vì dữ liệu không có mã đơn.",
        title="Cách gọi trong ứng dụng",
    )
    if has_cols(steps, ["step", "rows"]):
        with st.expander("Chi tiết từng bước lọc", icon=":material/table:"):
            t = steps.copy()
            table(t)
            download_csv(t, "cac_buoc_loc_du_lieu.csv", key="dl_steps")

static_figures(["fig01"], key="fig_overview")
