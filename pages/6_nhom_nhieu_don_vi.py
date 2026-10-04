import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (CLUSTER_COLORS, CLUSTER_NAMES_B, FIXED, callout, chart_title, download_csv, empty_state,
                   has_cols, how_to_read, kpi, load_clusters, page_header, read_csv, show, source,
                   static_figures, table, vn_int, vn_num, vn_pct, warn_missing)

NICHE_ID = 2
NICHE_NAME = CLUSTER_NAMES_B[NICHE_ID]
MIN_LINES = 300

page_header(
    f"Nhóm “{NICHE_NAME}”",
    "Nhóm nhỏ nhất của Model B, nổi bật vì số lượng trung bình mỗi dòng cao hơn hẳn. Trang này xem nhóm đó mua thiên về nhóm hàng nào.",
)

# ---------------------------------------------------------------- Cảnh báo n nhỏ
callout(
    "warning",
    f"Nhóm chỉ có <b>{vn_int(FIXED['niche_users'])}</b> người dùng. Các chỉ số lift ở đây là <b>mô tả</b>, "
    "chưa qua kiểm định thống kê và không cho phép suy ra nguyên nhân. Đừng diễn giải từng nhóm hàng riêng lẻ như một kết luận.",
    title="Cỡ mẫu nhỏ, không có kiểm định",
)

dfb = load_clusters("B_RFM_Behavior")
if dfb is not None and "Avg_Quantity" in dfb.columns:
    g = dfb.groupby("cluster")["Avg_Quantity"].median()
    n_niche = int((dfb["cluster"] == NICHE_ID).sum())
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("Người dùng trong nhóm", vn_int(n_niche), f"{vn_pct(n_niche / len(dfb) * 100)} tổng số người dùng")
    with c2:
        kpi("Số lượng TB mỗi dòng (trung vị)", vn_num(float(g.get(NICHE_ID, float("nan"))), 2), "Trung vị của nhóm này")
    with c3:
        others = g.drop(index=NICHE_ID, errors="ignore")
        kpi("Cao nhất ở các cụm còn lại", vn_num(float(others.max()), 2) if len(others) else "–", "Trung vị Avg_Quantity")

# ---------------------------------------------------------------- Lift
nv = read_csv("niche")
need = ["lines_niche", "share_units_niche", "share_units_rest", "lift_units"]
if warn_missing("niche"):
    st.stop()
if not has_cols(nv, need):
    st.warning("Tệp category_niche_vs_rest.csv thiếu cột: " + ", ".join(c for c in need if c not in nv.columns))
    st.stop()

cat_col = "Category" if "Category" in nv.columns else nv.columns[0]
t = nv.copy()
t[cat_col] = t[cat_col].astype(str)
t = t[t["lines_niche"] >= MIN_LINES].replace([np.inf, -np.inf], np.nan).dropna(subset=["lift_units"])

st.markdown("### Nhóm hàng được ưu tiên hơn so với các cụm còn lại")
if t.empty:
    empty_state("Chưa có nhóm hàng nào đủ dữ liệu", f"Cần ít nhất {MIN_LINES} dòng giao dịch của nhóm trong một nhóm hàng.")
    st.stop()

if len(t) > 5:
    top_n = st.slider("Số nhóm hàng hiển thị", 5, min(30, len(t)), min(12, len(t)), key="niche_topn")
else:
    top_n = len(t)
top = t.sort_values("lift_units", ascending=False).head(top_n).sort_values("lift_units", ascending=True)
best = top.iloc[-1]
chart_title(
    f"{best[cat_col].replace('_', ' ').title()} có lift cao nhất: {vn_num(best['lift_units'], 1)} lần",
    f"Lift = tỷ trọng đơn vị của nhóm / tỷ trọng đơn vị của các cụm còn lại. Chỉ xét nhóm hàng có từ {MIN_LINES} dòng trở lên trong nhóm",
)
fig = go.Figure(go.Bar(
    y=[c.replace("_", " ").title() for c in top[cat_col]], x=top["lift_units"], orientation="h",
    marker_color=CLUSTER_COLORS[NICHE_ID],
    text=[vn_num(v, 1) for v in top["lift_units"]], textposition="outside",
    customdata=top[["lines_niche", "share_units_niche", "share_units_rest"]].values,
    hovertemplate=("%{y}<br>Lift %{x:,.2f} lần<br>Dòng của nhóm: %{customdata[0]:,.0f}"
                   "<br>Tỷ trọng đơn vị trong nhóm: %{customdata[1]:,.2f}%"
                   "<br>Tỷ trọng ở cụm còn lại: %{customdata[2]:,.2f}%<extra></extra>"),
))
fig.add_vline(x=1, line_dash="dash", line_color="rgba(128,128,128,0.9)", annotation_text="Bằng nhau (1,0)",
              annotation_position="bottom right")
fig.update_xaxes(title="Lift theo số đơn vị (lần)", rangemode="tozero")
fig.update_yaxes(title="Nhóm hàng (Category)")
show(fig, key="niche_lift", height=max(360, 28 * top_n + 120), legend=False)
source("Lift theo số đơn vị, không phải theo số dòng.")
how_to_read("Lift bằng 2 nghĩa là nhóm này mua nhóm hàng đó với tỷ trọng đơn vị gấp đôi so với các cụm còn lại. Lift lớn có thể do vài người mua rất nhiều, nên cần đọc cùng số dòng trong tooltip.")

with st.expander("Bảng số liệu đầy đủ", icon=":material/table:"):
    show_t = t.sort_values("lift_units", ascending=False)[[cat_col, "lines_niche", "share_units_niche", "share_units_rest", "lift_units"]]
    show_t = show_t.rename(columns={cat_col: "Nhóm hàng", "lines_niche": "Số dòng của nhóm",
                                    "share_units_niche": "Tỷ trọng đơn vị trong nhóm (%)",
                                    "share_units_rest": "Tỷ trọng đơn vị ở cụm còn lại (%)", "lift_units": "Lift (lần)"})
    table(show_t, column_config={
        "Số dòng của nhóm": st.column_config.NumberColumn(format="%d"),
        "Tỷ trọng đơn vị trong nhóm (%)": st.column_config.NumberColumn(format="%.2f"),
        "Tỷ trọng đơn vị ở cụm còn lại (%)": st.column_config.NumberColumn(format="%.2f"),
        "Lift (lần)": st.column_config.NumberColumn(format="%.2f"),
    })
    download_csv(show_t, "lift_nhom_nhieu_don_vi.csv", key="dl_niche")

static_figures(["fig14"], key="fig_niche")
