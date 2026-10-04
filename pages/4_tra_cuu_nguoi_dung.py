import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (FEATURE_SHORT, FEATURE_VI, MODELS, callout, chart_title, cluster_color, cluster_label,
                   download_csv, empty_state, features_present, how_to_read, load_clusters, page_header, pct_rank,
                   pick, show, source, table, vn_int, vn_num, vn_pct, warn_missing)

page_header(
    "Tra cứu người dùng",
    "Nhập hoặc chọn một Survey_ResponseID để xem cụm và vị trí tương đối của người dùng đó so với toàn bộ và so với cụm.",
)

model = pick("Mô hình", MODELS, key="lookup_model", default="B_RFM_Behavior")
df = load_clusters(model)
if warn_missing(f"clusters_{model}") or df is None:
    st.stop()

feats = features_present(df)
ids = df["Survey_ResponseID"].astype(str).sort_values().tolist()

# ---------------------------------------------------------------- Chọn người dùng
c_in, c_sel = st.columns(2, gap="large")
with c_in:
    typed = st.text_input("Nhập Survey_ResponseID", placeholder="Ví dụ: R_3e8qukcDiaT0D0g", key="lookup_typed").strip()
with c_sel:
    chosen = st.selectbox("Hoặc chọn từ danh sách", ids, index=None, placeholder="Gõ để tìm ID", key="lookup_select")

uid = None
if typed:
    if typed in set(ids):
        uid = typed
    else:
        near = [i for i in ids if typed.lower() in i.lower()][:5]
        st.warning("Không có ID này trong dữ liệu." + (f" ID gần giống: {', '.join(near)}" if near else ""),
                   icon=":material/search_off:")
elif chosen:
    uid = chosen

if uid is None:
    empty_state("Chưa chọn người dùng", "Nhập một ID ở ô bên trái hoặc chọn từ danh sách để xem hồ sơ.")
else:
    row = df[df["Survey_ResponseID"].astype(str) == uid].iloc[0]
    cl = int(row["cluster"])
    same = df[df["cluster"] == cl]
    color = cluster_color(cl)

    # ------------------------------------------------------------ Thẻ người dùng
    st.markdown(
        f'<div class="seg-card"><div class="person"><span class="dot" style="background:{color}"></span>'
        f'<span class="id">{uid}</span><span class="chip" style="border-color:{color}">{cluster_label(model, cl)}</span>'
        f'<span class="kpi-note">Cụm này có {vn_int(len(same))} người dùng ({vn_pct(len(same) / len(df) * 100)})</span>'
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.write("")

    cols = st.columns(min(len(feats), 7))
    for col, f in zip(cols, feats):
        with col:
            v = float(row[f])
            st.metric(FEATURE_SHORT[f], vn_num(v, 2 if f == "Avg_Quantity" else 0 if v >= 100 else 1))

    p_all = {f: pct_rank(df[f], float(row[f])) for f in feats}
    p_cl = {f: pct_rank(same[f], float(row[f])) for f in feats}

    st.markdown("### Vị trí so với toàn bộ và so với cụm")
    view = pick("Dạng hiển thị", {"radar": "Radar", "bullet": "Thanh percentile"}, key="lookup_view", default="radar")
    hi_f = max(feats, key=lambda f: p_all[f])
    chart_title(
        f"Cao nhất ở {FEATURE_SHORT[hi_f]}: vượt {vn_num(p_all[hi_f], 0)}% người dùng, {vn_num(p_cl[hi_f], 0)}% trong cụm",
        "Percentile của giá trị gốc. Với Recency, percentile cao nghĩa là lần mua cuối cách đây lâu",
    )
    names = [FEATURE_SHORT[f] for f in feats]
    if view == "radar":
        fig = go.Figure()
        for lab, p, col, dash in (("So với toàn bộ", p_all, "#2F7BEA", "solid"), ("So với cụm", p_cl, color, "dot")):
            vals = [p[f] for f in feats]
            fig.add_trace(go.Scatterpolar(
                r=vals + vals[:1], theta=names + names[:1], name=lab, fill="toself", opacity=0.55,
                line=dict(color=col, dash=dash, width=2),
                hovertemplate="%{theta}<br>Percentile %{r:,.0f}<extra>" + lab + "</extra>",
            ))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], ticksuffix="", gridcolor="rgba(128,128,128,0.25)"),
                                     bgcolor="rgba(0,0,0,0)"))
        show(fig, key="lookup_radar", height=430)
    else:
        fig = go.Figure()
        for lab, p, col in (("So với toàn bộ", p_all, "#2F7BEA"), ("So với cụm", p_cl, color)):
            fig.add_trace(go.Bar(
                y=names, x=[p[f] for f in feats], orientation="h", name=lab, marker_color=col,
                text=[vn_num(p[f], 0) for f in feats], textposition="outside",
                hovertemplate="%{y}<br>Percentile %{x:,.0f}<extra>" + lab + "</extra>",
            ))
        fig.add_vline(x=50, line_dash="dash", line_color="rgba(128,128,128,0.7)", annotation_text="Trung vị")
        fig.update_xaxes(title="Percentile (0 đến 100)", range=[0, 108])
        fig.update_yaxes(title="Đặc trưng", autorange="reversed")
        fig.update_layout(barmode="group")
        show(fig, key="lookup_bullet", height=430)
    source()
    how_to_read("Percentile 50 là mức trung vị. Hai hình/thanh lệch nhau nghĩa là người dùng này khác biệt khác nhau khi so với cả tập và so với đúng cụm của mình.")

    detail = pd.DataFrame({
        "Đặc trưng": [FEATURE_VI[f] for f in feats],
        "Giá trị": [float(row[f]) for f in feats],
        "Trung vị của cụm": [float(same[f].median()) for f in feats],
        "Percentile toàn bộ": [p_all[f] for f in feats],
        "Percentile trong cụm": [p_cl[f] for f in feats],
    })
    with st.expander("Bảng chi tiết", icon=":material/table:"):
        table(detail, column_config={
            "Giá trị": st.column_config.NumberColumn(format="%.2f"),
            "Trung vị của cụm": st.column_config.NumberColumn(format="%.2f"),
            "Percentile toàn bộ": st.column_config.NumberColumn(format="%.0f"),
            "Percentile trong cụm": st.column_config.NumberColumn(format="%.0f"),
        })
        download_csv(detail, f"{uid}_ho_so.csv", key="dl_person")

# ---------------------------------------------------------------- Lọc danh sách
st.markdown("### Lọc danh sách người dùng")
with st.expander("Lọc theo cụm và mức hoạt động rồi tải CSV", expanded=False, icon=":material/filter_alt:"):
    f1, f2, f3 = st.columns(3, gap="large")
    clusters = sorted(df["cluster"].unique().tolist())
    with f1:
        sel = st.multiselect("Cụm", clusters, default=clusters, format_func=lambda c: cluster_label(model, c),
                             key="flt_cluster")
    with f2:
        fmax = int(df["Frequency"].max()) if "Frequency" in df else 0
        fmin = st.slider("Frequency tối thiểu (số ngày có mua)", 0, max(fmax, 1), 0, key="flt_freq") if "Frequency" in df else 0
    with f3:
        rmax_all = int(df["Recency"].max()) if "Recency" in df else 0
        rmax = st.slider("Recency tối đa (ngày)", 0, max(rmax_all, 1), rmax_all, key="flt_rec") if "Recency" in df else 0

    mask = df["cluster"].isin(sel)
    if "Frequency" in df:
        mask &= df["Frequency"] >= fmin
    if "Recency" in df:
        mask &= df["Recency"] <= rmax
    out = df.loc[mask, ["Survey_ResponseID", "cluster"] + feats].copy()
    out.insert(2, "cluster_name", [cluster_label(model, int(c), with_id=False) for c in out["cluster"]])
    st.write(f"**{vn_int(len(out))}** người dùng phù hợp bộ lọc.")
    if out.empty:
        empty_state("Không có người dùng nào phù hợp", "Thử nới lỏng bộ lọc.")
    else:
        table(out.head(200))
        if len(out) > 200:
            st.caption("Bảng chỉ hiện 200 dòng đầu; tệp CSV tải về chứa toàn bộ kết quả lọc.")
        download_csv(out, f"nguoi_dung_da_loc_{model}.csv", key="dl_filtered")
