import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (FEATURE_SHORT, FEATURE_VI, MODEL_COLS, MODELS, callout, chart_title, cluster_color,
                   cluster_label, cluster_sizes, download_csv, features_present, how_to_read, load_clusters,
                   page_header, pca_2d, pick, read_csv, show, source, static_figures, table, vn_int, vn_num,
                   vn_pct, warn_missing)

page_header("Khám phá phân khúc", "Chọn mô hình để xem kích thước cụm, hồ sơ trung bình và vị trí các cụm trên mặt phẳng PCA.")

model = pick("Mô hình", MODELS, key="explore_model", default="B_RFM_Behavior")
if model == "B_RFM_Behavior":
    st.caption("Model B là mô hình chính để profiling và đặt tên cụm.")
else:
    st.caption("Model A chỉ dùng Recency, Frequency, Monetary. Các cụm của Model A được gọi theo số, không đặt tên.")

df = load_clusters(model)
if warn_missing(f"clusters_{model}") or df is None:
    st.stop()

feats = features_present(df)
if not feats:
    st.warning("Tệp cụm không có cột đặc trưng gốc (Recency, Frequency, ...).")
    st.stop()

sizes = cluster_sizes(df)
order = sizes["cluster"].tolist()
labels = {c: cluster_label(model, c) for c in order}

# ---------------------------------------------------------------- Donut + heatmap
col_a, col_b = st.columns([2, 3], gap="large")

with col_a:
    big = sizes.loc[sizes["n_users"].idxmax()]
    chart_title(
        f"Cụm lớn nhất chiếm {vn_pct(big['share'])} người dùng",
        f"{labels[int(big['cluster'])]}",
    )
    fig = go.Figure(go.Pie(
        labels=[labels[c] for c in sizes["cluster"]], values=sizes["n_users"], hole=0.62, sort=False,
        marker=dict(colors=[cluster_color(c) for c in sizes["cluster"]], line=dict(color="rgba(0,0,0,0)", width=0)),
        textinfo="percent", texttemplate="%{percent:.1%}", direction="clockwise",
        hovertemplate="%{label}<br>%{value:,.0f} người dùng (%{percent:.1%})<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{vn_int(len(df))}</b><br>người dùng", showarrow=False, font=dict(size=15))
    fig.update_layout(legend=dict(orientation="v", y=0.5, x=1.02, xanchor="left", yanchor="middle"))
    show(fig, key=f"donut_{model}", height=360)
    source()

with col_b:
    idx = df.groupby("cluster")[feats].mean() / df[feats].mean()
    idx = idx.loc[order]
    stacked = idx.stack()
    far_c, far_f = (stacked - 1).abs().idxmax()
    far_v = float(idx.loc[far_c, far_f])
    chart_title(
        f"Cụm {far_c} lệch xa trung bình nhất ở {FEATURE_SHORT[far_f]} (index {vn_num(far_v, 2)})",
        "Index = trung bình cụm / trung bình chung. Mốc 1,0 là bằng trung bình toàn bộ người dùng",
    )
    fig = go.Figure(go.Heatmap(
        z=idx.values, x=[FEATURE_SHORT[f] for f in feats], y=[f"Cụm {c}" for c in idx.index],
        colorscale="RdBu", reversescale=True, zmid=1, zmin=0, zmax=2,
        text=[[vn_num(v, 2) for v in row] for row in idx.values], texttemplate="%{text}",
        textfont=dict(size=13), colorbar=dict(title="Index", thickness=12),
        hovertemplate="Cụm %{y}<br>%{x}: index %{z:,.2f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", title="Cụm", type="category")
    fig.update_xaxes(title="Đặc trưng", side="bottom")
    show(fig, key=f"heat_{model}", height=360, legend=False)
    source()

how_to_read("Ô đỏ là cụm có trung bình cao hơn mức chung, ô xanh là thấp hơn. Với Recency, giá trị cao nghĩa là lần mua cuối cách đây lâu hơn.")

# ---------------------------------------------------------------- Bảng median
st.markdown("### Hồ sơ trung vị của từng cụm")
prof = read_csv(f"profile_{model}")
if prof is not None and "cluster" in prof.columns and all(f in prof.columns for f in feats):
    shown = prof[["cluster"] + [c for c in ("n_users", "share") if c in prof.columns] + feats].copy()
else:
    shown = df.groupby("cluster")[feats].median().reset_index()
    shown.insert(1, "n_users", sizes.set_index("cluster").loc[shown["cluster"], "n_users"].values)
    shown.insert(2, "share", (shown["n_users"] / len(df)).values)
if "share" in shown.columns:
    shown["share"] = shown["share"] * 100
shown.insert(1, "Tên cụm", [cluster_label(model, int(c), with_id=False) for c in shown["cluster"]])
shown = shown.rename(columns={"cluster": "Cụm", "n_users": "Người dùng", "share": "Tỷ trọng (%)", **FEATURE_SHORT})
cfg = {"Tỷ trọng (%)": st.column_config.NumberColumn(format="%.1f%%"),
       "Người dùng": st.column_config.NumberColumn(format="%d")}
for f in feats:
    cfg[FEATURE_SHORT[f]] = st.column_config.NumberColumn(format="%.2f" if f == "Avg_Quantity" else "%.1f")
table(shown, column_config=cfg)
source("Số trong bảng là trung vị.")
download_csv(shown, f"ho_so_trung_vi_{model}.csv", key=f"dl_prof_{model}")

# ---------------------------------------------------------------- PCA
st.markdown("### Vị trí các cụm trên mặt phẳng PCA")
cols = [c for c in MODEL_COLS[model] if c in df.columns]
if len(cols) < len(MODEL_COLS[model]):
    st.warning("Thiếu cột log/chuẩn hóa để tính PCA: " + ", ".join(sorted(set(MODEL_COLS[model]) - set(cols))))
else:
    Z, evr = pca_2d(df[cols])
    chart_title(
        "Các cụm nối tiếp nhau trên một dải liên tục, không tách thành đám rời",
        f"Hai thành phần đầu giải thích {vn_pct(evr.sum() * 100)} phương sai của các đặc trưng đã log và chuẩn hóa",
    )
    fig = go.Figure()
    hover_cols = [f for f in feats if f in df.columns]
    for c in order:
        m = (df["cluster"] == c).values
        cd = df.loc[m, hover_cols].values
        tpl = "<b>" + labels[c] + "</b><br>" + "<br>".join(
            f"{FEATURE_SHORT[f]}: %{{customdata[{i}]:,.2f}}" for i, f in enumerate(hover_cols)) + "<extra></extra>"
        fig.add_trace(go.Scattergl(
            x=Z[m, 0], y=Z[m, 1], mode="markers", name=labels[c], customdata=cd, hovertemplate=tpl,
            marker=dict(size=6, color=cluster_color(c), opacity=0.7, line=dict(width=0)),
        ))
    fig.update_xaxes(title=f"PC1 ({vn_pct(evr[0] * 100)} phương sai)")
    fig.update_yaxes(title=f"PC2 ({vn_pct(evr[1] * 100)} phương sai)")
    show(fig, key=f"pca_{model}", height=480)
    source("PCA tính từ các đặc trưng log, chỉ để minh họa, không dùng để phân cụm.")
    how_to_read("Mỗi chấm là một người dùng. Các màu chồng lấn ở vùng biên cho thấy ranh giới cụm là tương đối, không phải ranh giới tự nhiên.")

# ---------------------------------------------------------------- Box / violin
st.markdown("### Phân bố của một đặc trưng theo cụm")
fc, tc, lc = st.columns([3, 2, 2], gap="large")
with fc:
    feat = st.selectbox("Đặc trưng", feats, format_func=lambda f: FEATURE_VI[f], key=f"dist_feat_{model}")
with tc:
    kind = pick("Dạng biểu đồ", {"box": "Hộp", "violin": "Violin"}, key=f"dist_kind_{model}", default="box")
with lc:
    use_log = st.toggle("Thang log", value=feat in ("Frequency", "Monetary", "Unique_Products", "Recency"), key=f"dist_log_{model}")

med = df.groupby("cluster")[feat].median()
hi_c = int(med.idxmax())
chart_title(
    f"Cụm {hi_c} có trung vị {FEATURE_SHORT[feat]} cao nhất ({vn_num(med.max(), 2)})",
    FEATURE_VI[feat],
)
fig = go.Figure()
for c in order:
    y = df.loc[df["cluster"] == c, feat]
    common = dict(y=y, name=f"Cụm {c}", marker_color=cluster_color(c), legendgroup=str(c))
    if kind == "violin":
        fig.add_trace(go.Violin(**common, line_color=cluster_color(c), box_visible=True, meanline_visible=True, points=False))
    else:
        fig.add_trace(go.Box(**common, boxmean=True, boxpoints="outliers", marker_size=3))
fig.update_yaxes(title=FEATURE_VI[feat], type="log" if use_log and (df[feat] > 0).all() else "linear")
fig.update_xaxes(title="Cụm")
show(fig, key=f"dist_{model}_{feat}_{kind}", height=400)
source("Đường đứt trong hộp là trung bình; vạch liền là trung vị.")
if use_log and not (df[feat] > 0).all():
    st.caption("Đặc trưng này có giá trị 0 nên thang log không áp dụng được, biểu đồ dùng thang tuyến tính.")

static_figures(["fig08", "fig09"], key=f"fig_explore_{model}")
