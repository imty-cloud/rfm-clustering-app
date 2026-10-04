import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (CLUSTER_COLORS, ENV_INFO, FIXED, INVALID_ID_EXAMPLES, callout, chart_title, download_csv,
                   has_cols, kpi, page_header, read_csv, show, source, static_figures, table, vn_int,
                   vn_num, vn_pct, warn_missing)

page_header(
    "Kỹ thuật Spark và chất lượng dữ liệu",
    "Một lỗi đọc CSV được phát hiện bằng audit, đo mức ảnh hưởng và kiểm tra lại kết quả phân cụm.",
)

# ---------------------------------------------------------------- 1. Lỗi ID
st.markdown("### Lỗi đọc CSV làm sai Survey_ResponseID")
chart_title(
    f"{vn_int(FIXED['invalid_rows'])} dòng bị lệch cột và được phát hiện nhờ kiểm tra dạng ID",
    "Dấu nháy kép trong tiêu đề sản phẩm khiến Spark (escape mặc định) cắt sai trường",
)
t1, t2 = st.columns([3, 2], gap="large")
with t1:
    st.markdown(
        '<div class="tl">'
        '<div class="it" style="--dot:#2F7BEA"><b>1. Đọc CSV bằng cấu hình mặc định</b>'
        "<span>Tiêu đề sản phẩm có dấu nháy kép (ví dụ kích thước 8\") làm trường bị tách sai.</span></div>"
        '<div class="it" style="--dot:#F5963A"><b>2. ID chứa mảnh của trường khác</b>'
        "<span>Cột Survey_ResponseID xuất hiện giá trị như RUG, CURTAIN, BLANKET thay vì dạng R_xxx.</span></div>"
        '<div class="it" style="--dot:#E5484D"><b>3. Audit bằng regex</b>'
        f"<span>Giữ ID khớp <code>^R_[A-Za-z0-9]+$</code>, loại {vn_int(FIXED['invalid_rows'])} dòng.</span></div>"
        '<div class="it" style="--dot:#22A06B"><b>4. Đọc lại đúng và kiểm tra K = 4</b>'
        f"<span>ARI so với nghiệm cuối: {vn_num(FIXED['ari_parser']['A_RFM'], 3)} (A), "
        f"{vn_num(FIXED['ari_parser']['B_RFM_Behavior'], 3)} (B). Chỉ K = 4 được kiểm tra lại.</span></div>"
        "</div>",
        unsafe_allow_html=True,
    )
with t2:
    chips = "".join(f'<span class="chip"><b>{n}</b> · {vn_int(c)} dòng</span>' for n, c in INVALID_ID_EXAMPLES)
    st.markdown(
        '<div class="seg-card"><div class="kpi-label">Ví dụ ID sai thường gặp nhất</div>'
        f'<div style="margin-top:.5rem">{chips}</div>'
        '<div class="kpi-note" style="margin-top:.6rem">Đây là tên nhóm hàng bị lẫn vào cột ID, không phải người dùng thật.</div></div>',
        unsafe_allow_html=True,
    )
source("Ví dụ lấy từ bước chẩn đoán ID trong notebook.")

# ---------------------------------------------------------------- 2. Histogram parser fix
st.markdown("### Mức ảnh hưởng của lỗi đọc lên từng người dùng")
pf = read_csv("parser_fix")
if warn_missing("parser_fix") or pf is None:
    kpi("Trung vị % dòng mất", vn_pct(FIXED["parser_median_pct"], 2), "Giá trị cố định từ báo cáo")
else:
    num_cols = [c for c in pf.columns if pd.api.types.is_numeric_dtype(pf[c])]
    pref = [c for c in num_cols if any(k in c.lower() for k in ("pct", "percent", "ty_le", "loss", "lost", "mat"))]
    cols_pick = pref or num_cols
    if not cols_pick:
        st.warning("Tệp parser_fix_user_level.csv không có cột số để vẽ histogram.")
    else:
        col = st.selectbox("Cột dùng để vẽ", cols_pick, index=0, key="pf_col") if len(cols_pick) > 1 else cols_pick[0]
        vals = pf[col].dropna().astype(float)
        if vals.max() <= 1.0:  # file lưu dạng tỷ lệ → đổi sang %
            vals = vals * 100
        med = float(vals.median())
        chart_title(
            f"Trung vị người dùng mất {vn_pct(med, 2)} số dòng khi đọc sai",
            f"Phân bố theo {len(vals):,} người dùng".replace(",", "."),
        )
        fig = go.Figure(go.Histogram(
            x=vals, nbinsx=40, marker=dict(color=CLUSTER_COLORS[0], line=dict(color="rgba(255,255,255,0.4)", width=0.5)),
            hovertemplate="%{x:,.2f}%<br>%{y:,.0f} người dùng<extra></extra>",
        ))
        fig.add_vline(x=med, line_dash="dash", line_color=CLUSTER_COLORS[3],
                      annotation_text=f"Trung vị {vn_pct(med, 2)}", annotation_position="top right")
        fig.update_xaxes(title="Tỷ lệ dòng của người dùng bị mất (%)", ticksuffix="%")
        fig.update_yaxes(title="Số người dùng")
        show(fig, key="pf_hist", height=340, legend=False)
        source()
        if abs(med - FIXED["parser_median_pct"]) > 0.05:
            st.caption(f"Lưu ý: trung vị tính từ tệp là {vn_pct(med, 2)}, báo cáo ghi {vn_pct(FIXED['parser_median_pct'], 2)}.")
        download_csv(pf, "parser_fix_user_level.csv", key="dl_pf")

# ---------------------------------------------------------------- 3. Code
st.markdown("### Đoạn mã chính của pipeline")
st.caption("Rút gọn từ notebook. Tham số escape/multiLine ở tab đầu là cách khắc phục chuẩn, đối chiếu lại với notebook của bạn.")
tab_read, tab_filter, tab_group, tab_km = st.tabs(["Đọc CSV", "Lọc ID và thời gian", "groupBy 7 đặc trưng", "K-Means"])
with tab_read:
    st.code('''# Mặc định (gây lỗi): spark.read.csv(RAW, header=True, inferSchema=True)
df = (spark.read
      .option("header", True)
      .option("inferSchema", True)
      .option("multiLine", True)
      .option("quote", '"')
      .option("escape", '"')      # dấu " trong chuỗi được nhân đôi, không dùng \\
      .csv(RAW))
df = df.toDF(*[re.sub(r"[^0-9a-zA-Z]+", "_", c).strip("_") for c in df.columns])''', language="python")
with tab_filter:
    st.code('''valid = F.col("Survey_ResponseID").rlike(r"^R_[A-Za-z0-9]+$")
START, END = "2018-01-01", "2022-12-31"

d = df.filter(valid)                                              # loại ID sai
d = d.filter(F.col("Order_Date").between(START, END))             # cửa sổ thời gian
d = d.filter((F.col("Purchase_Price_Per_Unit") > 0) & (F.col("Quantity") > 0))
clean = d.withColumn("Spend", F.col("Purchase_Price_Per_Unit") * F.col("Quantity"))
clean.write.mode("overwrite").parquet(f"{OUT}/audit/clean_transactions")''', language="python")
with tab_group:
    st.code('''u = tx.groupBy("Survey_ResponseID").agg(
    F.datediff(F.to_date(F.lit("2022-12-31")), F.max("Order_Date")).alias("Recency"),
    F.countDistinct("Order_Date").alias("Frequency"),          # số NGÀY có mua, không có mã đơn
    F.sum("Spend").alias("Monetary"),
    F.countDistinct("ASIN_ISBN_Product_Code").alias("Unique_Products"),
    F.countDistinct("Category").alias("Category_Diversity"),
    F.countDistinct(F.date_format("Order_Date", "yyyy-MM")).alias("Active_Months"),
    F.avg("Quantity").alias("Avg_Quantity"),
)
cap = u.approxQuantile("Avg_Quantity", [0.99], 0.001)[0]       # winsorize p99
u = u.withColumn("Avg_Quantity_w", F.least(F.col("Avg_Quantity"), F.lit(cap)))
for c in ["Frequency", "Monetary", "Unique_Products"]:
    u = u.withColumn(c + "_log", F.log1p(c))''', language="python")
with tab_km:
    st.code('''from pyspark.ml.clustering import KMeans

runs = []
for s in range(1, 31):                                          # 30 seed
    m = KMeans(k=4, seed=s, featuresCol="features", maxIter=100).fit(data)
    runs.append((m.summary.trainingCost, s, m))
best = min(runs, key=lambda r: r[0])                            # WSSSE nhỏ nhất
# Đánh số lại cụm theo Monetary trung vị tăng dần để nhãn nhất quán''', language="python")

# ---------------------------------------------------------------- 4. Scalability
st.markdown("### Thời gian chạy khi tăng dữ liệu")
sc = read_csv("scalability")
if warn_missing("scalability") or not has_cols(sc, ["fraction", "seconds"]):
    pass
else:
    s = sc.sort_values("fraction")
    pct = (s["fraction"] * 100).round().astype(int)
    first, last = float(s["seconds"].iloc[0]), float(s["seconds"].iloc[-1])
    chart_title(
        f"Dựng đặc trưng mất {vn_num(first, 1)} giây ở 10% dòng và {vn_num(last, 1)} giây ở 100% dòng",
        "Thử nghiệm trên một máy, chế độ local[*]",
    )
    fig = go.Figure(go.Scatter(
        x=pct, y=s["seconds"], mode="lines+markers+text", text=[vn_num(v, 1) for v in s["seconds"]],
        textposition="top center", line=dict(color=CLUSTER_COLORS[0], width=3), marker=dict(size=9),
        customdata=s["n_rows"] if "n_rows" in s else None,
        hovertemplate="%{x}% dòng<br>%{y:,.1f} giây<extra></extra>",
    ))
    fig.update_xaxes(title="Tỷ lệ dòng được dùng (%)", tickvals=pct.tolist(), ticksuffix="%")
    fig.update_yaxes(title="Thời gian (giây)", rangemode="tozero")
    sc_l, sc_r = st.columns([3, 2], gap="large")
    with sc_l:
        show(fig, key="scal", height=340, legend=False)
        source()
    with sc_r:
        callout("warning", "Kết quả này <b>chưa đủ để kết luận khả năng mở rộng</b>: chỉ một máy, dữ liệu 1,77 triệu dòng, "
                           "thời gian chỉ vài giây nên dễ bị nhiễu bởi khởi động và cache.", title="Đọc cẩn thận")
        table(pd.DataFrame(ENV_INFO, columns=["Môi trường", "Giá trị"]))
    download_csv(sc, "scalability.csv", key="dl_scal")

# ---------------------------------------------------------------- 5. Chất lượng dữ liệu thô
st.markdown("### Chất lượng dữ liệu thô")
by_year, miss = read_csv("rows_by_year"), read_csv("missing")
tab_y, tab_m = st.tabs(["Số dòng theo năm", "Giá trị thiếu theo cột"])
with tab_y:
    if warn_missing("rows_by_year") or by_year is None:
        pass
    else:
        ycol = "year" if "year" in by_year.columns else by_year.columns[0]
        ncol = "count" if "count" in by_year.columns else by_year.columns[-1]
        chart_title("Số dòng giao dịch thô theo năm đặt hàng", "Cửa sổ phân tích là 2018 đến 2022")
        fig = go.Figure(go.Bar(x=by_year[ycol].astype(str), y=by_year[ncol], marker_color=CLUSTER_COLORS[0],
                               hovertemplate="Năm %{x}<br>%{y:,.0f} dòng<extra></extra>"))
        fig.update_xaxes(title="Năm", type="category")
        fig.update_yaxes(title="Số dòng")
        show(fig, key="by_year", height=320, legend=False)
        source()
with tab_m:
    if warn_missing("missing") or miss is None:
        pass
    else:
        ccol = miss.columns[0]
        pcol = "pct" if "pct" in miss.columns else miss.columns[-1]
        m = miss.sort_values(pcol, ascending=True)
        chart_title("Tỷ lệ giá trị thiếu theo cột trong dữ liệu thô", "Chỉ hiển thị phần trăm dòng thiếu")
        fig = go.Figure(go.Bar(x=m[pcol], y=m[ccol].astype(str), orientation="h", marker_color=CLUSTER_COLORS[2],
                               hovertemplate="%{y}<br>%{x:,.3f}% thiếu<extra></extra>"))
        fig.update_xaxes(title="Tỷ lệ thiếu (%)", ticksuffix="%")
        fig.update_yaxes(title="Cột")
        show(fig, key="missing", height=340, legend=False)
        source()
        download_csv(miss, "missing_by_column.csv", key="dl_missing")

static_figures(["fig02"], key="fig_quality")
