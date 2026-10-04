"""Tiện ích dùng chung cho app Streamlit (chỉ đọc file đã lưu, không chạy lại Spark/K-Means)."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------------------
# 1. CẤU HÌNH ĐƯỜNG DẪN (đổi bằng biến môi trường DULIEULON_DIR nếu thư mục nằm chỗ khác)
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

REL = {
    "row_filter": "audit/row_filter_steps.csv",
    "rows_by_year": "audit/rows_by_year.csv",
    "missing": "audit/missing_by_column.csv",
    "user_features": "features/user_features.csv",
    "k_sweep": "validation/k_sweep.csv",
    "ari_models": "validation/ari_between_models.csv",
    "scalability": "validation/scalability.csv",
    "stability": "stability/stability.csv",
    "profile_A_RFM": "profiling/cluster_profile_A_RFM.csv",
    "profile_B_RFM_Behavior": "profiling/cluster_profile_B_RFM_Behavior.csv",
    "niche": "profiling/category_niche_vs_rest.csv",
    "clusters_A_RFM": "final/user_clusters_A_RFM.csv",
    "clusters_B_RFM_Behavior": "final/user_clusters_B_RFM_Behavior.csv",
    "robustness": "final/robustness.csv",
    "window": "final/window_sensitivity.csv",
    "final_summary": "final/final_model_summary.csv",
    "parser_fix": "report_exports/parser_fix_user_level.csv",
}

def _count_found(d: Path) -> int:
    try:
        return sum((d / rel).exists() for rel in REL.values())
    except OSError:
        return 0


def _find_data_dir() -> Path:
    """Chọn thư mục chứa nhiều tệp kết quả nhất: biến môi trường, rồi các vị trí thường gặp."""
    cands: list[Path] = []
    env = os.environ.get("DULIEULON_DIR")
    if env:
        cands.append(Path(env).expanduser())
    cands += [ROOT / "Dulieulon", ROOT.parent, ROOT.parent.parent]
    docs = Path.home() / "Documents"
    try:
        cands += sorted(docs.glob("*/Dulieulon")) + [docs / "Dulieulon"]
    except OSError:
        pass
    best, best_n = cands[0] if cands else ROOT / "Dulieulon", -1
    for c in cands:
        n = _count_found(c)
        if n > best_n:
            best, best_n = c, n
    return best


DATA_DIR = _find_data_dir()
FIG_DIR = DATA_DIR / "figures"

SOURCE = "Nguồn: Tác giả xử lý từ dữ liệu nghiên cứu."

# --------------------------------------------------------------------------------------
# 2. SỐ LIỆU CỐ ĐỊNH (lấy từ đề bài, không tính lại)
# --------------------------------------------------------------------------------------
FIXED = {
    "raw_rows": 1_850_717,
    "valid_id_rows": 1_811_238,
    "invalid_rows": 39_479,
    "clean_rows": 1_772_132,
    "users": 5_026,
    "survey_ids": 5_027,
    "n_features": 7,
    "k": 4,
    "n_seeds": 30,
    "winsor_p99": 1.45,
    "parser_median_pct": 1.91,
    "ari_parser": {"A_RFM": 0.976, "B_RFM_Behavior": 0.973},
    "ari_ab": 0.396,
    "niche_users": 298,
}

# Ví dụ ID sai: kết quả chẩn đoán trong notebook (mục "Chẩn đoán Survey_ResponseID không hợp lệ")
INVALID_ID_EXAMPLES = [
    ("RUG", 541),
    ("CURTAIN", 516),
    ("BLANKET", 362),
    ("PORTABLE_ELECTRONIC_DEVICE_COVER", 359),
]

# Môi trường thử nghiệm: trích từ phần "Môi trường thực nghiệm" của notebook
ENV_INFO = [
    ("Apache Spark", "4.2.0"),
    ("Chế độ chạy", "local[*] (một máy)"),
    ("CPU / RAM", "10 nhân / 16 GB"),
    ("spark.driver.memory", "8g"),
    ("spark.sql.shuffle.partitions", "32"),
]

MODELS = {"A_RFM": "Model A (RFM)", "B_RFM_Behavior": "Model B (RFM + hành vi)"}
MODEL_SHORT = {"A_RFM": "A", "B_RFM_Behavior": "B"}

FEATURES = ["Recency", "Frequency", "Monetary", "Unique_Products",
            "Category_Diversity", "Active_Months", "Avg_Quantity"]
FEATURE_VI = {
    "Recency": "Recency (ngày kể từ lần mua cuối)",
    "Frequency": "Frequency (số ngày có mua hàng)",
    "Monetary": "Monetary (tổng chi tiêu)",
    "Unique_Products": "Số sản phẩm khác nhau",
    "Category_Diversity": "Số nhóm hàng khác nhau",
    "Active_Months": "Số tháng có hoạt động",
    "Avg_Quantity": "Số lượng trung bình mỗi dòng",
}
FEATURE_SHORT = {
    "Recency": "Recency", "Frequency": "Frequency", "Monetary": "Monetary",
    "Unique_Products": "Sản phẩm", "Category_Diversity": "Nhóm hàng",
    "Active_Months": "Tháng hoạt động", "Avg_Quantity": "SL trung bình",
}
LOG_FEATURES = ["Recency", "Frequency", "Monetary", "Unique_Products", "Category_Diversity"]
MODEL_COLS = {
    "A_RFM": ["Recency_log", "Frequency_log", "Monetary_log"],
    "B_RFM_Behavior": ["Recency_log", "Frequency_log", "Monetary_log", "Unique_Products_log",
                       "Category_Diversity_log", "Active_Months", "Avg_Quantity_w"],
}

# Tên cụm của Model B (thống nhất với báo cáo, KHÔNG dùng cluster_names.csv cũ)
CLUSTER_NAMES_B = {
    0: "Hoạt động thấp, lần mua cuối xa",
    1: "Hoạt động trung bình",
    2: "Nhiều đơn vị mỗi dòng",
    3: "Hoạt động cao, mua gần đây",
}
CLUSTER_SHARE_B = {0: 16.6, 1: 35.4, 2: 5.9, 3: 42.0}

# Palette cố định 4 cụm: xanh dương, xanh lục, cam, đỏ
CLUSTER_COLORS = {0: "#2F7BEA", 1: "#22A06B", 2: "#F5963A", 3: "#E5484D"}
NEUTRAL = "#8A94A6"
FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"


# --------------------------------------------------------------------------------------
# 3. ĐỊNH DẠNG SỐ KIỂU VIỆT NAM (1.234,56)
# --------------------------------------------------------------------------------------
def vn_num(x, d: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    s = f"{x:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def vn_int(x) -> str:
    return vn_num(x, 0)


def vn_pct(x, d: int = 1) -> str:
    """x đã ở dạng phần trăm (16.6 → '16,6%')."""
    return vn_num(x, d) + "%"


# --------------------------------------------------------------------------------------
# 4. ĐỌC DỮ LIỆU (cache, an toàn khi thiếu file)
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _read_csv(path: str, mtime: float) -> pd.DataFrame:
    # mtime nằm trong khóa cache: file đổi thì cache tự làm mới
    return pd.read_csv(path)


def read_csv(key_or_rel: str) -> pd.DataFrame | None:
    rel = REL.get(key_or_rel, key_or_rel)
    p = DATA_DIR / rel
    if not p.exists():
        return None
    try:
        return _read_csv(str(p), p.stat().st_mtime)
    except Exception:  # file hỏng / sai định dạng
        return None


def has_cols(df: pd.DataFrame | None, cols) -> bool:
    return df is not None and all(c in df.columns for c in cols)


def warn_missing(*keys: str) -> bool:
    """Hiện st.warning thân thiện nếu thiếu file. Trả về True nếu có file bị thiếu."""
    miss = [REL.get(k, k) for k in keys if read_csv(k) is None]
    if miss:
        st.warning(
            "Chưa đọc được: " + ", ".join(f"`{m}`" for m in miss)
            + f" trong thư mục `{DATA_DIR}`. Phần này sẽ hiện khi có đủ tệp. "
            "Nếu dữ liệu nằm chỗ khác, đặt biến môi trường `DULIEULON_DIR` trỏ tới thư mục chứa `audit/`, `final/`, `validation/`.",
            icon=":material/folder_off:",
        )
    return bool(miss)


def load_clusters(model: str) -> pd.DataFrame | None:
    """Đọc user_clusters_<model>.csv, bổ sung các cột log nếu file chưa có."""
    df = read_csv(f"clusters_{model}")
    if df is None:
        return None
    if "cluster" not in df.columns or "Survey_ResponseID" not in df.columns:
        st.warning(f"Tệp user_clusters_{model}.csv thiếu cột `cluster` hoặc `Survey_ResponseID`.")
        return None
    d = df.copy()
    d["cluster"] = d["cluster"].astype(int)
    for c in LOG_FEATURES:
        if f"{c}_log" not in d.columns and c in d.columns:
            d[f"{c}_log"] = np.log1p(d[c])
    if "Avg_Quantity_w" not in d.columns and "Avg_Quantity" in d.columns:
        d["Avg_Quantity_w"] = d["Avg_Quantity"].clip(upper=FIXED["winsor_p99"])
    return d


def features_present(df: pd.DataFrame) -> list[str]:
    return [f for f in FEATURES if f in df.columns]


@st.cache_data(show_spinner=False)
def pca_2d(X: pd.DataFrame):
    """PCA 2 chiều trên các cột log đã chuẩn hóa (chỉ để vẽ, không phải bước phân cụm)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    Z = StandardScaler().fit_transform(X.values)
    p = PCA(n_components=2, random_state=0).fit(Z)
    return p.transform(Z), p.explained_variance_ratio_


def pct_rank(series: pd.Series, value: float) -> float:
    s = series.dropna()
    return float((s <= value).mean() * 100) if len(s) else float("nan")


# --------------------------------------------------------------------------------------
# 5. NHÃN & MÀU CỤM
# --------------------------------------------------------------------------------------
def cluster_label(model: str, c: int, with_id: bool = True) -> str:
    if model == "B_RFM_Behavior":
        return f"{c} · {CLUSTER_NAMES_B.get(int(c), '')}" if with_id else CLUSTER_NAMES_B.get(int(c), f"Cụm {c}")
    return f"Cụm {c} (Model A)"


def cluster_color(c: int) -> str:
    return CLUSTER_COLORS.get(int(c), NEUTRAL)


def cluster_sizes(df: pd.DataFrame) -> pd.DataFrame:
    s = df["cluster"].value_counts().sort_index()
    return pd.DataFrame({"cluster": s.index, "n_users": s.values, "share": s.values / len(df) * 100})


# --------------------------------------------------------------------------------------
# 6. PLOTLY: TEMPLATE THỐNG NHẤT
# --------------------------------------------------------------------------------------
PLOT_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "bieu_do_phan_khuc"},
}


def style(fig, height: int | None = 380, legend: bool = True):
    fig.update_layout(
        font=dict(family=FONT, size=13),
        separators=",.",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=28, b=8),
        hoverlabel=dict(font_size=13, font_family=FONT),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False, linecolor="rgba(128,128,128,0.35)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.18)", zeroline=False, linecolor="rgba(128,128,128,0.35)")
    return fig


def show(fig, key: str | None = None, height: int | None = 380, legend: bool = True):
    style(fig, height, legend)
    try:
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, key=key)
    except TypeError:  # Streamlit cũ chưa có tham số width
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG, key=key)


def table(df: pd.DataFrame, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


# --------------------------------------------------------------------------------------
# 7. THÀNH PHẦN GIAO DIỆN
# --------------------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, button, input, textarea, select { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1280px; }
h1, h2, h3 { letter-spacing: -0.01em; font-weight: 650; }
h2, h3 { margin-top: 1.6rem; }

.hero { padding: .4rem 0 1.2rem 0; }
.hero h1 { font-size: 2.1rem; line-height: 1.2; margin: .9rem 0 .5rem 0; max-width: 46ch; }
.hero p { font-size: 1.02rem; opacity: .78; max-width: 70ch; margin: 0; }
.stripe { display: flex; gap: 3px; height: 8px; border-radius: 99px; overflow: hidden; }
.stripe span { display: block; height: 100%; }

.seg-card { min-height: 8.6rem; border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: 1rem 1.2rem;
  background: rgba(128,128,128,.06); box-shadow: 0 1px 3px rgba(0,0,0,.10); height: 100%; }
.kpi-label { font-size: .85rem; opacity: .72; }
.kpi-value { font-size: 2.3rem; font-weight: 700; line-height: 1.15; font-variant-numeric: tabular-nums; margin: .1rem 0; }
.kpi-note { font-size: .8rem; opacity: .65; }

.callout { border-radius: 12px; padding: .75rem 1rem; margin: .6rem 0 .9rem 0; border-left: 4px solid; font-size: .93rem; line-height: 1.5; }
.callout .ct { font-weight: 600; margin-bottom: .1rem; }
.callout.info { background: rgba(47,123,234,.10); border-color: #2F7BEA; }
.callout.insight { background: rgba(34,160,107,.11); border-color: #22A06B; }
.callout.warning { background: rgba(245,150,58,.14); border-color: #F5963A; }

.chart-title { font-weight: 600; font-size: 1.05rem; margin: .5rem 0 .1rem 0; line-height: 1.35; }
.chart-sub { opacity: .66; font-size: .85rem; margin-bottom: .25rem; }

.pipe { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .9rem; margin: .4rem 0 1rem 0; }
.pipe .step { position: relative; border: 1px solid rgba(128,128,128,.28); border-radius: 12px; padding: .7rem .8rem;
  background: rgba(128,128,128,.06); }
.pipe .step:not(:last-child)::after { content: "›"; position: absolute; right: -.7rem; top: 50%; transform: translateY(-50%);
  font-size: 1.4rem; opacity: .5; }
.pipe .t { font-weight: 600; font-size: .95rem; }
.pipe .s { font-size: .8rem; opacity: .7; margin-top: .15rem; line-height: 1.35; }

.tl { border-left: 2px solid rgba(128,128,128,.35); margin: .4rem 0 1rem .5rem; padding-left: 1.1rem; }
.tl .it { position: relative; margin-bottom: .9rem; }
.tl .it::before { content: ""; position: absolute; left: -1.5rem; top: .35rem; width: 10px; height: 10px; border-radius: 50%;
  background: var(--dot, #2F7BEA); }
.tl .it b { display: block; }
.tl .it span { font-size: .88rem; opacity: .75; }

.chip { display: inline-block; border-radius: 99px; border: 1px solid rgba(128,128,128,.35); padding: .15rem .75rem;
  font-size: .82rem; margin: .15rem .3rem .15rem 0; font-variant-numeric: tabular-nums; }
.empty { text-align: center; padding: 2rem 1rem; border: 1px dashed rgba(128,128,128,.45); border-radius: 14px; opacity: .85; }

.person { display: flex; gap: .8rem; align-items: center; flex-wrap: wrap; }
.person .dot { width: 14px; height: 14px; border-radius: 50%; display: inline-block; }
.person .id { font-weight: 600; font-size: 1.1rem; }

.side-brand { font-weight: 700; font-size: 1.15rem; letter-spacing: -0.01em; margin-bottom: .1rem; }
.side-topic { font-size: .82rem; opacity: .75; line-height: 1.4; margin: .4rem 0 .7rem 0; }
.side-status { font-size: .8rem; display: flex; align-items: center; gap: .45rem; margin-top: .5rem; }
.side-status i { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def stripe_html(height_px: int | None = None) -> str:
    spans = "".join(
        f'<span style="width:{CLUSTER_SHARE_B[c]}%;background:{CLUSTER_COLORS[c]}"></span>' for c in range(4)
    )
    style_h = f' style="height:{height_px}px"' if height_px else ""
    return f'<div class="stripe"{style_h}>{spans}</div>'


def hero(title: str, sub: str):
    st.markdown(f'<div class="hero">{stripe_html()}<h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


def page_header(title: str, sub: str):
    st.markdown(f"## {title}")
    st.caption(sub)


def kpi(label: str, value: str, note: str = ""):
    st.markdown(
        f'<div class="seg-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def callout(kind: str, text: str, title: str | None = None):
    t = f'<div class="ct">{title}</div>' if title else ""
    st.markdown(f'<div class="callout {kind}">{t}<div>{text}</div></div>', unsafe_allow_html=True)


def how_to_read(text: str):
    callout("info", text, title="Cách đọc")


def chart_title(text: str, sub: str | None = None):
    s = f'<div class="chart-sub">{sub}</div>' if sub else ""
    st.markdown(f'<div class="chart-title">{text}</div>{s}', unsafe_allow_html=True)


def source(extra: str = ""):
    st.caption(SOURCE + (" " + extra if extra else ""))


def empty_state(title: str, hint: str = ""):
    st.markdown(f'<div class="empty"><b>{title}</b><br><span style="font-size:.9rem">{hint}</span></div>',
                unsafe_allow_html=True)


def pick(label: str, options: dict, key: str, default: str | None = None):
    """Segmented control (Streamlit mới) hoặc radio ngang (bản cũ). options: {giá trị: nhãn}."""
    keys = list(options)
    default = default or keys[0]
    if hasattr(st, "segmented_control"):
        v = st.segmented_control(label, keys, default=default, format_func=lambda k: options[k],
                                 key=key, label_visibility="collapsed")
        return v or default
    return st.radio(label, keys, index=keys.index(default), format_func=lambda k: options[k],
                    horizontal=True, key=key, label_visibility="collapsed")


# --------------------------------------------------------------------------------------
# 8. TẢI VỀ
# --------------------------------------------------------------------------------------
def download_csv(df: pd.DataFrame, filename: str, key: str, label: str = "Tải bảng CSV"):
    st.download_button(label, df.to_csv(index=False).encode("utf-8-sig"), file_name=filename,
                       mime="text/csv", key=key, icon=":material/download:")


def find_figure(prefix: str) -> Path | None:
    if not FIG_DIR.exists():
        return None
    hits = sorted(FIG_DIR.glob(f"{prefix}*.png"))
    return hits[0] if hits else None


def static_figures(prefixes: list[str], title: str = "Xem hình tĩnh dùng trong báo cáo", key: str = "fig"):
    """Expander hiển thị hình PNG đã xuất sẵn + nút tải. Biểu đồ Plotly cũng có nút chụp ảnh ở góc."""
    paths = [(p, find_figure(p)) for p in prefixes]
    paths = [(p, f) for p, f in paths if f is not None]
    with st.expander(title, icon=":material/image:"):
        if not paths:
            empty_state("Chưa có hình PNG", f"Thêm tệp {', '.join(prefixes)}*.png vào thư mục figures/.")
            return
        for p, f in paths:
            st.image(str(f), caption=f.name, width="stretch")
            st.download_button("Tải hình PNG", f.read_bytes(), file_name=f.name, mime="image/png",
                               key=f"{key}_{p}", icon=":material/download:")


# --------------------------------------------------------------------------------------
# 9. TRẠNG THÁI DỮ LIỆU (sidebar)
# --------------------------------------------------------------------------------------
def data_status() -> tuple[int, int, str | None]:
    total = len(REL)
    found, latest = 0, 0.0
    for rel in REL.values():
        p = DATA_DIR / rel
        if p.exists():
            found += 1
            latest = max(latest, p.stat().st_mtime)
    stamp = datetime.fromtimestamp(latest).strftime("%d/%m/%Y %H:%M") if latest else None
    return found, total, stamp


def sidebar_block():
    found, total, stamp = data_status()
    ok = found == total
    color = "#22A06B" if ok else ("#F5963A" if found else "#E5484D")
    status = f"Dữ liệu cập nhật {stamp}" if stamp else "Chưa tìm thấy dữ liệu"
    with st.sidebar:
        st.markdown(
            f'{stripe_html(5)}'
            '<div class="side-topic">Phân khúc người dùng từ dữ liệu giao dịch quy mô lớn bằng Apache Spark: '
            'so sánh RFM và RFM + đặc trưng hành vi với K-Means (Amazon Purchases)</div>'
            f'<div class="side-status"><i style="background:{color}"></i>{status}</div>'
            f'<div class="side-topic">{found}/{total} tệp dữ liệu sẵn sàng</div>',
            unsafe_allow_html=True,
        )
