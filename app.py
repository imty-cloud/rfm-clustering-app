"""Điểm vào của app. Chạy: streamlit run app.py"""
import streamlit as st

st.set_page_config(
    page_title="Phân khúc người dùng bằng Spark",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils import ROOT, inject_css, sidebar_block  # noqa: E402  (import sau set_page_config)

inject_css()

try:
    st.logo(str(ROOT / "assets" / "logo.svg"))
except Exception:
    pass  # logo chỉ là phần trang trí

pages = [
    st.Page("pages/1_tong_quan.py", title="Tổng quan", icon=":material/dashboard:", default=True),
    st.Page("pages/2_ky_thuat_spark.py", title="Kỹ thuật Spark & chất lượng dữ liệu", icon=":material/bolt:"),
    st.Page("pages/3_kham_pha_phan_khuc.py", title="Khám phá phân khúc", icon=":material/bubble_chart:"),
    st.Page("pages/4_tra_cuu_nguoi_dung.py", title="Tra cứu người dùng", icon=":material/person_search:"),
    st.Page("pages/5_danh_gia_mo_hinh.py", title="Đánh giá mô hình", icon=":material/fact_check:"),
    st.Page("pages/6_nhom_nhieu_don_vi.py", title="Nhóm nhiều đơn vị mỗi dòng", icon=":material/inventory_2:"),
    st.Page("pages/7_insights_han_che.py", title="Insights & Hạn chế", icon=":material/lightbulb:"),
]

nav = st.navigation(pages)
sidebar_block()
nav.run()
