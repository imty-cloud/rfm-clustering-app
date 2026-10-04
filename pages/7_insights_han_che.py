import streamlit as st

from utils import (CLUSTER_COLORS, CLUSTER_NAMES_B, FIXED, callout, has_cols, load_clusters, page_header,
                   read_csv, vn_int, vn_num, vn_pct)

page_header("Insights và Hạn chế", "Ba phát hiện chính và những giới hạn cần nêu rõ khi dùng kết quả.")


def finding(color: str, title: str, body: str):
    st.markdown(
        f'<div class="seg-card" style="border-top:4px solid {color}">'
        f'<div style="font-weight:650;font-size:1.05rem;margin-bottom:.4rem;line-height:1.35">{title}</div>'
        f'<div style="font-size:.93rem;line-height:1.55;opacity:.9">{body}</div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- Số liệu cho phát hiện 1
n_niche, share_niche, q_niche, q_other = FIXED["niche_users"], 5.9, None, None
dfb = load_clusters("B_RFM_Behavior")
if dfb is not None and "Avg_Quantity" in dfb.columns:
    n_niche = int((dfb["cluster"] == 2).sum())
    share_niche = n_niche / len(dfb) * 100
    med = dfb.groupby("cluster")["Avg_Quantity"].median()
    q_niche = float(med.get(2, float("nan")))
    q_other = float(med.drop(index=2, errors="ignore").max())

# ---------------------------------------------------------------- Số liệu cho phát hiện 2
rob = read_csv("robustness")
ari = {}
if has_cols(rob, ["model", "variant", "ARI_vs_final"]):
    for _, r in rob.iterrows():
        ari[(r["model"], r["variant"])] = float(r["ARI_vs_final"])
else:  # số cố định từ đề bài
    ari = {("A_RFM", "baseline_sklearn"): 0.991, ("B_RFM_Behavior", "baseline_sklearn"): 1.000,
           ("A_RFM", "bo_duplicate"): 0.992, ("B_RFM_Behavior", "bo_duplicate"): 0.971,
           ("A_RFM", "bo_user_<=2_ngay"): 0.881, ("B_RFM_Behavior", "bo_user_<=2_ngay"): 0.895,
           ("A_RFM", "moc_2022-10-31"): 0.311, ("B_RFM_Behavior", "moc_2022-10-31"): 0.797}
g = lambda m, v: ari.get((m, v))
fmt = lambda x: vn_num(x, 3) if x is not None else "–"

c1, c2, c3 = st.columns(3, gap="large")
with c1:
    extra = (f" Trung vị Avg_Quantity của nhóm là {vn_num(q_niche, 2)}, trong khi cụm cao nhất còn lại là {vn_num(q_other, 2)}."
             if q_niche is not None else "")
    finding(
        CLUSTER_COLORS[2],
        "Đặc trưng hành vi tách thêm một nhóm nhỏ",
        f"Với Model B, nhóm “{CLUSTER_NAMES_B[2]}” gồm {vn_int(n_niche)} người dùng ({vn_pct(share_niche)}), "
        f"nổi bật ở số lượng trung bình mỗi dòng.{extra} Model A chỉ dùng RFM nên chia người dùng thuần theo mức hoạt động.",
    )
with c2:
    finding(
        CLUSTER_COLORS[1],
        "K = 4 ổn định với seed và cách chạy, nhạy với mốc thời gian",
        f"Chạy lại bằng sklearn giữ ARI {fmt(g('A_RFM', 'baseline_sklearn'))} (A) và {fmt(g('B_RFM_Behavior', 'baseline_sklearn'))} (B); "
        f"bỏ dòng trùng giữ {fmt(g('A_RFM', 'bo_duplicate'))} và {fmt(g('B_RFM_Behavior', 'bo_duplicate'))}. "
        f"Nhưng lùi mốc quan sát về 2022-10-31 thì Model A còn {fmt(g('A_RFM', 'moc_2022-10-31'))}, Model B còn {fmt(g('B_RFM_Behavior', 'moc_2022-10-31'))}.",
    )
with c3:
    finding(
        CLUSTER_COLORS[0],
        "Hai mô hình khác nhau, kết quả vững trước lỗi đọc CSV",
        f"ARI giữa A và B chỉ là {vn_num(FIXED['ari_ab'], 3)}, nên thêm đặc trưng hành vi thực sự đổi cách chia. "
        f"Sau khi sửa lỗi đọc CSV (trung vị mất {vn_pct(FIXED['parser_median_pct'], 2)} số dòng mỗi người dùng), "
        f"K = 4 vẫn giữ ARI {vn_num(FIXED['ari_parser']['A_RFM'], 3)} (A) và {vn_num(FIXED['ari_parser']['B_RFM_Behavior'], 3)} (B).",
    )

# ---------------------------------------------------------------- Hạn chế
st.markdown("### Hạn chế")
sweep = read_csv("k_sweep")
sil_note = "Silhouette khoảng 0,5"
if has_cols(sweep, ["model", "k", "silhouette"]):
    s4 = sweep[(sweep["k"] == FIXED["k"]) & sweep["model"].isin(["A_RFM", "B_RFM_Behavior"])]
    if len(s4):
        sil_note += " (tại K = 4: " + ", ".join(
            f"{'A' if m == 'A_RFM' else 'B'} = {vn_num(v, 3)}" for m, v in zip(s4["model"], s4["silhouette"])) + ")"

callout(
    "warning",
    "<ul style='margin:.2rem 0 0 1.1rem;padding:0'>"
    "<li><b>Ảnh chụp tại một thời điểm.</b> Phân khúc tính tại 2022-12-31, không theo dõi sự thay đổi của người dùng theo thời gian.</li>"
    "<li><b>Cụm hoạt động nằm trên dải liên tục.</b> Ranh giới giữa các cụm là tương đối, không phải nhóm tách biệt tự nhiên.</li>"
    f"<li><b>Mẫu không đại diện.</b> Dữ liệu gồm {vn_int(FIXED['survey_ids'])} mã khảo sát (còn {vn_int(FIXED['users'])} sau lọc), "
    "không đại diện cho toàn bộ khách hàng Amazon.</li>"
    f"<li><b>Cấu trúc cụm ở mức vừa phải.</b> {sil_note}.</li>"
    f"<li><b>Quy mô thử nghiệm nhỏ.</b> K-Means chạy trên {vn_int(FIXED['users'])} dòng, một máy; chưa đủ để kết luận khả năng mở rộng.</li>"
    "<li><b>Chỉ mô tả, không nhân quả.</b> Các cụm và lift mô tả hành vi trong dữ liệu, không cho biết nguyên nhân.</li>"
    "</ul>",
    title="Cần ghi nhớ khi dùng kết quả",
)
callout(
    "info",
    "Survey_ResponseID là người dùng ở cấp dữ liệu nghiên cứu. Frequency đếm số ngày có phát sinh mua hàng vì không có mã đơn.",
    title="Quy ước dùng trong ứng dụng",
)
