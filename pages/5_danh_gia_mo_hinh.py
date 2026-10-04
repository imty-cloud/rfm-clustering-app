import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from utils import (CLUSTER_COLORS, FIXED, MODEL_SHORT, MODELS, callout, chart_title, cluster_label, download_csv,
                   has_cols, how_to_read, kpi, load_clusters, page_header, pick, read_csv, show, source,
                   static_figures, table, vn_int, vn_num, warn_missing)

page_header(
    "Đánh giá mô hình",
    "Năm góc nhìn để kiểm tra K = 4 có đáng tin hay không: chỉ số nội tại, độ ổn định, độ bền, độ nhạy thời gian và sự khác biệt giữa hai mô hình.",
)

MODEL_COLORS = {"A_RFM": CLUSTER_COLORS[0], "B_RFM_Behavior": CLUSTER_COLORS[2]}
K_FINAL = FIXED["k"]

tab_k, tab_stab, tab_rob, tab_win, tab_ab = st.tabs(
    ["Chọn K", "Stability", "Robustness", "Độ nhạy cửa sổ", "So sánh A và B"]
)

# ======================================================================================
# 1. CHỌN K
# ======================================================================================
with tab_k:
    sweep = read_csv("k_sweep")
    needed = ["model", "k", "silhouette", "davies_bouldin", "calinski_harabasz", "max_share"]
    if warn_missing("k_sweep"):
        pass
    elif not has_cols(sweep, needed):
        st.warning("Tệp k_sweep.csv thiếu cột: " + ", ".join(c for c in needed if c not in sweep.columns))
    else:
        sw = sweep[sweep["model"].isin(MODELS.keys())].copy()  # chỉ A_RFM và B_RFM_Behavior
        at4 = sw[sw["k"] == K_FINAL].set_index("model")
        cols = st.columns(len(MODELS))
        for col, m in zip(cols, MODELS):
            with col:
                if m in at4.index:
                    kpi(f"Silhouette tại K = {K_FINAL}, {MODELS[m]}", vn_num(at4.loc[m, "silhouette"], 3),
                        f"Davies-Bouldin {vn_num(at4.loc[m, 'davies_bouldin'], 2)}, cụm lớn nhất {vn_num(at4.loc[m, 'max_share'] * 100 if at4.loc[m, 'max_share'] <= 1 else at4.loc[m, 'max_share'], 1)}%")
        chart_title(
            f"Bốn chỉ số theo K: K = {K_FINAL} là mức được chọn để profiling",
            "Đường đứt đánh dấu K = 4",
        )
        metrics = [("silhouette", "Silhouette (cao tốt)"), ("davies_bouldin", "Davies-Bouldin (thấp tốt)"),
                   ("calinski_harabasz", "Calinski-Harabasz (cao tốt)"), ("max_share", "Tỷ trọng cụm lớn nhất")]
        fig = make_subplots(rows=2, cols=2, subplot_titles=[t for _, t in metrics], vertical_spacing=0.2, horizontal_spacing=0.1)
        for i, (col_name, _) in enumerate(metrics):
            r, c = i // 2 + 1, i % 2 + 1
            for m in MODELS:
                s = sw[sw["model"] == m].sort_values("k")
                fig.add_trace(go.Scatter(
                    x=s["k"], y=s[col_name], mode="lines+markers", name=MODELS[m], legendgroup=m,
                    showlegend=(i == 0), line=dict(color=MODEL_COLORS[m], width=2.5), marker=dict(size=7),
                    hovertemplate=f"{MODELS[m]}<br>K = %{{x}}<br>%{{y:,.3f}}<extra></extra>",
                ), row=r, col=c)
            fig.add_vline(x=K_FINAL, line_dash="dash", line_color="rgba(128,128,128,0.7)", row=r, col=c)
            fig.update_xaxes(title_text="Số cụm K", dtick=1, row=r, col=c)
        fig.update_yaxes(title_text="Giá trị chỉ số")
        show(fig, key="ksweep", height=560)
        source()
        how_to_read("Silhouette và Calinski-Harabasz càng cao càng tốt, Davies-Bouldin càng thấp càng tốt. "
                    "Tỷ trọng cụm lớn nhất tăng cao nghĩa là một cụm đang nuốt phần lớn dữ liệu.")
        download_csv(sw, "k_sweep_A_B.csv", key="dl_ksweep")
    static_figures(["fig06"], key="fig_k")

# ======================================================================================
# 2. STABILITY
# ======================================================================================
with tab_stab:
    stab = read_csv("stability")
    need = ["model", "k", "seed_ari_mean", "seed_ari_min", "boot_ari_mean", "boot_ari_min"]
    if warn_missing("stability"):
        pass
    elif not has_cols(stab, need):
        st.warning("Tệp stability.csv thiếu cột: " + ", ".join(c for c in need if c not in stab.columns))
    else:
        st_ = stab[stab["model"].isin(MODELS.keys())]
        at4 = st_[st_["k"] == K_FINAL].set_index("model")
        cols = st.columns(len(MODELS))
        for col, m in zip(cols, MODELS):
            with col:
                if m in at4.index:
                    kpi(f"{MODELS[m]} tại K = {K_FINAL}",
                        f"{vn_num(at4.loc[m, 'seed_ari_mean'], 3)} / {vn_num(at4.loc[m, 'boot_ari_mean'], 3)}",
                        "ARI trung bình: đổi seed / mẫu con 80%")
        both = all(m in at4.index for m in MODELS)
        seed_higher = both and all(at4.loc[m, "seed_ari_mean"] >= at4.loc[m, "boot_ari_mean"] for m in MODELS)
        chart_title(
            f"Tại K = {K_FINAL}, đổi seed ổn định hơn đổi mẫu dữ liệu" if seed_higher
            else f"Độ ổn định theo seed và theo mẫu con tại K = {K_FINAL}",
            "ARI trung bình (đường liền) và nhỏ nhất (đường đứt) theo K",
        )
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Đổi seed (Spark, cùng dữ liệu)", "Đổi mẫu con 80% (sklearn)"],
                            horizontal_spacing=0.1)
        for ci, (mean_c, min_c) in enumerate([("seed_ari_mean", "seed_ari_min"), ("boot_ari_mean", "boot_ari_min")], start=1):
            for m in MODELS:
                s = st_[st_["model"] == m].sort_values("k")
                fig.add_trace(go.Scatter(x=s["k"], y=s[mean_c], mode="lines+markers", name=MODELS[m], legendgroup=m,
                                         showlegend=(ci == 1), line=dict(color=MODEL_COLORS[m], width=2.5),
                                         hovertemplate=f"{MODELS[m]}<br>K = %{{x}}<br>ARI trung bình %{{y:,.3f}}<extra></extra>"),
                              row=1, col=ci)
                fig.add_trace(go.Scatter(x=s["k"], y=s[min_c], mode="lines", name=f"{MODELS[m]} (nhỏ nhất)", legendgroup=m,
                                         showlegend=False, line=dict(color=MODEL_COLORS[m], width=1.5, dash="dot"),
                                         hovertemplate=f"{MODELS[m]}<br>K = %{{x}}<br>ARI nhỏ nhất %{{y:,.3f}}<extra></extra>"),
                              row=1, col=ci)
            fig.add_vline(x=K_FINAL, line_dash="dash", line_color="rgba(128,128,128,0.6)", row=1, col=ci)
            fig.update_xaxes(title_text="Số cụm K", dtick=1, row=1, col=ci)
        fig.update_yaxes(title_text="ARI (1 là giống hệt)", range=[0, 1.02])
        show(fig, key="stability", height=400)
        source()
        callout("info",
                "<b>Seed</b>: giữ nguyên dữ liệu, chỉ đổi điểm khởi tạo K-Means rồi so ARI giữa các lần chạy, nên chỉ đo độ nhạy với khởi tạo. "
                "<b>Mẫu con</b>: chạy lại trên 80% người dùng rồi gán nhãn cho toàn bộ, nên còn đo độ nhạy với dữ liệu. "
                "Vì nguồn biến động thứ hai lớn hơn, ARI theo mẫu con thường thấp hơn ARI theo seed.",
                title="Vì sao hai biểu đồ khác nhau")
        how_to_read("ARI gần 1 nghĩa là các lần chạy chia người dùng gần như y hệt. Đường đứt là lần chạy tệ nhất, cho biết mức rủi ro xấu nhất.")
        download_csv(st_, "stability_A_B.csv", key="dl_stab")

    summ = read_csv("final_summary")
    if has_cols(summ, ["model"]):
        st.markdown(f"#### Nghiệm cuối cùng: tốt nhất trong {FIXED['n_seeds']} seed theo WSSSE")
        s2 = summ.copy()
        s2["model"] = s2["model"].map(MODELS).fillna(s2["model"])
        s2 = s2.rename(columns={"model": "Mô hình", "best_seed": "Seed tốt nhất", "wssse": "WSSSE",
                                "seeds_ari_gt_0_9": f"Số seed có ARI > 0,9 (trên {FIXED['n_seeds']})",
                                "niche_found": "Số seed có nhóm nhỏ Avg_Quantity cao"})
        table(s2, column_config={"WSSSE": st.column_config.NumberColumn(format="%.1f")})
        source()
    static_figures(["fig07"], key="fig_stab")

# ======================================================================================
# 3. ROBUSTNESS
# ======================================================================================
with tab_rob:
    rob = read_csv("robustness")
    need = ["model", "variant", "ARI_vs_final"]
    if warn_missing("robustness"):
        pass
    elif not has_cols(rob, need):
        st.warning("Tệp robustness.csv thiếu cột: " + ", ".join(c for c in need if c not in rob.columns))
    else:
        VAR_ORDER = ["baseline_sklearn", "bo_duplicate", "bo_user_<=2_ngay", "moc_2022-10-31"]
        VAR_VI = {"baseline_sklearn": "Chạy lại bằng sklearn", "bo_duplicate": "Bỏ dòng trùng",
                  "bo_user_<=2_ngay": "Bỏ người dùng có ≤ 2 ngày mua", "moc_2022-10-31": "Đổi mốc quan sát sang 2022-10-31"}
        rb = rob[rob["model"].isin(MODELS.keys())].copy()
        piv = rb.pivot_table(index="variant", columns="model", values="ARI_vs_final")
        order = [v for v in VAR_ORDER if v in piv.index] + [v for v in piv.index if v not in VAR_ORDER]
        piv = piv.loc[order]
        worst_v, worst_m = piv.stack().idxmin()
        worst = float(piv.stack().min())
        chart_title(
            f"Điểm yếu nhất: {MODELS[worst_m]} khi {VAR_VI.get(worst_v, worst_v).lower()} (ARI {vn_num(worst, 3)})",
            "ARI giữa nhãn cuối cùng và nhãn sau mỗi thay đổi dữ liệu hoặc cách chạy. Vạch đứt là ngưỡng tham chiếu 0,85 của đồ án",
        )
        fig = go.Figure()
        for m in MODELS:
            if m not in piv.columns:
                continue
            fig.add_trace(go.Bar(
                y=[VAR_VI.get(v, v) for v in piv.index], x=piv[m], orientation="h", name=MODELS[m],
                marker_color=MODEL_COLORS[m], text=[vn_num(v, 3) for v in piv[m]], textposition="outside",
                hovertemplate=f"{MODELS[m]}<br>%{{y}}<br>ARI %{{x:,.3f}}<extra></extra>",
            ))
        fig.add_vline(x=0.85, line_dash="dash", line_color="rgba(128,128,128,0.9)", annotation_text="0,85",
                      annotation_position="top")
        fig.update_xaxes(title="ARI so với nghiệm cuối (K = 4)", range=[0, 1.12])
        fig.update_yaxes(title="Kịch bản kiểm tra", autorange="reversed")
        fig.update_layout(barmode="group")
        show(fig, key="robust", height=420)
        source()
        how_to_read("Thanh dài là kết quả giữ nguyên khi thay đổi điều kiện. Thanh ngắn hơn vạch 0,85 là điểm cần thận trọng khi diễn giải cụm.")
        if "moc_2022-10-31" in piv.index and has_cols(piv.reset_index(), list(MODELS)):
            a_v, b_v = piv.loc["moc_2022-10-31", "A_RFM"], piv.loc["moc_2022-10-31", "B_RFM_Behavior"]
            if a_v < 0.85:
                callout("warning", f"Khi lùi mốc quan sát về 2022-10-31, ARI của Model A chỉ còn {vn_num(a_v, 3)} "
                                   f"(Model B: {vn_num(b_v, 3)}). Cụm của Model A phụ thuộc nhiều vào thời điểm chụp dữ liệu.",
                        title="Nhạy với mốc thời gian")
        callout("insight",
                f"Sau khi sửa lỗi đọc CSV (39.479 dòng ID sai), K = 4 vẫn giữ ARI {vn_num(FIXED['ari_parser']['A_RFM'], 3)} (A) và "
                f"{vn_num(FIXED['ari_parser']['B_RFM_Behavior'], 3)} (B). Chỉ K = 4 được kiểm tra lại.",
                title="Kiểm tra sau khi sửa parser")
        show_tbl = piv.reset_index().rename(columns={"variant": "Kịch bản", **MODELS})
        show_tbl["Kịch bản"] = show_tbl["Kịch bản"].map(lambda v: VAR_VI.get(v, v))
        download_csv(show_tbl, "robustness_ARI.csv", key="dl_rob")
    static_figures(["fig11"], key="fig_rob")

# ======================================================================================
# 4. ĐỘ NHẠY CỬA SỔ
# ======================================================================================
with tab_win:
    win = read_csv("window")
    if warn_missing("window"):
        pass
    else:
        kcol = "k" if "k" in win.columns else win.columns[0]
        mcols = [m for m in MODELS if m in win.columns]
        if not mcols:
            st.warning("Tệp window_sensitivity.csv không có cột A_RFM / B_RFM_Behavior.")
        else:
            w = win.sort_values(kcol)
            at4 = w[w[kcol] == K_FINAL]
            title_bits = ", ".join(f"{MODEL_SHORT[m]} = {vn_num(float(at4[m].iloc[0]), 3)}" for m in mcols) if len(at4) else ""
            chart_title(
                f"Ở K = {K_FINAL}, ARI giữa hai mốc quan sát là {title_bits}" if title_bits else "ARI giữa mốc 2022-12-31 và 2022-10-31 theo K",
                "So sánh phân cụm ở mốc 2022-12-31 với mốc 2022-10-31, cùng K",
            )
            fig = go.Figure()
            for m in mcols:
                fig.add_trace(go.Scatter(x=w[kcol], y=w[m], mode="lines+markers", name=MODELS[m],
                                         line=dict(color=MODEL_COLORS[m], width=2.5), marker=dict(size=8),
                                         hovertemplate=f"{MODELS[m]}<br>K = %{{x}}<br>ARI %{{y:,.3f}}<extra></extra>"))
            fig.add_hline(y=0.85, line_dash="dash", line_color="rgba(128,128,128,0.9)", annotation_text="0,85",
                          annotation_position="top left")
            fig.add_vline(x=K_FINAL, line_dash="dot", line_color="rgba(128,128,128,0.6)")
            fig.update_xaxes(title="Số cụm K", dtick=1)
            fig.update_yaxes(title="ARI giữa hai mốc quan sát", range=[0, 1.02])
            show(fig, key="window", height=400)
            source()
            how_to_read("Đường thấp nghĩa là chỉ cần đổi mốc quan sát hai tháng là nhãn cụm đã khác đi nhiều. Đường đứt xám là ngưỡng tham chiếu 0,85.")
            download_csv(w, "window_sensitivity.csv", key="dl_win")
    static_figures(["fig13"], key="fig_win")

# ======================================================================================
# 5. SO SÁNH A VÀ B
# ======================================================================================
with tab_ab:
    dfa, dfb = load_clusters("A_RFM"), load_clusters("B_RFM_Behavior")
    if warn_missing("clusters_A_RFM", "clusters_B_RFM_Behavior") or dfa is None or dfb is None:
        pass
    else:
        mg = dfa[["Survey_ResponseID", "cluster"]].merge(
            dfb[["Survey_ResponseID", "cluster"]], on="Survey_ResponseID", suffixes=("_A", "_B"))
        ct = pd.crosstab(mg["cluster_A"], mg["cluster_B"])
        ari4 = FIXED["ari_ab"]
        ari_df = read_csv("ari_models")
        if has_cols(ari_df, ["A_vs_B"]):
            kc = "k" if "k" in ari_df.columns else ari_df.columns[0]
            hit = ari_df[ari_df[kc] == K_FINAL]
            if len(hit):
                ari4 = float(hit["A_vs_B"].iloc[0])

        mode = pick("Giá trị hiển thị", {"n": "Số người dùng", "row": "% theo hàng (Model A)"}, key="ab_mode", default="n")
        z = ct.values if mode == "n" else ct.div(ct.sum(axis=1), axis=0).values * 100
        txt = [[vn_int(v) if mode == "n" else vn_num(v, 1) + "%" for v in row] for row in z]
        top_share = float(ct.max(axis=1).sum() / ct.values.sum() * 100)
        chart_title(
            f"ARI giữa A và B chỉ là {vn_num(ari4, 3)}: hai mô hình chia người dùng khác nhau",
            f"Mỗi hàng là một cụm Model A, mỗi cột là một cụm Model B. {vn_num(top_share, 1)}% người dùng thuộc cặp (A, B) trùng nhiều nhất của hàng đó",
        )
        fig = go.Figure(go.Heatmap(
            z=z, x=[f"B{c}: {cluster_label('B_RFM_Behavior', int(c), with_id=False)}" for c in ct.columns],
            y=[f"A{c}" for c in ct.index], text=txt, texttemplate="%{text}", textfont=dict(size=14),
            colorscale="Blues", colorbar=dict(title="Người dùng" if mode == "n" else "%", thickness=12),
            hovertemplate="%{y} và %{x}<br>%{text}<extra></extra>",
        ))
        fig.update_xaxes(title="Cụm của Model B", side="bottom", tickangle=-20)
        fig.update_yaxes(title="Cụm của Model A", autorange="reversed", type="category")
        show(fig, key="ab_cross", height=420, legend=False)
        source()
        how_to_read("Nếu hai mô hình giống nhau, mỗi hàng chỉ sáng ở một ô. Ở đây một số hàng của A dàn sang nhiều cột của B, "
                    "nghĩa là đặc trưng hành vi làm thay đổi cách chia người dùng.")
        if has_cols(ari_df, ["A_vs_B"]):
            kc = "k" if "k" in ari_df.columns else ari_df.columns[0]
            a = ari_df.sort_values(kc)
            with st.expander("ARI giữa A và B theo K", icon=":material/show_chart:"):
                fig2 = go.Figure(go.Scatter(x=a[kc], y=a["A_vs_B"], mode="lines+markers", line=dict(color=CLUSTER_COLORS[3], width=2.5),
                                            hovertemplate="K = %{x}<br>ARI %{y:,.3f}<extra></extra>"))
                fig2.add_vline(x=K_FINAL, line_dash="dot", line_color="rgba(128,128,128,0.6)")
                fig2.update_xaxes(title="Số cụm K", dtick=1)
                fig2.update_yaxes(title="ARI giữa Model A và Model B", range=[0, 1.02])
                show(fig2, key="ab_k", height=320, legend=False)
                source()
        download_csv(ct.reset_index().rename(columns={"cluster_A": "cluster_A \\ cluster_B"}), "ma_tran_cheo_A_vs_B.csv", key="dl_ab")
    static_figures(["fig10"], key="fig_ab")
