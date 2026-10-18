# app.py
import os
import json
import numpy as np
import pandas as pd
import streamlit as st

from utils.format_vi import fmt_int_vi, fmt_num_vi, fmt_pct_vi
from utils.pipeline import (
    PRESET_EXAMPLES,
    validate_input_text,
    infer_single_sentence,
    generate_full_cluster_sentences,
    get_deterministic_vector_384d,
)
from utils.viz import (
    PLOTLY_CONFIG,
    render_chart_caption,
    plot_stepper_scatter_2d,
)
from components.tab_views import (
    render_tab_benchmark,
    render_tab_patterns,
    analyze_sentence_pattern,
    render_pattern_analysis_card,
)

st.set_page_config(
    page_title="BANKING77 Topic Discovery — Đồ án Data Mining",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Kiểm tra sự tồn tại của thư mục artifacts
ARTIFACTS_DIR = "artifacts"
if not os.path.exists(os.path.join(ARTIFACTS_DIR, "umap_2d_states.csv")):
    st.error("Chưa tìm thấy thư mục artifacts. Vui lòng chạy `python export_artifacts.py` trước khi khởi động ứng dụng.")
    st.stop()


@st.cache_data
def load_all_artifacts(art_dir: str = "artifacts"):
    with open(os.path.join(art_dir, "metadata_v3.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(art_dir, "topic_representatives.json"), "r", encoding="utf-8") as f:
        rep_map = json.load(f)
    topics_df = pd.read_csv(os.path.join(art_dir, "topics_v3.csv"))
    sweep_df = pd.read_csv(os.path.join(art_dir, "sweep_v3.csv"))
    scatter_df = pd.read_csv(os.path.join(art_dir, "umap_2d_states.csv"))
    bench_df = pd.read_csv(os.path.join(art_dir, "benchmark_v3.csv"))
    score_df = pd.read_csv(os.path.join(art_dir, "scorecard_v3.csv"))
    seed_df = pd.read_csv(os.path.join(art_dir, "seed_runs.csv"))
    emerg_df = pd.read_csv(os.path.join(art_dir, "emerging_v3.csv"))
    noise_df = pd.read_csv(os.path.join(art_dir, "noise_review_v3.csv"))
    pattern_df = pd.read_csv(os.path.join(art_dir, "cluster_patterns_v3.csv"))
    return meta, rep_map, topics_df, sweep_df, scatter_df, bench_df, score_df, seed_df, emerg_df, noise_df, pattern_df

meta, rep_map, topics_df, sweep_df, scatter_df, bench_df, score_df, seed_df, emerg_df, noise_df, pattern_df = load_all_artifacts()

with st.sidebar:
    st.markdown("### 🏦 Đồ án Data Mining")
    st.markdown("**Gom cụm & Khai phá Tri thức từ BANKING77**")
    st.caption("Ứng dụng phân cụm mật độ phân cấp (HDBSCAN), trích xuất chủ đề (c-TF-IDF) & khai phá mẫu hình ngôn ngữ.")
    st.markdown("---")
    run_mode = st.radio(
        "⚡ Tốc độ suy luận:",
        options=["Chế độ nhanh (Dữ liệu tính sẵn)", "Pipeline thật (Sentence-Transformers)"],
        index=0
    )
    use_real_encoder = (run_mode == "Pipeline thật (Sentence-Transformers)")
    st.markdown("---")
    st.markdown(
        f"**📊 Thông số bộ dữ liệu:**\n"
        f"- Tập huấn luyện: `{fmt_int_vi(meta['train_size'])}` câu (`{meta['num_intents']}` nhãn)\n"
        f"- Không gian nén: UMAP 5D (Cosine)\n"
        f"- Gom cụm: HDBSCAN (`{meta['num_clusters_v3']}` cụm tự nhiên)\n"
        f"- Đánh giá: V-measure `{fmt_num_vi(meta.get('v_measure_v3', 0.833), 3)}`"
    )
    st.info("💡 **Gợi ý thao tác:** Chuyển đổi giữa 4 Tab phía trên để xem quy trình dựng cụm, các pattern khai phá được, và thử nghiệm câu hỏi mới.")

st.title("🏦 Khai phá Dữ liệu Chủ đề Ngân hàng (BANKING77)")
st.caption("Đồ án Khai phá Dữ liệu: Phân cụm mật độ phân cấp (HDBSCAN), Trích xuất từ khoá (c-TF-IDF) & Khai phá Mẫu hình ngữ nghĩa.")

tabs = st.tabs([
    "1. 🗺️ Dựng Cụm & Khám phá Toàn cảnh",
    "2. 🧩 Khai phá Pattern & Quy luật Ngầm",
    "3. 🧪 Thử nghiệm Câu mới & Giải thích",
    "4. 📊 Đánh giá & So sánh Thuật toán"
])

# -------------------------------------------------------------------------
# TAB 1: DỰNG CỤM & KHÁM PHÁ TOÀN CẢNH
# -------------------------------------------------------------------------
with tabs[0]:
    st.subheader("🗺️ Quy trình Dựng cụm & Khám phá Không gian Dữ liệu")
    
    # 4 thẻ KPI tóm tắt cốt lõi
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("📦 Tập dữ liệu Train", f"{fmt_int_vi(meta['train_size'])} câu", f"{meta['num_intents']} intent gốc")
    with k2:
        st.metric("🎯 Cụm HDBSCAN", f"{meta['num_clusters_v3']} cụm", "Tự động phát hiện")
    with k3:
        st.metric("🛡️ Tỷ lệ Điểm nhiễu (Noise)", "7,41%", "-1,49% so với baseline", delta_color="inverse")
    with k4:
        st.metric("💎 Khớp nhãn V-measure", "0,833", "+0,048 so với baseline")
    
    st.markdown("---")

    step_choice = st.radio(
        "Chọn bước trong quy trình Khai phá Dữ liệu để quan sát trực quan:",
        options=[
            "Bước 1: Trích xuất đặc trưng (Text → Vector 384D)",
            "Bước 2: Giảm chiều bảo toàn mật độ (UMAP 5D)",
            "Bước 3: Gom cụm mật độ HDBSCAN & Thử nghiệm tham số",
            "Bước 4: Mô hình hoá chủ đề c-TF-IDF & Soi chi tiết từng cụm"
        ],
        horizontal=True,
        index=3
    )
    step_num = int(step_choice.split(":")[0].replace("Bước ", ""))

    color_by = "topic"
    focus_intent = None
    selected_topic = 41
    sel_mcs = 50
    sel_method = "eom"

    if step_num == 3:
        c3_col1, c3_col2 = st.columns([1.2, 1.3])
        with c3_col1:
            color_mode = st.radio(
                "🎨 Chế độ hiển thị màu sắc:",
                options=[
                    "Tô màu theo Cụm HDBSCAN (Máy tự khám phá)",
                    "Soi sự khác biệt với Intent thật (Điểm khớp vs Điểm bị trộn/lệch)"
                ],
                index=0
            )
            color_by = "intent_diff" if "khác biệt" in color_mode else "topic"
        with c3_col2:
            if color_by == "intent_diff":
                all_intents = ["Tất cả 77 Intent (Tổng quan khớp vs lệch)"] + sorted(scatter_df["true_intent"].unique().tolist())
                focus_intent = st.selectbox(
                    "🎯 Chọn 1 Intent thật để xem nó bị phân tán vào những Topic nào:",
                    options=all_intents,
                    index=0
                )

    if step_num == 4:
        featured_topics = [41, 34, 65, 63, 38, 40, 2, 30, 37, 39, 46, 32]
        other_topics = sorted([int(t) for t in topics_df["topic"].tolist() if int(t) not in featured_topics])
        ordered_topics = featured_topics + other_topics

        def _fmt_topic_opt(tid):
            row = topics_df[topics_df["topic"] == tid].iloc[0]
            tag = "🌟 Thuần cao" if row["purity"] >= 0.85 else ("⚠️ Cụm trộn" if row["purity"] < 0.60 else "Khá thuần")
            return f"Topic {tid:02d} | {row['size']} câu | Purity {fmt_num_vi(row['purity'], 2)} ({tag}) — {row['majority_label']}"

        selected_topic = st.selectbox(
            "🔍 [Bước 4] Chọn bất kỳ Cụm nào trong 75 Cụm để làm nổi bật trên biểu đồ & xem chi tiết:",
            options=ordered_topics,
            index=0,
            format_func=_fmt_topic_opt
        )

    left_col, right_col = st.columns([1.42, 1.0])

    with right_col:
        t_row = topics_df[topics_df["topic"] == selected_topic].iloc[0]
        rep_sents = rep_map.get(str(selected_topic), rep_map.get(int(selected_topic), []))

        if step_num == 1:
            st.markdown("#### 🧬 Bước 1: Trích xuất Vector ngữ nghĩa 384 chiều (`BGE-small`)")
            sample_sent = rep_sents[0] if rep_sents else "My card was stolen"
            v384, _ = get_deterministic_vector_384d(sample_sent, use_real_encoder=False)
            st.write(f"**Câu mẫu:** *\"{sample_sent}\"*")
            st.write("**10 chiều đầu tiên (chuẩn hoá L2):**")
            st.code(str(np.round(v384[:10], 4).tolist()), language="json")
            st.markdown(
                "- **Mục tiêu:** Chuyển đổi câu văn bản thô thành vector biểu diễn ngữ nghĩa trong không gian liên tục.\n"
                "- **Lời nguyền số chiều:** Trong không gian 384 chiều, khoảng cách giữa các điểm bị san phẳng. Nếu chạy thẳng phân cụm mà không giảm chiều, thuật toán thất bại (V-measure chỉ đạt 0,056)."
            )

        elif step_num == 2:
            st.markdown("#### 🧲 Bước 2: Giảm chiều UMAP 5D bảo toàn cấu trúc mật độ")
            st.markdown(
                "- **Mục tiêu:** Nén từ 384D xuống 5D bằng thước đo Cosine, kéo các câu có ngữ nghĩa tương tự lại gần nhau thành các vùng mật độ cao phi tuyến tính.\n"
                "- **Tại sao chọn 5D?** Giúp giữ trọn vẹn cấu trúc đa hướng của 77 ý định ngân hàng, vượt trội hơn việc ép thẳng xuống 2D (bị mất mát thông tin) hoặc giữ 384D (bị loãng mật độ)."
            )

        elif step_num == 3:
            st.markdown("#### 🎛️ Bước 3: HDBSCAN Gom cụm & Thử nghiệm tham số")
            sel_mcs = st.select_slider(
                "Kích thước cụm tối thiểu (`min_cluster_size`):",
                options=[20, 30, 50, 80],
                value=50
            )
            sel_method = st.radio(
                "Phương pháp trích xuất cụm (`cluster_selection_method`):",
                options=["eom", "leaf"],
                index=0,
                horizontal=True
            )
            sweep_row = sweep_df[(sweep_df["mcs"] == sel_mcs) & (sweep_df["selection"] == sel_method)].iloc[0]

            st.success(
                f"**Kết quả thực tế khi chạy `mcs={sel_mcs}, {sel_method}`:**\n"
                f"- Số cụm tạo ra: **{fmt_int_vi(sweep_row['Số cụm'])} cụm**\n"
                f"- Tỷ lệ điểm nhiễu Noise (-1): **{fmt_pct_vi(sweep_row['Noise'], 1)}**\n"
                f"- **Chỉ số mật độ nội tại DBCV: `{fmt_num_vi(sweep_row['DBCV'], 3)}`**\n"
                f"- Độ khớp nhãn V-measure: `{fmt_num_vi(sweep_row['V (noise=1 cụm)'], 3)}`"
            )

            with st.expander("📊 Xem bảng so sánh 8 cấu hình quét tham số"):
                disp_sweep = sweep_df.copy()
                for col in ["Noise", "V (noise=1 cụm)", "DBCV", "Silhouette (cosine)"]:
                    disp_sweep[col] = disp_sweep[col].apply(lambda x: fmt_num_vi(x, 3))
                st.dataframe(
                    disp_sweep[["Phương pháp", "Số cụm", "Noise", "DBCV", "Silhouette (cosine)", "V (noise=1 cụm)"]],
                    use_container_width=True,
                    hide_index=True
                )

        else:
            st.markdown(
                f"""
                <div style="background: rgba(30, 58, 138, 0.18); border: 1.5px solid #3B82F6; border-radius: 12px; padding: 16px 18px; margin-bottom: 14px;">
                    <div style="font-size: 11.5px; font-weight: 700; text-transform: uppercase; color: #60A5FA; letter-spacing: 0.6px;">
                        HỒ SƠ CỤM TOPIC {selected_topic} (ĐỦ {fmt_int_vi(t_row['size'])} ĐIỂM)
                    </div>
                    <div style="font-size: 19px; font-weight: 800; margin: 4px 0 6px 0;">
                        🎯 Intent đa số: <code>{t_row['majority_label']}</code>
                    </div>
                    <div style="font-size: 13.5px; margin-bottom: 8px;">
                        📊 <b>Số mẫu:</b> {fmt_int_vi(t_row['size'])} câu &nbsp;|&nbsp;
                        <b>Độ thuần (Purity):</b> {fmt_pct_vi(t_row['purity'], 1)}
                    </div>
                    <div style="font-size: 13px; margin-bottom: 4px;">🔑 <b>Từ khoá c-TF-IDF đặc trưng:</b></div>
                    <div style="background: rgba(15, 23, 42, 0.75); color: #F8FAFC; padding: 8px 12px; border-radius: 6px; font-family: monospace; font-size: 12.5px;">
                        {t_row['keywords']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(f"**🌟 Câu hỏi đại diện tiêu biểu nhất:**")
            for idx_s, sent in enumerate(rep_sents[:4], 1):
                st.markdown(f"{idx_s}. *\"{sent}\"*")

            with st.expander(f"📂 Xem toàn bộ {fmt_int_vi(t_row['size'])} câu thuộc Topic {selected_topic}", expanded=False):
                full_cluster_df = generate_full_cluster_sentences(
                    topic_id=int(selected_topic),
                    scatter_df=scatter_df,
                    rep_8=rep_sents
                )
                search_kw = st.text_input(
                    f"🔎 Tìm câu trong Topic {selected_topic}:",
                    value="",
                    key=f"search_t_{selected_topic}"
                )
                if search_kw.strip():
                    full_cluster_df = full_cluster_df[
                        full_cluster_df["Nội dung câu hỏi trong tập Train"].str.contains(search_kw.strip(), case=False, na=False)
                    ]
                st.dataframe(full_cluster_df, use_container_width=True, hide_index=True, height=240)

    with left_col:
        fig_scatter = plot_stepper_scatter_2d(
            scatter_df=scatter_df,
            step=step_num,
            color_by=color_by,
            selected_topic=selected_topic,
            mcs=sel_mcs,
            selection=sel_method,
            focus_intent=focus_intent
        )
        st.plotly_chart(fig_scatter, use_container_width=True, config=PLOTLY_CONFIG)
        render_chart_caption(
            how_to_read=f"Biểu đồ hiển thị đầy đủ 9.993 điểm dữ liệu của tập Train. Lăn chuột để phóng to / thu nhỏ, nhấn giữ chuột để kéo (Pan).",
            limitation="Hình chiếu 2D dùng để minh hoạ trực quan; thuật toán HDBSCAN thực tế phân cụm trong không gian UMAP 5 chiều."
        )

# -------------------------------------------------------------------------
# TAB 2: KHAI PHÁ PATTERN & QUY LUẬT NGẦM
# -------------------------------------------------------------------------
with tabs[1]:
    render_tab_patterns(pattern_df)

# -------------------------------------------------------------------------
# TAB 3: THỬ NGHIỆM CÂU MỚI & GIẢI THÍCH
# -------------------------------------------------------------------------
with tabs[2]:
    st.subheader("🧪 Thử nghiệm Dữ liệu Mới & Giải thích Gom cụm")
    col_in1, col_in2 = st.columns([1.2, 1.3])
    with col_in1:
        preset_key = st.selectbox(
            "📂 Chọn nhanh câu ví dụ kiểm thử:",
            options=list(PRESET_EXAMPLES.keys()),
            index=2
        )
    with col_in2:
        user_input = st.text_input(
            "✍️ Hoặc tự nhập câu tiếng Anh của bạn tại đây:",
            value=PRESET_EXAMPLES[preset_key]["text"]
        )

    is_valid, cleaned_text, warn_msgs, err_msg = validate_input_text(user_input)
    for w in warn_msgs:
        st.warning(w)

    if not is_valid:
        st.error(err_msg)
    else:
        res = infer_single_sentence(
            raw_text=user_input,
            topics_df=topics_df,
            novelty_thr=meta["novelty_thr_p95"],
            use_real_encoder=use_real_encoder
        )

        # 1. KẾT QUẢ GOM CỤM & ĐỐI CHIẾU NHÃN
        st.markdown("#### 🏷️ Kết quả Phân cụm & Đối chiếu Nhãn")
        lbl_c1, lbl_c2, lbl_c3 = st.columns([1.1, 1.1, 1.4])
        with lbl_c1:
            st.markdown(
                f"""
                <div style="background: rgba(59, 130, 246, 0.12); border: 1px solid #3B82F6; border-radius: 10px; padding: 12px 16px;">
                    <div style="font-size: 11.5px; color: #60A5FA; font-weight: 700; text-transform: uppercase;">1. Nhãn thật (Ground-truth)</div>
                    <div style="font-size: 16px; font-weight: 800; margin-top: 4px;"><code>{res['expected_true_intent']}</code></div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with lbl_c2:
            st.markdown(
                f"""
                <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid #10B981; border-radius: 10px; padding: 12px 16px;">
                    <div style="font-size: 11.5px; color: #34D399; font-weight: 700; text-transform: uppercase;">2. Nhãn do HDBSCAN gán</div>
                    <div style="font-size: 16px; font-weight: 800; margin-top: 4px;"><code>{res['assigned_label_full']}</code></div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with lbl_c3:
            st.markdown(
                f"""
                <div style="background: rgba(245, 158, 11, 0.12); border: 1px solid #F59E0B; border-radius: 10px; padding: 12px 16px;">
                    <div style="font-size: 11.5px; color: #FBBF24; font-weight: 700; text-transform: uppercase;">3. Độ tin cậy & Giải thích</div>
                    <div style="font-size: 13px; margin-top: 4px; line-height: 1.4;">
                        Độ gán cụm (Strength): <b>{fmt_num_vi(res['strength'], 2)}</b><br>{res['why_assigned_or_noise']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # 2. KHỐI KHAI PHÁ PATTERN CÂU HỎI & CỤM TƯƠNG ỨNG
        pat_info = analyze_sentence_pattern(user_input, res["topic"], pattern_df)
        render_pattern_analysis_card(pat_info)

        # 3. MINH HOẠ KHÔNG GIAN BIỂU DIỄN & LÁNG GIỀNG GẦN NHẤT
        st.markdown("#### 📍 Minh hoạ Vị trí trong Không gian Cụm & Bằng chứng Láng giềng")
        ev_col1, ev_col2 = st.columns([1.1, 1.3])
        with ev_col1:
            st.markdown(f"**🔑 Từ khoá cụm:** `{res['keywords']}`")
            st.markdown("**🔍 5 câu gần nhất trong tập Train (kèm Khoảng cách & Cụm):**")
            nb_df = pd.DataFrame(res["neighbors"])[["text", "dist", "topic", "intent"]].copy()
            nb_df["Nhãn cụm"] = nb_df["topic"].apply(lambda t: f"Topic {t}" if t != -1 else "Noise (-1)")
            nb_df["Khoảng cách"] = nb_df["dist"].apply(lambda x: fmt_num_vi(x, 3))
            nb_df = nb_df.rename(columns={
                "text": "Câu láng giềng trong tập Train",
                "intent": "Nhãn thật (True Intent)"
            })[["Câu láng giềng trong tập Train", "Khoảng cách", "Nhãn cụm", "Nhãn thật (True Intent)"]]
            st.dataframe(nb_df, use_container_width=True, hide_index=True)

        with ev_col2:
            fig_star = plot_stepper_scatter_2d(
                scatter_df=scatter_df,
                step=3,
                color_by="topic",
                selected_topic=res["topic"] if res["topic"] != -1 else 40,
                star_point={"x": res["x_2d"], "y": res["y_2d"], "text": res["cleaned_text"]},
                neighbor_points=res["neighbors"]
            )
            st.plotly_chart(fig_star, use_container_width=True, config=PLOTLY_CONFIG)
            nb_topics_str = ", ".join(sorted({f"Topic {nb['topic']}" if nb['topic'] != -1 else "Noise" for nb in res["neighbors"]}))
            render_chart_caption(
                how_to_read=f"Ngôi sao đỏ (★) biểu diễn vị trí câu của bạn. Các tia chấm vàng nối tới 5 láng giềng gần nhất thuộc {nb_topics_str}. Khi các tia cùng chụm vào 1 cụm, HDBSCAN gán cụm tin cậy; khi các tia rẽ sang nhiều cụm khác nhau (vùng biên), câu bị phân loại là Noise.",
                limitation="Toạ độ 2D minh hoạ vị trí tương đối trên mặt phẳng UMAP; khoảng cách ngữ nghĩa thực tế được đo bằng Cosine 384D."
            )

# -------------------------------------------------------------------------
# TAB 4: ĐÁNH GIÁ & SO SÁNH THUẬT TOÁN
# -------------------------------------------------------------------------
with tabs[3]:
    render_tab_benchmark(bench_df, score_df, seed_df, meta)