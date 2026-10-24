# utils/viz.py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.format_vi import fmt_num_vi

# BẬT Zoom In (+), Zoom Out (-), Kéo Pan và Lăn chuột phóng to; chỉ xoá Box Select & Lasso Select gây khung đen
PLOTLY_CONFIG = {
    "displayModeBar": True,
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]
}

def render_chart_caption(how_to_read: str, limitation: str):
    """Hiển thị cố định 1 câu Cách đọc + 1 câu Lưu ý giới hạn ngay dưới mỗi biểu đồ."""
    st.markdown(
        f"""
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-left: 4px solid #0284C7; padding: 12px 16px; border-radius: 6px; font-size: 13.5px; margin-top: 6px; margin-bottom: 16px; color: #1E293B;">
            <div style="margin-bottom: 5px; line-height: 1.45;">📖 <strong>Cách đọc:</strong> {how_to_read}</div>
            <div style="color: #DC2626; line-height: 1.45;">⚠️ <strong>Lưu ý giới hạn:</strong> {limitation}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def plot_stepper_scatter_2d(
    scatter_df: pd.DataFrame,
    step: int = 3,
    color_by: str = "topic",
    selected_topic: int = 41,
    mcs: int = 50,
    selection: str = "eom",
    focus_intent: str = None,
    star_point: dict = None,
    neighbor_points: list = None
) -> go.Figure:
    """
    Biểu đồ Scatter 2D tương tác toàn diện:
    - Hỗ trợ Zoom In/Out (+ / - / lăn chuột) và Kéo trượt (Pan).
    - Ở Bước 3: Thay đổi trực tiếp số cụm & mật độ điểm Noise khi người dùng chỉnh mcs (20/30/50/80) và eom/leaf.
    - Khi chọn chế độ 'intent_diff': Làm nổi bật sự khác biệt giữa Cụm HDBSCAN và Nhãn Intent thật (Điểm khớp chuẩn vs Điểm bị trộn/gom nhầm vs Điểm bị bỏ sót thành Noise).
    """
    df = scatter_df.copy()
    x_col = "x_b1" if step == 1 else "x_umap"
    y_col = "y_b1" if step == 1 else "y_umap"

    # Lấy cột phân cụm tương ứng với tham số mcs & selection ở Bước 3
    dyn_col = f"topic_{mcs}_{selection}"
    if step == 3 and dyn_col in df.columns:
        active_topic_col = dyn_col
    else:
        active_topic_col = "topic"

    fig = go.Figure()
    palette = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24 + px.colors.qualitative.Light24

    if step in (1, 2):
        fig.add_trace(go.Scattergl(
            x=df[x_col],
            y=df[y_col],
            mode="markers",
            name="Câu hỏi khách hàng (9.993 mẫu)",
            marker=dict(size=6.5, color="#60A5FA", opacity=0.70),
            text=df["text"],
            hovertemplate="<b>Câu:</b> %{text}<br>Toạ độ 2D: (%{x:.2f}, %{y:.2f})<extra></extra>"
        ))
    else:
        is_noise_mask = (df[active_topic_col] == -1)
        noise_df = df[is_noise_mask]
        clustered_df = df[~is_noise_mask]

        if color_by == "intent_diff":
            # CHẾ ĐỘ ĐỐI CHIẾU SỰ KHÁC BIỆT GIỮA INTENT THẬT VÀ HDBSCAN (Khắc phục Hình 3)
            if focus_intent and focus_intent != "Tất cả 77 Intent (Tổng quan khớp vs lệch)":
                # Chế độ 3B: Soi 1 Intent thật cụ thể xem bị HDBSCAN xé lẻ vào những Topic nào
                in_intent = df[df["true_intent"] == focus_intent]
                other_pts = df[df["true_intent"] != focus_intent]

                fig.add_trace(go.Scattergl(
                    x=other_pts[x_col], y=other_pts[y_col],
                    mode="markers",
                    name="Các Intent khác (Làm mờ)",
                    marker=dict(size=5, color="#475569", opacity=0.18),
                    hoverinfo="skip"
                ))

                for tid_val, sub in in_intent.groupby(active_topic_col):
                    tid_int = int(tid_val)
                    lbl = f"Bị đẩy thành Noise (-1) [{len(sub)} mẫu]" if tid_int == -1 else f"Nằm ở Topic {tid_int:02d} [{len(sub)} mẫu]"
                    sym = "x" if tid_int == -1 else "diamond"
                    clr = "#EF4444" if tid_int == -1 else palette[tid_int % len(palette)]
                    sub_txt = sub["text"].str.slice(0, 85)
                    sub_ti = sub["true_intent"].fillna("").astype(str).str.slice(0, 35)
                    sub_tl = sub["topic_label"].fillna("").astype(str).str.slice(0, 35)
                    sub_maj = sub["majority_intent_of_cluster"].fillna("").astype(str).str.slice(0, 35)
                    fig.add_trace(go.Scattergl(
                        x=sub[x_col], y=sub[y_col],
                        mode="markers",
                        name=lbl,
                        marker=dict(size=11, color=clr, symbol=sym, opacity=1.0, line=dict(width=1.5, color="#FFFFFF")),
                        customdata=np.stack([sub_txt, sub_ti, sub_tl, sub_maj], axis=-1),
                        hovertemplate=(
                            "<b>Intent thật đang soi:</b> %{customdata[1]}<br>"
                            "<b>HDBSCAN gán vào:</b> %{customdata[2]} (Đa số cụm: %{customdata[3]})<br>"
                            "<b>Câu:</b> %{customdata[0]}<extra></extra>"
                        )
                    ))
            else:
                # Chế độ 3A: Tổng quan toàn bộ sự khác biệt (Khớp chuẩn vs Bị gom trộn lệch intent vs Bị đẩy thành Noise)
                matched_df = clustered_df[clustered_df["is_intent_match"] == True]
                mismatched_df = clustered_df[clustered_df["is_intent_match"] == False]

                # 1. Điểm khớp chuẩn (Màu xanh ngọc dịu, hơi mờ để làm nền)
                m_txt = matched_df["text"].str.slice(0, 85)
                m_tl = matched_df["topic_label"].fillna("").astype(str).str.slice(0, 35)
                m_ti = matched_df["true_intent"].fillna("").astype(str).str.slice(0, 35)
                fig.add_trace(go.Scattergl(
                    x=matched_df[x_col], y=matched_df[y_col],
                    mode="markers",
                    name="✅ Khớp chuẩn (Nhãn thật = Đa số cụm HDBSCAN) [78,6%]",
                    marker=dict(size=5.5, color="#10B981", opacity=0.38),
                    customdata=np.stack([m_txt, m_tl, m_ti], axis=-1),
                    hovertemplate=(
                        "<b>✅ KHỚP CHUẨN</b><br>"
                        "<b>Cụm HDBSCAN:</b> %{customdata[1]}<br>"
                        "<b>Intent thật:</b> %{customdata[2]}<br>"
                        "<b>Câu:</b> %{customdata[0]}<extra></extra>"
                    )
                ))

                # 2. Điểm bị HDBSCAN gom trộn/lệch (NỔI BẬT bằng hình thoi màu cam sáng viền trắng)
                mm_txt = mismatched_df["text"].str.slice(0, 85)
                mm_tl = mismatched_df["topic_label"].fillna("").astype(str).str.slice(0, 35)
                mm_ti = mismatched_df["true_intent"].fillna("").astype(str).str.slice(0, 35)
                mm_maj = mismatched_df["majority_intent_of_cluster"].fillna("").astype(str).str.slice(0, 35)
                mm_pur = np.round(mismatched_df["purity"].fillna(0).values, 2)
                fig.add_trace(go.Scattergl(
                    x=mismatched_df[x_col], y=mismatched_df[y_col],
                    mode="markers",
                    name="⚠️ Khác biệt 1: Bị HDBSCAN gộp trộn (Nhãn thật ≠ Đa số cụm) [21,4%]",
                    marker=dict(size=8.5, color="#F97316", symbol="diamond", opacity=0.95, line=dict(width=1, color="#FFFFFF")),
                    customdata=np.stack([mm_txt, mm_tl, mm_ti, mm_maj, mm_pur], axis=-1),
                    hovertemplate=(
                        "<b>⚠️ BỊ GỘP TRỘN LỆCH INTENT</b><br>"
                        "<b>Intent thật của câu:</b> %{customdata[2]}<br>"
                        "<b>Nhưng bị HDBSCAN gom vào:</b> %{customdata[1]} (vốn có đa số là <i>%{customdata[3]}</i>, Purity=%{customdata[4]:.2f})<br>"
                        "<b>Câu:</b> %{customdata[0]}<extra></extra>"
                    )
                ))

                # 3. Điểm có Intent thật nhưng bị HDBSCAN bỏ sót thành Noise (-1)
                n_txt = noise_df["text"].str.slice(0, 85)
                n_ti = noise_df["true_intent"].fillna("").astype(str).str.slice(0, 35)
                fig.add_trace(go.Scattergl(
                    x=noise_df[x_col], y=noise_df[y_col],
                    mode="markers",
                    name="❌ Khác biệt 2: Có Intent thật nhưng bị đánh Noise (-1)",
                    marker=dict(size=7, color="#EC4899", symbol="x", opacity=0.85),
                    customdata=np.stack([n_txt, n_ti], axis=-1),
                    hovertemplate=(
                        "<b>❌ BỊ ĐÁNH NOISE (-1)</b><br>"
                        "<b>Intent thật ban đầu:</b> %{customdata[1]}<br>"
                        "<b>Câu:</b> %{customdata[0]}<extra></extra>"
                    )
                ))
        else:
            # CHẾ ĐỘ TÔ MÀU THEO CỤM HDBSCAN
            if star_point is not None:
                # -------------------------------------------------------------
                # TỐI ƯU TAB 3 (THỬ MỘT CÂU):
                # Chỉ làm nổi bật các CỤM CÓ LÁNG GIỀNG GẦN NHẤT.
                # Tất cả các cụm xa khác được gom vào 1 trace nền mờ nhẹ -> Tải cực nhanh, không tràn RAM!
                # -------------------------------------------------------------
                neighbor_tids = set()
                if neighbor_points:
                    neighbor_tids = {int(nb["topic"]) for nb in neighbor_points if int(nb.get("topic", -1)) != -1}

                # 1. Điểm nền thuộc các cụm không liên quan
                bg_pts = clustered_df[~clustered_df[active_topic_col].isin(neighbor_tids)]
                if len(bg_pts) > 2000:
                    bg_pts = bg_pts.sample(2000, random_state=42)
                fig.add_trace(go.Scattergl(
                    x=bg_pts[x_col], y=bg_pts[y_col],
                    mode="markers",
                    name="Cụm nền xa (Làm mờ)",
                    marker=dict(size=4.5, color="#94A3B8", opacity=0.18),
                    hoverinfo="skip"
                ))

                # 2. Vùng Noise (-1) nền
                sample_noise = noise_df.sample(min(250, len(noise_df)), random_state=42) if len(noise_df) > 250 else noise_df
                fig.add_trace(go.Scattergl(
                    x=sample_noise[x_col], y=sample_noise[y_col],
                    mode="markers",
                    name="Vùng Noise (-1)",
                    marker=dict(size=4.5, color="#CBD5E1", symbol="x", opacity=0.20),
                    hoverinfo="skip"
                ))

                # 3. Chỉ vẽ các cụm chứa láng giềng gần nhất (Đậm nét, nổi bật)
                for tid_int in sorted(neighbor_tids):
                    sub = clustered_df[clustered_df[active_topic_col] == tid_int]
                    if sub.empty:
                        continue
                    maj_intent = sub["majority_intent_of_cluster"].iloc[0] if len(sub) > 0 else ""
                    lbl_name = f"Topic {tid_int:02d} ({maj_intent[:18]})"
                    sub_text = sub["text"].str.slice(0, 85)
                    sub_intent = sub["true_intent"].fillna("").astype(str).str.slice(0, 35)
                    sub_cdata = np.stack([sub_text, sub_intent], axis=-1)

                    fig.add_trace(go.Scattergl(
                        x=sub[x_col], y=sub[y_col],
                        mode="markers",
                        name=f"🎯 {lbl_name}",
                        showlegend=True,
                        marker=dict(
                            size=8.5,
                            color=palette[tid_int % len(palette)],
                            opacity=0.92,
                            line=dict(width=1.5, color="#FFFFFF")
                        ),
                        customdata=sub_cdata,
                        hovertemplate=(
                            f"<b>{lbl_name}</b><br>"
                            "<b>Intent thật:</b> %{customdata[1]}<br>"
                            "<b>Câu Train:</b> %{customdata[0]}<extra></extra>"
                        )
                    ))
            else:
                # -------------------------------------------------------------
                # CHẾ ĐỘ TAB 1 (DỰNG CỤM TỪNG BƯỚC):
                # Vẽ đầy đủ các cụm nhưng cắt ngắn chuỗi customdata để tiết kiệm RAM
                # -------------------------------------------------------------
                for tid in sorted(clustered_df[active_topic_col].unique()):
                    sub = clustered_df[clustered_df[active_topic_col] == tid]
                    tid_int = int(tid)
                    is_sel = (step == 4 and tid_int == int(selected_topic))

                    if step == 4:
                        pt_size = 11 if is_sel else 5.5
                        pt_opacity = 1.0 if is_sel else 0.20
                    else:
                        pt_size = 6.5
                        pt_opacity = 0.82

                    maj_intent = sub["majority_intent_of_cluster"].iloc[0] if len(sub) > 0 else ""
                    lbl_name = f"Topic {tid_int:02d} ({maj_intent[:16]})" if (mcs == 50 and selection == "eom") else f"Cụm #{tid_int:03d}"

                    sub_text = sub["text"].str.slice(0, 85)
                    sub_kw = sub["keywords"].fillna("").astype(str).str.slice(0, 50)
                    sub_lbl = sub["topic_label"].fillna("").astype(str).str.slice(0, 35)
                    sub_intent = sub["true_intent"].fillna("").astype(str).str.slice(0, 35)
                    sub_purity = np.round(sub["purity"].fillna(0).values, 2)
                    sub_cdata = np.stack([sub_text, sub_lbl, sub_kw, sub_purity, sub_intent], axis=-1)

                    fig.add_trace(go.Scattergl(
                        x=sub[x_col], y=sub[y_col],
                        mode="markers",
                        name=lbl_name,
                        showlegend=True,
                        marker=dict(
                            size=pt_size,
                            color=palette[tid_int % len(palette)],
                            opacity=pt_opacity,
                            line=dict(width=1.8, color="#FFFFFF") if is_sel else dict(width=0)
                        ),
                        customdata=sub_cdata,
                        hovertemplate=(
                            f"<b>{lbl_name}</b> | Purity gốc: %{{customdata[3]:.2f}}<br>"
                            "<b>Intent thật của câu:</b> %{customdata[4]}<br>"
                            "<b>Câu:</b> %{customdata[0]}<br>"
                            "<b>Keywords:</b> %{customdata[2]}<extra></extra>"
                        )
                    ))

                # Điểm Noise (-1)
                noise_pct_str = f"{len(noise_df) / len(df) * 100:.1f}%".replace(".", ",")
                fig.add_trace(go.Scattergl(
                    x=noise_df[x_col], y=noise_df[y_col],
                    mode="markers",
                    name=f"Noise (-1) [{noise_pct_str}]",
                    showlegend=True,
                    marker=dict(
                        size=6.0,
                        color="#94A3B8",
                        symbol="x",
                        opacity=0.20 if step == 4 else 0.72
                    ),
                    text=noise_df["text"].str.slice(0, 85),
                    hovertemplate="<b>[NOISE -1]</b> Không thuộc cụm nào<br><b>Câu:</b> %{text}<extra></extra>"
                ))

    # Vẽ 5 láng giềng gần nhất và đường tia nối tới Ngôi sao ★ ở Tab 3
    if star_point is not None and neighbor_points:
        sx, sy = float(star_point["x"]), float(star_point["y"])
        line_x, line_y = [], []
        nb_x, nb_y, nb_labels, nb_hovers = [], [], [], []

        for rank, nb in enumerate(neighbor_points, 1):
            if "x_2d" in nb and "y_2d" in nb and nb["x_2d"] is not None:
                nx, ny = float(nb["x_2d"]), float(nb["y_2d"])
            else:
                tid = nb.get("topic", -1)
                pts = scatter_df[scatter_df["topic"] == tid] if tid != -1 else pd.DataFrame()
                if not pts.empty:
                    idx_pt = (rank - 1) % len(pts)
                    nx = float(pts.iloc[idx_pt]["x_umap"])
                    ny = float(pts.iloc[idx_pt]["y_umap"])
                else:
                    angle = (rank - 1) * (2 * np.pi / max(1, len(neighbor_points)))
                    d_val = max(0.6, float(nb.get("dist", 0.08)) * 12.0)
                    nx = sx + d_val * np.cos(angle)
                    ny = sy + d_val * np.sin(angle)
            line_x.extend([sx, nx, None])
            line_y.extend([sy, ny, None])
            nb_x.append(nx)
            nb_y.append(ny)
            nb_labels.append(f"#{rank} (T.{nb['topic']})")
            nb_hovers.append(
                f"<b>Láng giềng #{rank} trong không gian 384D (Topic {nb['topic']} - {nb['intent']})</b><br>"
                f"Khoảng cách Cosine 384D: {fmt_num_vi(nb['dist'], 3)}<br>"
                f"Câu Train: \"{nb['text']}\""
            )

        fig.add_trace(go.Scatter(
            x=line_x, y=line_y,
            mode="lines",
            name="Liên kết kNN 384D (Top 5)",
            line=dict(color="#F59E0B", width=2.2, dash="dot"),
            hoverinfo="skip"
        ))

        fig.add_trace(go.Scatter(
            x=nb_x, y=nb_y,
            mode="markers+text",
            name="🎯 5 câu Train gần nhất (ở 384D)",
            text=nb_labels,
            textposition="bottom center",
            textfont=dict(size=11, color="#FBBF24", family="Arial Black"),
            marker=dict(size=12, color="#F59E0B", symbol="diamond", line=dict(width=2, color="#1E293B")),
            hovertext=nb_hovers,
            hoverinfo="text"
        ))

    if star_point is not None:
        fig.add_trace(go.Scatter(
            x=[star_point["x"]],
            y=[star_point["y"]],
            mode="markers+text",
            name="★ Câu bạn vừa thử",
            text=["★ Câu của bạn"],
            textposition="top center",
            textfont=dict(size=13, color="#EF4444", family="Arial Black"),
            marker=dict(size=20, color="#EF4444", symbol="star", line=dict(width=2.2, color="#FFFFFF")),
            hovertemplate=f"<b>★ Câu của bạn:</b> {star_point.get('text', '')}<br>Toạ độ 2D: ({star_point['x']:.2f}, {star_point['y']:.2f})<extra></extra>"
        ))

    n_clusters_shown = len(set(df[active_topic_col]) - {-1})
    step_titles = {
        1: "Bước 1: Vector 384D chiếu phẳng (Trước khi giảm chiều UMAP)",
        2: "Bước 2: Sau UMAP (Co cụm mật độ theo khoảng cách Cosine)",
        3: f"Bước 3: Sau HDBSCAN (mcs={mcs}, {selection}) — Tạo ra {n_clusters_shown} cụm",
        4: f"Bước 4: c-TF-IDF theo từng câu — Đang soi nổi bật Topic {selected_topic}"
    }

    fig.update_layout(
        template="plotly_white",
        title=dict(text=step_titles.get(step, "Không gian 2D minh hoạ"), font=dict(size=15.5)),
        dragmode="pan",  # Cho phép kéo chuột di chuyển biểu đồ mượt mà + lăn chuột để Zoom In/Out!
        xaxis=dict(
            title=dict(text="Chiều UMAP 1 (Minh hoạ 2D — Lăn chuột để Zoom, Kéo để di chuyển)", standoff=15),
            range=[-10.5, 10.5],
            zeroline=False,
            fixedrange=False
        ),
        yaxis=dict(
            title=dict(text="Chiều UMAP 2 (Minh hoạ 2D)", standoff=10),
            range=[-9.5, 9.5],
            zeroline=False,
            fixedrange=False
        ),
        height=550,
        margin=dict(l=20, r=230, t=50, b=55),
        legend=dict(
            title=dict(text="<b>Chú thích / Bộ lọc</b><br><span style='font-size:11px'>(Bấm để ẩn/hiện)</span>"),
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,
            font=dict(size=11.5),
            bgcolor="rgba(0,0,0,0)"
        )
    )
    return fig

def plot_novelty_gauge(novelty_val: float, thr_p95: float = 0.118) -> go.Figure:
    max_axis = max(0.38, round(novelty_val * 1.25, 2))
    is_flagged = novelty_val >= thr_p95
    bar_color = "#C65D3B" if is_flagged else "#3B82F6"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=novelty_val,
        number={"valueformat": ".3f", "font": {"size": 34, "color": bar_color}},
        delta={
            "reference": thr_p95,
            "valueformat": ".3f",
            "increasing": {"color": "#C65D3B"},
            "decreasing": {"color": "#10B981"}
        },
        title={
            "text": f"<b>Điểm kNN Novelty (k=10)</b><br><span style='font-size:12px;color:gray'>Ngưỡng p95 Train = {fmt_num_vi(thr_p95, 3)}</span>"
        },
        gauge={
            "axis": {"range": [0, max_axis], "tickwidth": 1},
            "bar": {"color": bar_color, "thickness": 0.45},
            "steps": [
                {"range": [0, thr_p95], "color": "rgba(59, 130, 246, 0.18)"},
                {"range": [thr_p95, max_axis], "color": "rgba(239, 68, 68, 0.22)"}
            ],
            "threshold": {
                "line": {"color": "#EF4444", "width": 4},
                "thickness": 0.85,
                "value": thr_p95
            }
        }
    ))
    fig.update_layout(template="plotly_white", height=250, margin=dict(l=25, r=25, t=55, b=15), dragmode=False)
    return fig