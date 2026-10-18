# components/insight.py
import uuid
import html
import streamlit as st
import streamlit.components.v1 as components

# Từ điển giải thích thuật ngữ tự động cho người xem không chuyên ML
TERM_TOOLTIPS = {
    "V-measure": "Thước đo từ 0 đến 1 đánh giá mức độ khớp giữa cụm máy tự gom và nhãn thật (càng gần 1 càng tốt).",
    "ARI": "Adjusted Rand Index: Đo tỷ lệ các cặp câu được gom đúng vào cùng cụm hoặc khác cụm sau khi đã trừ phần trùng hợp ngẫu nhiên.",
    "Purity": "Độ tinh khiết: Tỷ lệ phần trăm của nhóm câu chiếm đa số trong một cụm (Purity = 0,97 nghĩa là 97% câu trong cụm thuộc cùng 1 chủ đề thật).",
    "DBCV": "Chỉ số đánh giá chất lượng cụm dựa trên mật độ nội tại mà KHÔNG cần dùng nhãn thật.",
    "AUROC": "Đo khả năng phân biệt câu thuộc chủ đề mới và câu thuộc chủ đề cũ ở mọi ngưỡng (0,5 = đoán mò ngẫu nhiên; 1,0 = phân biệt hoàn hảo).",
    "FPR": "False Positive Rate (Tỷ lệ báo động giả): Tỷ lệ câu thuộc chủ đề cũ đã biết nhưng bị gắn cờ nhầm là chủ đề mới.",
    "Recall": "Độ phủ phát hiện: Tỷ lệ câu thuộc chủ đề mới thực sự được hệ thống phát hiện và gắn cờ.",
    "Lift": "Hệ số nâng (Recall / FPR): Cho biết bộ phát hiện hiệu quả gấp bao nhiêu lần so với việc bốc ngẫu nhiên.",
    "NPMI": "Normalized Pointwise Mutual Information: Đo mức độ các từ khóa trong cùng 1 cụm thường xuyên xuất hiện cùng nhau trong một câu.",
    "CI95": "Khoảng tin cậy 95% (Bootstrap): Nếu khoảng chênh lệch không chứa số 0, sự cải thiện là có ý nghĩa thống kê thực sự."
}

def _inject_tooltips(text: str) -> str:
    """Tự động gắn biểu tượng (?) có tooltip giải thích cho các thuật ngữ ML xuất hiện trong chuỗi."""
    if not text:
        return ""
    safe_text = html.escape(str(text))
    for term, desc in TERM_TOOLTIPS.items():
        # Chỉ thay thế lần xuất hiện đầu tiên trong mỗi đoạn để giao diện thoáng
        pattern = re.compile(rf"\b({re.escape(term)})\b")
        replacement = (
            rf'\1<span class="ins-tt" tabindex="0" data-tip="{html.escape(desc)}">?</span>'
        )
        safe_text, count = pattern.subn(replacement, safe_text, count=1)
    return safe_text

import re

def render_insight(
    title: str,
    left: dict,
    right: dict,
    finding: str,
    footnote: str = "BANKING77: train 9.993 mẫu, test 3.076 mẫu, 77 intent. Nhãn chỉ dùng để đánh giá.",
    page: str = "1/8",
    enable_actions: bool = True,
    height: int = 580
):
    """
    Hàm tái sử dụng hiển thị InsightSlide theo đúng chuẩn thiết kế.
    Tham số `left` và `right` là dict gồm:
      - name: Tên đối tượng so sánh (vd: "Pipeline gốc (MiniLM-L6)")
      - condition: Điều kiện đo in đậm (vd: "Train, 9.993 mẫu")
      - metric_value: Số lớn màu navy (vd: "0,790")
      - metric_unit: Đơn vị / Tên chỉ số cạnh số lớn (vd: "V-measure")
      - subtext: Dòng nhỏ giải thích tỷ lệ/ý nghĩa dưới số lớn
      - cross_ref: Khối đối chiếu chéo trong cùng điều kiện đo
      - repeatability: Dòng kiểm tra độ lặp lại (nếu None hoặc rỗng sẽ hiện 'Chưa có kiểm tra lặp')
    """
    uid = f"slide_{uuid.uuid4().hex[:8]}"

    left_rep = left.get("repeatability") or "Chưa có kiểm tra lặp"
    right_rep = right.get("repeatability") or "Chưa có kiểm tra lặp"

    html_code = f"""
    <!DOCTYPE html>
    <html lang="vi">
    <head>
    <meta charset="utf-8">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
    <style>
        :root {{
            --navy: #1B3A5C;
            --navy-text: #1B3A5C;
            --card-bg: #E8F0F8;
            --card-border: #C9DDF0;
            --cross-bg: rgba(27, 58, 92, 0.06);
            --slide-bg: #FFFFFF;
            --text-main: #1A2530;
            --text-sub: #4A5A6A;
            --orange-hdr: #C65D3B;
            --orange-bg: #FDF1EC;
            --orange-border: #F2C9B8;
        }}
        @media (prefers-color-scheme: dark) {{
            :root {{
                --navy: #1B3A5C;
                --navy-text: #90C2F9;
                --card-bg: #162436;
                --card-border: #284260;
                --cross-bg: rgba(255, 255, 255, 0.05);
                --slide-bg: #0E1621;
                --text-main: #EAEFF5;
                --text-sub: #B0C0D0;
                --orange-hdr: #C65D3B;
                --orange-bg: #2B1914;
                --orange-border: #6E3422;
            }}
        }}
        * {{ box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        body {{ margin: 0; padding: 4px; background: transparent; color: var(--text-main); }}
        .toolbar {{
            display: {"flex" if enable_actions else "none"};
            justify-content: flex-end; gap: 8px; margin-bottom: 8px;
        }}
        .btn-act {{
            background: var(--card-bg); color: var(--navy-text);
            border: 1px solid var(--card-border); border-radius: 6px;
            padding: 5px 12px; font-size: 12.5px; font-weight: 600;
            cursor: pointer; transition: all 0.15s;
        }}
        .btn-act:hover {{ opacity: 0.85; transform: translateY(-1px); }}
        .slide-box {{
            background: var(--slide-bg);
            border: 1.5px solid var(--card-border);
            border-radius: 14px;
            padding: 22px 26px 16px 26px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.06);
            position: relative;
        }}
        .slide-box:fullscreen {{
            padding: 48px 64px;
            display: flex; flex-direction: column; justify-content: center;
            overflow-y: auto;
        }}
        .slide-title {{
            color: var(--navy-text);
            font-size: 22px; font-weight: 800;
            margin: 0 0 18px 0; line-height: 1.3;
            letter-spacing: -0.2px;
        }}
        .cards-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 18px;
            margin-bottom: 16px;
        }}
        @media (max-width: 680px) {{
            .cards-grid {{ grid-template-columns: 1fr; }}
        }}
        .ins-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            overflow: hidden;
            display: flex; flex-direction: column;
        }}
        .card-hdr {{
            background: var(--navy);
            color: #FFFFFF;
            font-weight: 700; font-size: 15.5px;
            padding: 10px 16px;
            letter-spacing: 0.1px;
        }}
        .card-body {{
            padding: 14px 16px;
            display: flex; flex-direction: column; flex: 1;
        }}
        .cond-line {{
            font-weight: 700; font-size: 13.5px;
            color: var(--text-main);
            margin-bottom: 10px;
        }}
        .big-num-wrap {{
            display: flex; align-items: baseline; gap: 8px;
            margin: 2px 0 4px 0;
        }}
        .big-num {{
            font-size: 38px; font-weight: 800;
            color: var(--navy-text);
            line-height: 1.05;
        }}
        .big-unit {{
            font-size: 15px; font-weight: 700;
            color: var(--navy-text);
        }}
        .sub-line {{
            font-size: 13px; color: var(--text-sub);
            margin-bottom: 12px; line-height: 1.4;
        }}
        .cross-box {{
            background: var(--cross-bg);
            border-left: 3.5px solid var(--navy-text);
            padding: 8px 11px;
            border-radius: 0 6px 6px 0;
            font-size: 13px; line-height: 1.45;
            margin-bottom: 10px;
        }}
        .rep-line {{
            margin-top: auto;
            font-style: italic; font-size: 12.5px;
            color: var(--text-sub);
            padding-top: 6px;
            border-top: 1px dashed var(--card-border);
        }}
        .finding-banner {{
            border: 1px solid var(--orange-border);
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 12px;
        }}
        .finding-hdr {{
            background: var(--orange-hdr);
            color: #FFFFFF;
            font-weight: 700; font-size: 14px;
            padding: 6px 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .finding-body {{
            background: var(--orange-bg);
            color: var(--text-main);
            padding: 11px 14px;
            font-size: 14px; line-height: 1.5;
        }}
        .footer-row {{
            display: flex; justify-content: space-between; align-items: center;
            font-size: 12px; color: var(--text-sub);
            margin-top: 4px;
        }}
        .footnote-center {{
            text-align: center; flex: 1;
        }}
        .page-num {{
            font-weight: 700; color: var(--navy-text);
            margin-left: 12px;
        }}
        /* Tooltip (?) */
        .ins-tt {{
            display: inline-flex; align-items: center; justify-content: center;
            width: 15px; height: 15px; border-radius: 50%;
            background: var(--navy-text); color: var(--slide-bg);
            font-size: 10px; font-weight: 800; margin-left: 4px;
            cursor: help; position: relative; vertical-align: middle;
            font-style: normal;
        }}
        .ins-tt:hover::after, .ins-tt:focus::after {{
            content: attr(data-tip);
            position: absolute; bottom: 135%; left: 50%;
            transform: translateX(-50%);
            width: 250px; padding: 8px 10px;
            background: #1B3A5C; color: #FFFFFF;
            font-size: 11.5px; font-weight: 400; line-height: 1.4;
            border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.25);
            z-index: 9999; text-align: left; pointer-events: none;
        }}
    </style>
    </head>
    <body>
        <div class="toolbar">
            <button class="btn-act" onclick="toggleFS_{uid}()" title="Toàn màn hình (hỗ trợ phím ←/→)">⛶ Chế độ trình chiếu</button>
            <button class="btn-act" onclick="exportPNG_{uid}()" title="Tải ảnh PNG của slide này">⬇ Xuất PNG</button>
        </div>

        <div id="{uid}" class="slide-box" tabindex="0">
            <div class="slide-title">{_inject_tooltips(title)}</div>

            <div class="cards-grid">
                <!-- THẺ TRÁI -->
                <div class="ins-card">
                    <div class="card-hdr">{html.escape(str(left.get("name", "")))}</div>
                    <div class="card-body">
                        <div class="cond-line">{html.escape(str(left.get("condition", "")))}</div>
                        <div class="big-num-wrap">
                            <span class="big-num">{html.escape(str(left.get("metric_value", "")))}</span>
                            <span class="big-unit">{_inject_tooltips(str(left.get("metric_unit", "")))}</span>
                        </div>
                        <div class="sub-line">{_inject_tooltips(str(left.get("subtext", "")))}</div>
                        <div class="cross-box">
                            <strong>Đối chiếu cùng điều kiện:</strong> {_inject_tooltips(str(left.get("cross_ref", "")))}
                        </div>
                        <div class="rep-line">Kiểm tra độ lặp lại: {_inject_tooltips(str(left_rep))}</div>
                    </div>
                </div>

                <!-- THẺ PHẢI -->
                <div class="ins-card">
                    <div class="card-hdr">{html.escape(str(right.get("name", "")))}</div>
                    <div class="card-body">
                        <div class="cond-line">{html.escape(str(right.get("condition", "")))}</div>
                        <div class="big-num-wrap">
                            <span class="big-num">{html.escape(str(right.get("metric_value", "")))}</span>
                            <span class="big-unit">{_inject_tooltips(str(right.get("metric_unit", "")))}</span>
                        </div>
                        <div class="sub-line">{_inject_tooltips(str(right.get("subtext", "")))}</div>
                        <div class="cross-box">
                            <strong>Đối chiếu cùng điều kiện:</strong> {_inject_tooltips(str(right.get("cross_ref", "")))}
                        </div>
                        <div class="rep-line">Kiểm tra độ lặp lại: {_inject_tooltips(str(right_rep))}</div>
                    </div>
                </div>
            </div>

            <!-- BANNER PHÁT HIỆN -->
            <div class="finding-banner">
                <div class="finding-hdr">Phát hiện</div>
                <div class="finding-body">{_inject_tooltips(finding)}</div>
            </div>

            <!-- FOOTER & SỐ TRANG -->
            <div class="footer-row">
                <div class="footnote-center">{html.escape(footnote)}</div>
                <div class="page-num">{html.escape(str(page))}</div>
            </div>
        </div>

        <script>
        function toggleFS_{uid}() {{
            const el = document.getElementById("{uid}");
            if (!document.fullscreenElement) {{
                el.requestFullscreen().catch(err => alert("Không thể bật toàn màn hình: " + err.message));
                el.focus();
            }} else {{
                document.exitFullscreen();
            }}
        }}
        function exportPNG_{uid}() {{
            const el = document.getElementById("{uid}");
            html2canvas(el, {{ scale: 2, useCORS: true }}).then(canvas => {{
                const link = document.createElement("a");
                link.download = "BANKING77_Insight_{page.replace('/', '_')}.png";
                link.href = canvas.toDataURL("image/png");
                link.click();
            }});
        }}
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=height, scrolling=False)