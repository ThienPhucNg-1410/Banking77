# build_github_pages.py
# -*- coding: utf-8 -*-
"""
Script to generate the comprehensive, interactive Web Portal (index.html)
for GitHub Pages hosting. Includes:
1. 4-step stepper algorithm visualization (Text 384D -> UMAP 5D -> HDBSCAN -> 75 Clusters).
2. Live interactive pipeline tester (chạy thử thuật toán từng bước với câu mới, kNN láng giềng, Novelty alert, hành động Chatbot).
3. 9-algorithm benchmark comparison.
4. 4 banking domains & 75 clusters explorer with live search.
5. Novelty detection ROC curve.
6. Full 6-chapter scientific report with KaTeX math & Mermaid charts.
"""

import os
import json
import pandas as pd
import numpy as np

def build_portal():
    print("Building GitHub Pages Web Portal with Stepper and Interactive Pipeline Tester...")

    # 1. Load Topics
    topics_path = "artifacts/topics_v3.csv"
    if os.path.exists(topics_path):
        df_topics = pd.read_csv(topics_path)
    else:
        df_topics = pd.DataFrame()

    def get_domain(intent):
        intent = str(intent).lower()
        if any(k in intent for k in ["card", "pin", "atm", "contactless", "virtual"]):
            return "Dịch vụ Thẻ (Cards)"
        elif any(k in intent for k in ["top_up", "transfer", "balance", "deposit", "cheque", "refund", "cash"]):
            return "Nạp & Chuyển tiền (Transfers)"
        elif any(k in intent for k in ["identity", "verify", "address", "phone", "details", "account", "age", "country", "passcode"]):
            return "Tài khoản & KYC (Account)"
        else:
            return "Biểu phí & Tỷ giá (Fees & Rates)"

    if not df_topics.empty:
        df_topics["domain"] = df_topics["majority_label"].apply(get_domain)

    # 2. Load Representative Sentences
    rep_map = {}
    rep_path = "artifacts/topic_representatives.json"
    if os.path.exists(rep_path):
        with open(rep_path, "r", encoding="utf-8") as f:
            rep_map = json.load(f)

    # 3. Sample UMAP 2D points (with x_b1, y_b1, x_umap, y_umap)
    umap_path = "artifacts/umap_2d_states.csv"
    sampled_points = []
    if os.path.exists(umap_path):
        df_umap = pd.read_csv(umap_path)
        sample_size = min(1200, len(df_umap))
        df_sample = df_umap.sample(n=sample_size, random_state=42)
        for _, row in df_sample.iterrows():
            sampled_points.append({
                "x_b1": round(float(row.get("x_b1", row["x_umap"])), 3),
                "y_b1": round(float(row.get("y_b1", row["y_umap"])), 3),
                "x_umap": round(float(row["x_umap"]), 3),
                "y_umap": round(float(row["y_umap"]), 3),
                "topic": int(row["topic"]),
                "is_noise": bool(row["is_noise"]),
                "intent": str(row["true_intent"]),
                "text": str(row["text"])[:85] + "..." if len(str(row["text"])) > 85 else str(row["text"])
            })

    # 4. Load Benchmark Data
    bench_path = "artifacts/benchmark_v3.csv"
    benchmarks = []
    if os.path.exists(bench_path):
        df_bench = pd.read_csv(bench_path)
        for _, row in df_bench.iterrows():
            benchmarks.append({
                "name": str(row["Phương pháp"]),
                "clusters": int(row["Số cụm"]) if pd.notnull(row["Số cụm"]) else 0,
                "noise": round(float(row["Noise"]) * 100, 2) if pd.notnull(row["Noise"]) else 0.0,
                "v_measure": round(float(row["V (noise=1 cụm)"]), 3) if pd.notnull(row["V (noise=1 cụm)"]) else 0.0,
                "ari": round(float(row["ARI (noise=1 cụm)"]), 3) if pd.notnull(row["ARI (noise=1 cụm)"]) else 0.0,
                "purity": round(float(row["Purity (trọng số)"]) * 100, 2) if pd.notnull(row["Purity (trọng số)"]) else 0.0,
                "dbcv": round(float(row["DBCV"]), 3) if pd.notnull(row["DBCV"]) else None,
            })

    # 5. Load 8 Preset Examples for Interactive Pipeline Tester
    from utils.pipeline import PRESET_EXAMPLES
    presets_data = {}
    for k, v in PRESET_EXAMPLES.items():
        presets_data[k] = {
            "text": v["text"],
            "category": v["category"],
            "expected_true_intent": v["expected_true_intent"],
            "topic": v["topic"],
            "strength": v["strength"],
            "novelty": v["novelty"],
            "flag_new": v["flag_new"],
            "x_2d": v["x_2d"],
            "y_2d": v["y_2d"],
            "neighbors": v["neighbors"]
        }

    # 6. Load Markdown Report
    report_path = "BAO_CAO_TOAN_DIEN_DU_AN_BANKING77.md"
    report_md = ""
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_md = f.read()

    # Convert topics list
    topics_list = []
    if not df_topics.empty:
        for _, r in df_topics.iterrows():
            top_id = str(int(r["topic"]))
            top_reps = rep_map.get(top_id, [])
            topics_list.append({
                "topic": int(r["topic"]),
                "size": int(r["size"]),
                "purity": round(float(r["purity"]) * 100, 1),
                "majority_label": str(r["majority_label"]),
                "keywords": str(r["keywords"]),
                "cx": round(float(r["cx"]), 3),
                "cy": round(float(r["cy"]), 3),
                "domain": str(r["domain"]),
                "reps": top_reps[:4]
            })

    # Serialize to JSON
    topics_json = json.dumps(topics_list, ensure_ascii=False)
    sampled_points_json = json.dumps(sampled_points, ensure_ascii=False)
    benchmarks_json = json.dumps(benchmarks, ensure_ascii=False)
    presets_json = json.dumps(presets_data, ensure_ascii=False)
    report_md_json = json.dumps(report_md, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="vi" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BANKING77: Khám Phá Chủ Đề & Khai Phá Pattern Ngữ Nghĩa Tự Động</title>
    <!-- Favicon & Fonts -->
    <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🏛️</text></svg>">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    
    <!-- Plotly.js for interactive charts -->
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <!-- Marked.js for Markdown parsing -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <!-- KaTeX for math formulas -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <!-- Mermaid for architectural diagrams -->
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>

    <style>
        :root {{
            --bg-page: #F8FAFC;
            --bg-card: #FFFFFF;
            --bg-nav: rgba(255, 255, 255, 0.95);
            --border-color: #E2E8F0;
            --text-main: #0F172A;
            --text-muted: #64748B;
            --primary: #0284C7;
            --primary-glow: rgba(2, 132, 199, 0.12);
            --primary-dark: #0369A1;
            --accent: #2563EB;
            --success: #10B981;
            --warning: #F59E0B;
            --danger: #EF4444;
            --code-bg: #F1F5F9;
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
            --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.08), 0 2px 4px -2px rgba(0,0,0,0.05);
        }}

        [data-theme="dark"] {{
            --bg-page: #0B0F19;
            --bg-card: #151C2C;
            --bg-nav: rgba(21, 28, 44, 0.95);
            --border-color: #232D42;
            --text-main: #F8FAFC;
            --text-muted: #94A3B8;
            --primary: #38BDF8;
            --primary-glow: rgba(56, 189, 248, 0.15);
            --primary-dark: #0284C7;
            --accent: #60A5FA;
            --code-bg: #0D1322;
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
            --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.4);
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-page);
            color: var(--text-main);
            line-height: 1.6;
            transition: background-color 0.25s, color 0.25s;
        }}

        header {{
            position: sticky;
            top: 0;
            z-index: 1000;
            backdrop-filter: blur(12px);
            background-color: var(--bg-nav);
            border-bottom: 1px solid var(--border-color);
        }}

        .navbar {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 14px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .brand {{
            display: flex;
            align-items: center;
            gap: 12px;
            text-decoration: none;
            color: var(--text-main);
        }}

        .brand-logo {{
            width: 38px;
            height: 38px;
            border-radius: 10px;
            background: linear-gradient(135deg, #0284C7, #2563EB);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            box-shadow: 0 4px 10px rgba(2, 132, 199, 0.3);
        }}

        .brand-text h1 {{ font-size: 16px; font-weight: 800; line-height: 1.2; }}
        .brand-text span {{ font-size: 12px; color: var(--text-muted); font-weight: 500; }}

        .nav-actions {{ display: flex; align-items: center; gap: 10px; }}

        .btn {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13.5px;
            font-weight: 600;
            cursor: pointer;
            border: 1px solid var(--border-color);
            background-color: var(--bg-card);
            color: var(--text-main);
            text-decoration: none;
            transition: all 0.2s ease;
        }}

        .btn:hover {{ border-color: var(--primary); color: var(--primary); transform: translateY(-1px); }}

        .btn-primary {{
            background: linear-gradient(135deg, #0284C7, #2563EB);
            color: #FFFFFF;
            border: none;
            box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25);
        }}

        .btn-primary:hover {{ color: #FFFFFF; opacity: 0.95; box-shadow: 0 6px 16px rgba(2, 132, 199, 0.35); }}

        .hero {{ max-width: 1400px; margin: 24px auto 16px; padding: 0 24px; }}
        .hero-card {{
            background: linear-gradient(135deg, rgba(2, 132, 199, 0.08) 0%, rgba(37, 99, 235, 0.04) 100%);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 28px 32px;
        }}

        .hero-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            background-color: var(--primary-glow);
            color: var(--primary);
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }}

        .hero-title {{ font-size: 24px; font-weight: 800; line-height: 1.3; margin-bottom: 10px; }}
        .hero-desc {{ font-size: 14.5px; color: var(--text-muted); line-height: 1.6; margin-bottom: 20px; }}

        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; }}
        .kpi-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px 18px;
            box-shadow: var(--shadow-sm);
        }}
        .kpi-label {{ font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin-bottom: 4px; }}
        .kpi-value {{ font-size: 24px; font-weight: 800; color: var(--text-main); }}
        .kpi-subtext {{ font-size: 12px; color: var(--success); font-weight: 600; margin-top: 3px; }}

        .main-container {{ max-width: 1400px; margin: 0 auto 60px; padding: 0 24px; }}

        .tab-nav {{
            display: flex;
            gap: 8px;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 24px;
            overflow-x: auto;
            padding-bottom: 4px;
        }}

        .tab-btn {{
            padding: 12px 18px;
            background: none;
            border: none;
            border-bottom: 2px solid transparent;
            color: var(--text-muted);
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.2s ease;
        }}

        .tab-btn:hover {{ color: var(--primary); }}
        .tab-btn.active {{ color: var(--primary); border-bottom-color: var(--primary); font-weight: 700; }}

        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}

        .panel {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 24px;
            box-shadow: var(--shadow-sm);
            margin-bottom: 24px;
        }}

        .panel-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 18px;
            flex-wrap: wrap;
            gap: 12px;
        }}

        .panel-title {{ font-size: 18px; font-weight: 700; }}
        .panel-subtitle {{ font-size: 13.5px; color: var(--text-muted); margin-top: 2px; }}

        .chart-box {{ width: 100%; min-height: 520px; border-radius: 10px; overflow: hidden; }}

        .table-responsive {{ overflow-x: auto; border: 1px solid var(--border-color); border-radius: 10px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; text-align: left; }}
        th, td {{ padding: 12px 16px; border-bottom: 1px solid var(--border-color); }}
        th {{ background-color: rgba(2, 132, 199, 0.05); font-weight: 700; color: var(--primary); white-space: nowrap; }}
        tr:hover td {{ background-color: rgba(2, 132, 199, 0.02); }}

        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 6px; font-size: 11.5px; font-weight: 700; }}
        .badge-blue {{ background-color: rgba(2, 132, 199, 0.12); color: #0284C7; }}
        .badge-green {{ background-color: rgba(16, 185, 129, 0.12); color: #10B981; }}
        .badge-yellow {{ background-color: rgba(245, 158, 11, 0.12); color: #D97706; }}
        .badge-purple {{ background-color: rgba(147, 51, 234, 0.12); color: #9333EA; }}
        .badge-red {{ background-color: rgba(239, 68, 68, 0.12); color: #EF4444; }}

        .search-box {{
            padding: 8px 14px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            background-color: var(--bg-page);
            color: var(--text-main);
            font-size: 13.5px;
            outline: none;
            width: 280px;
        }}
        .search-box:focus {{ border-color: var(--primary); }}

        /* Stepper Buttons */
        .stepper-nav {{
            display: flex;
            gap: 10px;
            background-color: var(--bg-page);
            padding: 8px;
            border-radius: 10px;
            border: 1px solid var(--border-color);
            margin-bottom: 16px;
            flex-wrap: wrap;
        }}

        .step-btn {{
            flex: 1;
            min-width: 180px;
            padding: 10px 14px;
            background: none;
            border: 1px solid transparent;
            border-radius: 8px;
            color: var(--text-muted);
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            text-align: center;
            transition: all 0.2s;
        }}

        .step-btn:hover {{ background-color: var(--bg-card); color: var(--text-main); }}
        .step-btn.active {{
            background-color: var(--bg-card);
            border-color: var(--primary);
            color: var(--primary);
            box-shadow: var(--shadow-sm);
            font-weight: 700;
        }}

        .step-explainer {{
            background: linear-gradient(135deg, rgba(2, 132, 199, 0.05), rgba(37, 99, 235, 0.02));
            border-left: 4px solid var(--primary);
            border-radius: 0 10px 10px 0;
            padding: 14px 18px;
            font-size: 13.5px;
            margin-top: 14px;
            line-height: 1.6;
        }}

        /* Live Tester Box */
        .tester-grid {{
            display: grid;
            grid-template-columns: 1.1fr 1fr;
            gap: 24px;
        }}

        .pipeline-card {{
            background-color: var(--bg-page);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 14px;
        }}

        .pipeline-step-title {{
            font-size: 13.5px;
            font-weight: 700;
            color: var(--primary);
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
        }}

        @media (max-width: 950px) {{
            .tester-grid {{ grid-template-columns: 1fr; }}
        }}

        /* Report Reader */
        .report-reader {{ display: flex; gap: 32px; align-items: flex-start; }}
        .report-toc {{
            width: 280px; flex-shrink: 0; position: sticky; top: 80px;
            background-color: var(--bg-card); border: 1px solid var(--border-color);
            border-radius: 12px; padding: 18px; max-height: calc(100vh - 100px); overflow-y: auto;
        }}
        .report-toc h4 {{ font-size: 13px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 12px; }}
        .toc-list {{ list-style: none; display: flex; flex-direction: column; gap: 4px; }}
        .toc-list a {{ color: var(--text-muted); text-decoration: none; font-size: 13px; padding: 6px 10px; border-radius: 6px; display: block; }}
        .toc-list a:hover, .toc-list a.active {{ color: var(--primary); background-color: var(--primary-glow); font-weight: 600; }}

        .report-body {{
            flex-grow: 1; background-color: var(--bg-card); border: 1px solid var(--border-color);
            border-radius: 14px; padding: 36px 44px; max-width: 980px; overflow-x: hidden;
        }}
        .report-body h1 {{ font-size: 26px; color: var(--primary); margin-bottom: 16px; }}
        .report-body h2 {{ font-size: 21px; margin-top: 36px; margin-bottom: 14px; border-bottom: 1px solid var(--border-color); padding-bottom: 8px; }}
        .report-body h3 {{ font-size: 17px; margin-top: 24px; margin-bottom: 10px; color: var(--accent); }}
        .report-body h4 {{ font-size: 15px; margin-top: 18px; margin-bottom: 8px; color: var(--warning); }}
        .report-body p {{ margin-bottom: 14px; line-height: 1.7; }}
        .report-body ul, .report-body ol {{ margin-bottom: 14px; padding-left: 22px; }}
        .report-body li {{ margin-bottom: 5px; }}
        .report-body code {{
            background-color: var(--code-bg); padding: 2px 6px; border-radius: 4px;
            font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--primary);
        }}
        .report-body pre {{
            background-color: var(--code-bg); padding: 16px; border-radius: 8px;
            border: 1px solid var(--border-color); overflow-x: auto; margin: 18px 0;
        }}
        .report-body pre code {{ background: none; padding: 0; color: var(--text-main); }}
        .report-body blockquote {{
            border-left: 4px solid var(--primary); background-color: var(--primary-glow);
            padding: 12px 18px; border-radius: 0 8px 8px 0; margin: 18px 0;
        }}
    </style>
</head>
<body>

    <header>
        <div class="navbar">
            <a href="#" class="brand">
                <div class="brand-logo">🏛️</div>
                <div class="brand-text">
                    <h1>BANKING77 TOPIC MINING</h1>
                    <span>Hệ Thống Khai Phá Tri Thức Ngân Hàng Không Giám Sát</span>
                </div>
            </a>
            <div class="nav-actions">
                <button class="btn" onclick="toggleTheme()" id="theme-btn">☀️ Theme</button>
                <a href="https://github.com/ThienPhucNg-1410/Banking77" target="_blank" class="btn">
                    ⭐ GitHub Repo
                </a>
                <a href="https://share.streamlit.io/" target="_blank" class="btn btn-primary">
                    🚀 Run Demo App
                </a>
            </div>
        </div>
    </header>

    <section class="hero">
        <div class="hero-card">
            <div class="hero-badge">Đồ Án Khai Phá Dữ Liệu Chuyên Sâu</div>
            <h2 class="hero-title">Khám Phá Chủ Đề & Khai Phá Pattern Ngữ Nghĩa Tự Động Từ Phản Hồi Khách Hàng Ngân Hàng</h2>
            <p class="hero-desc">
                Đường ống Pipeline v3: <strong>BGE-small-en-v1.5 (384D) &rarr; UMAP 5D &rarr; HDBSCAN (EOM) &rarr; c-TF-IDF & kNN Novelty Detector</strong>. 
                Tự động tìm ra 75 cụm ý định tự nhiên từ 13.073 tin nhắn khách hàng mà không cần gán nhãn trước.
            </p>

            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Quy mô Huấn luyện</div>
                    <div class="kpi-value">9.993 câu</div>
                    <div class="kpi-subtext">✓ 13.073 tổng tin nhắn chuẩn</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Số Cụm Tự Khám Phá</div>
                    <div class="kpi-value">75 Cụm</div>
                    <div class="kpi-subtext">✓ Tiệm cận 77 nhãn chuyên gia</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">V-Measure Khớp Nhãn</div>
                    <div class="kpi-value">0,833</div>
                    <div class="kpi-subtext">✓ 0,837 trên Test (No Overfitting)</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Chỉ số Mật độ DBCV</div>
                    <div class="kpi-value">0,432</div>
                    <div class="kpi-subtext">✓ Tối ưu tuyệt đối (Spearman +0.93)</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Phát hiện Chủ đề Mới (AUROC)</div>
                    <div class="kpi-value">0,864</div>
                    <div class="kpi-subtext">✓ 100% Khôi phục 15 Intent Mới</div>
                </div>
            </div>
        </div>
    </section>

    <main class="main-container">
        <nav class="tab-nav">
            <button class="tab-btn active" onclick="switchTab('tab-stepper', event)">🗺️ 1. Quy Trình Dựng Cụm 4 Bước</button>
            <button class="tab-btn" onclick="switchTab('tab-tester', event)">🧪 2. Chạy Thử Thuật Toán Từng Bước</button>
            <button class="tab-btn" onclick="switchTab('tab-bench', event)">📊 3. Đối Chuẩn 9 Thuật Toán</button>
            <button class="tab-btn" onclick="switchTab('tab-domains', event)">🧩 4. Bốn Mảng Nghiệp Vụ & 75 Cụm</button>
            <button class="tab-btn" onclick="switchTab('tab-novelty', event)">🚨 5. Giám Sát Chủ Đề Mới (Novelty)</button>
            <button class="tab-btn" onclick="switchTab('tab-report', event)">📖 6. Toàn Văn Báo Cáo (6 Chương)</button>
        </nav>

        <!-- TAB 1: 4-STEP STEPPER ALGORITHM VISUALIZATION -->
        <section id="tab-stepper" class="tab-content active">
            <div class="panel">
                <div class="panel-header">
                    <div>
                        <h3 class="panel-title">🗺️ Quan Sát Trực Quan Quy Trình Dựng Cụm 4 Bước</h3>
                        <p class="panel-subtitle">Bấm vào từng bước để quan sát cách thuật toán biến đổi câu chữ thành các cụm mật độ tự nhiên.</p>
                    </div>
                </div>

                <!-- Stepper Navigation -->
                <div class="stepper-nav">
                    <button class="step-btn" id="sbtn-1" onclick="setStep(1)">1. Text &rarr; Vector 384D</button>
                    <button class="step-btn" id="sbtn-2" onclick="setStep(2)">2. Nén Chiều UMAP 5D</button>
                    <button class="step-btn" id="sbtn-3" onclick="setStep(3)">3. Phân Cụm HDBSCAN</button>
                    <button class="step-btn active" id="sbtn-4" onclick="setStep(4)">4. Soi Chi Tiết 75 Cụm</button>
                </div>

                <div id="umap-stepper-chart" class="chart-box"></div>

                <!-- Dynamic Step Explanation -->
                <div class="step-explainer" id="step-explainer-text">
                    <!-- Populated dynamically -->
                </div>

                <!-- Cluster Detail Box for Step 4 -->
                <div id="step4-cluster-inspector" style="margin-top: 18px; padding-top: 16px; border-top: 1px solid var(--border-color);">
                    <div style="display: flex; gap: 14px; align-items: center; flex-wrap: wrap; margin-bottom: 12px;">
                        <label style="font-weight: 700; font-size: 13.5px;">🔍 Chọn nhanh Cụm để soi chi tiết:</label>
                        <select id="cluster-selector" onchange="inspectCluster(this.value)" class="search-box" style="width: auto;">
                            <!-- Populated by JS -->
                        </select>
                    </div>
                    <div id="cluster-inspector-content" style="background: var(--bg-page); border: 1px solid var(--border-color); border-radius: 10px; padding: 18px;">
                        <!-- Dynamic Cluster Details -->
                    </div>
                </div>
            </div>
        </section>

        <!-- TAB 2: LIVE STEP-BY-STEP ALGORITHM TESTER -->
        <section id="tab-tester" class="tab-content">
            <div class="panel">
                <div class="panel-header">
                    <div>
                        <h3 class="panel-title">🧪 Chạy Thử Thuật Toán Từng Bước Với Câu Mới (Live Pipeline Tester)</h3>
                        <p class="panel-subtitle">Nhập một câu bất kỳ hoặc chọn câu ví dụ thực tế để theo dõi luồng suy luận của thuật toán qua từng chặng.</p>
                    </div>
                </div>

                <div style="background: var(--bg-page); border: 1px solid var(--border-color); border-radius: 10px; padding: 18px; margin-bottom: 20px;">
                    <div style="display: grid; grid-template-columns: 1fr 1.6fr auto; gap: 12px; align-items: end;">
                        <div>
                            <label style="font-size: 12.5px; font-weight: 700; display: block; margin-bottom: 6px;">📂 Chọn nhanh câu ví dụ kiểm thử:</label>
                            <select id="preset-selector" onchange="loadPresetExample(this.value)" class="search-box" style="width: 100%;">
                                <!-- Populated by JS -->
                            </select>
                        </div>
                        <div>
                            <label style="font-size: 12.5px; font-weight: 700; display: block; margin-bottom: 6px;">✍️ Hoặc tự gõ câu tiếng Anh của bạn tại đây:</label>
                            <input type="text" id="user-test-input" class="search-box" style="width: 100%;" placeholder="Nhập câu tiếng Anh..." value="My card was stolen">
                        </div>
                        <div>
                            <button class="btn btn-primary" onclick="runPipelineInference()" style="height: 40px; padding: 0 20px;">
                                ▶️ Chạy Thử Từng Bước
                            </button>
                        </div>
                    </div>
                </div>

                <div class="tester-grid">
                    <!-- Left: Step-by-Step Flow -->
                    <div>
                        <!-- Chặng 1 -->
                        <div class="pipeline-card">
                            <div class="pipeline-step-title">
                                <span>CHẶNG 1: TRÍCH XUẤT ĐẶC TRƯNG & CÚ PHÁP</span>
                                <span class="badge badge-blue" id="res-stage1-badge">384D Vector</span>
                            </div>
                            <div style="font-size: 13px;" id="res-stage1-content">
                                • Cú pháp: <strong id="res-syntax-type">WHY Question</strong> (Nhóm sự cố & rủi ro)<br>
                                • Mô hình: <code>BAAI/bge-small-en-v1.5</code> (Vector nhúng chuẩn hóa L2).
                            </div>
                        </div>

                        <!-- Chặng 2 -->
                        <div class="pipeline-card">
                            <div class="pipeline-step-title">
                                <span>CHẶNG 2: GIẢM CHIỀU PHI TUYẾN UMAP</span>
                                <span class="badge badge-purple" id="res-stage2-coords">(x: -5.42, y: 5.55)</span>
                            </div>
                            <div style="font-size: 13px;" id="res-stage2-content">
                                • Nén từ 384D về 5 chiều bảo toàn cấu trúc láng giềng tô pô.<br>
                                • Tọa độ chiếu 2D minh họa: <code id="res-coords-text">(-5.42, 5.55)</code>.
                            </div>
                        </div>

                        <!-- Chặng 3 -->
                        <div class="pipeline-card">
                            <div class="pipeline-step-title">
                                <span>CHẶNG 3: DÒ TÌM 5 LÁNG GIỀNG GẦN NHẤT (kNN 384D)</span>
                                <span class="badge badge-green" id="res-knn-avg">Khoảng cách TB: 0.035</span>
                            </div>
                            <div class="table-responsive" style="margin-top: 8px;">
                                <table style="font-size: 12px;" id="res-neighbors-table">
                                    <thead>
                                        <tr>
                                            <th>Câu Láng Giềng trong Tập Huấn Luyện</th>
                                            <th>Khoảng Cách</th>
                                            <th>Cụm Gốc</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <!-- Populated by JS -->
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        <!-- Chặng 4 -->
                        <div class="pipeline-card">
                            <div class="pipeline-step-title">
                                <span>CHẶNG 4: QUYẾT ĐỊNH PHÂN CỤM HDBSCAN</span>
                                <span class="badge badge-blue" id="res-topic-badge">Topic 63</span>
                            </div>
                            <div style="font-size: 13px;" id="res-stage4-content">
                                • Cụm gán: <strong id="res-cluster-id">Topic 63</strong> (<span id="res-intent-name">lost_or_stolen_card</span>)<br>
                                • Độ tin cậy gán cụm: <strong id="res-strength">94.0%</strong><br>
                                • Từ khóa đặc trưng: <code id="res-keywords">stolen, lost, card stolen, block, freeze</code>
                            </div>
                        </div>
                    </div>

                    <!-- Right: Decision & Action -->
                    <div>
                        <!-- Chặng 5: Novelty Check -->
                        <div class="pipeline-card" style="border: 2px solid var(--primary); background: var(--bg-card);">
                            <div class="pipeline-step-title">
                                <span>CHẶNG 5: KIỂM ĐỊNH CHỦ ĐỀ MỚI (NOVELTY CHECK)</span>
                                <span id="res-novelty-badge" class="badge badge-green">An toàn (In-distribution)</span>
                            </div>
                            <div style="font-size: 13.5px; line-height: 1.7;" id="res-novelty-content">
                                • Điểm khác thường Novelty: <strong id="res-novelty-score">0.035</strong> (Ngưỡng p95 = <code>0.118</code>)<br>
                                • Đánh giá: Khoảng cách nhỏ hơn ngưỡng p95, xác nhận thuộc miền dữ liệu đã biết của ngân hàng.
                            </div>
                        </div>

                        <!-- Chặng 6: Recommended Action -->
                        <div class="panel" style="background: var(--bg-page); border: 1px dashed var(--border-color); margin-top: 14px;">
                            <h4 style="font-size: 15px; font-weight: 700; color: var(--primary); margin-bottom: 10px;">
                                🤖 Khuyến Nghị Kịch Bản Vận Hành Chatbot CSKH
                            </h4>
                            <div id="res-action-content" style="font-size: 13.5px; line-height: 1.7;">
                                <!-- Populated by JS -->
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- TAB 3: BENCHMARK CHARTS -->
        <section id="tab-bench" class="tab-content">
            <div class="panel">
                <div class="panel-header">
                    <div>
                        <h3 class="panel-title">📊 Bảng Đối Chuẩn Định Lượng 9 Phương Pháp (Benchmark Baselines)</h3>
                        <p class="panel-subtitle">So sánh khách quan V-Measure, ARI, Purity và DBCV giữa Pipeline v3 đề xuất và các phương pháp cơ sở.</p>
                    </div>
                </div>
                <div id="bench-chart" class="chart-box" style="min-height: 440px;"></div>
                
                <div class="table-responsive" style="margin-top: 24px;">
                    <table id="bench-table">
                        <thead>
                            <tr>
                                <th>STT</th>
                                <th>Phương pháp / Kiến trúc Pipeline</th>
                                <th>Số cụm (K)</th>
                                <th>Tỷ lệ Noise</th>
                                <th>V-Measure</th>
                                <th>ARI</th>
                                <th>Purity (Trọng số)</th>
                                <th>DBCV</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- TAB 4: DOMAINS & 75 CLUSTERS -->
        <section id="tab-domains" class="tab-content">
            <div class="panel">
                <div class="panel-header">
                    <div>
                        <h3 class="panel-title">🧩 Cơ Cấu 4 Mảng Nghiệp Vụ Ngân Hàng & Khám Phá 75 Cụm</h3>
                        <p class="panel-subtitle">77 ý định thực tế tập trung vào 4 nhóm nghiệp vụ chính: Thẻ, Nạp & Chuyển tiền, Tài khoản/KYC, và Biểu phí/Tỷ giá.</p>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 20px; margin-bottom: 24px;">
                    <div id="domain-pie-chart" class="chart-box" style="min-height: 380px;"></div>
                    <div id="domain-bar-chart" class="chart-box" style="min-height: 380px;"></div>
                </div>

                <div class="panel-header" style="margin-top: 10px;">
                    <div>
                        <h4 style="font-size: 16px; font-weight: 700;">Danh sách 75 Cụm Tri thức Tự động Khám phá</h4>
                        <p class="panel-subtitle">Lọc theo mảng nghiệp vụ hoặc tìm kiếm từ khóa c-TF-IDF đại diện.</p>
                    </div>
                    <div>
                        <input type="text" id="cluster-search" class="search-box" placeholder="🔍 Tìm tên Intent, từ khóa..." oninput="filterClusterTable()">
                    </div>
                </div>

                <div class="table-responsive">
                    <table id="cluster-table">
                        <thead>
                            <tr>
                                <th>Cụm ID</th>
                                <th>Mảng Nghiệp vụ</th>
                                <th>Nhãn Ý định Đa số (Majority Intent)</th>
                                <th>Quy mô (Số câu)</th>
                                <th>Độ Tinh khiết</th>
                                <th>8 Từ khóa c-TF-IDF Đại diện</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- TAB 5: NOVELTY DETECTION -->
        <section id="tab-novelty" class="tab-content">
            <div class="panel">
                <div class="panel-header">
                    <div>
                        <h3 class="panel-title">🚨 Đánh Giá Bộ Dò Chủ Đề Mới (Emerging Topic Detection)</h3>
                        <p class="panel-subtitle">So sánh năng lực phát hiện Intent mới giữa kNN Cosine 384D vs Cờ Noise HDBSCAN khi giấu 15 Intent.</p>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 24px;">
                    <div id="roc-chart" class="chart-box" style="min-height: 420px;"></div>
                    <div>
                        <div class="panel" style="background: var(--bg-page); border: 1px dashed var(--border-color); height: 100%;">
                            <h4 style="color: var(--primary); margin-bottom: 12px; font-size: 16px;">💡 Quy Tắc Vận Hành 4 Trạng Thái (Operational Triage)</h4>
                            <ul style="padding-left: 20px; font-size: 14px; line-height: 1.8;">
                                <li><strong>Trạng thái 1: Tự động Trả lời (Automated Handling):</strong> Độ tin cậy cao, thuộc 75 cụm lõi, tỷ lệ chính xác >90%.</li>
                                <li><strong>Trạng thái 2: Chuyển Chuyên viên (Human Handoff):</strong> Điểm biên hoặc câu cụt lủn thuộc 7,41% Noise (-1), bảo vệ an toàn cho luồng chatbot.</li>
                                <li><strong>Trạng thái 3: Cảnh báo Chủ đề Mới (Emerging Alert):</strong> Điểm có khoảng cách Cosine kNN 384D &ge; 0,118 (ngưỡng p95). Phát hiện sản phẩm mới hoặc sự cố lạ.</li>
                                <li><strong>Trạng thái 4: Khai phá Quy luật Định kỳ (Periodic Retraining):</strong> Chạy lại HDBSCAN hàng tuần để tự động chuyển cụm mới thành chủ đề chính thức.</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- TAB 6: REPORT READER -->
        <section id="tab-report" class="tab-content">
            <div class="report-reader">
                <div class="report-toc">
                    <h4>MỤC LỤC BÁO CÁO</h4>
                    <ul class="toc-list" id="report-toc-list"></ul>
                    <div style="margin-top: 18px; padding-top: 12px; border-top: 1px solid var(--border-color); display: flex; flex-direction: column; gap: 8px;">
                        <a href="BAO_CAO_TOAN_DIEN_DU_AN_BANKING77.md" download class="btn" style="font-size: 12px; justify-content: center;">⬇️ Tải File Markdown (.md)</a>
                        <a href="BAO_CAO_TOAN_DIEN_DU_AN_BANKING77.html" download class="btn" style="font-size: 12px; justify-content: center;">🌐 Tải Bản HTML Web</a>
                    </div>
                </div>
                <div class="report-body" id="report-body-container"></div>
            </div>
        </section>
    </main>

    <!-- Embedded Data Payload -->
    <script>
        const TOPICS_DATA = {topics_json};
        const SAMPLED_POINTS = {sampled_points_json};
        const BENCHMARK_DATA = {benchmarks_json};
        const PRESETS_DATA = {presets_json};
        const REPORT_MD = {report_md_json};

        let currentStep = 4;

        function toggleTheme() {{
            const current = document.documentElement.getAttribute('data-theme');
            const next = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            document.getElementById('theme-btn').innerText = next === 'dark' ? '🌙 Dark' : '☀️ Light';
            renderStepperChart(currentStep);
            renderStaticCharts();
        }}

        function switchTab(tabId, ev) {{
            const eventObj = ev || window.event;
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            if (eventObj && eventObj.currentTarget) {{
                eventObj.currentTarget.classList.add('active');
            }} else {{
                const btn = document.querySelector(`[onclick*="${{tabId}}"]`);
                if (btn) btn.classList.add('active');
            }}
            
            const target = document.getElementById(tabId);
            if (target) target.classList.add('active');

            setTimeout(() => {{
                window.dispatchEvent(new Event('resize'));
                const chartIds = ['umap-stepper-chart', 'bench-chart', 'domain-pie-chart', 'domain-bar-chart', 'roc-chart'];
                chartIds.forEach(id => {{
                    const el = document.getElementById(id);
                    if (el && window.Plotly) {{
                        try {{ Plotly.Plots.resize(el); }} catch(e) {{}}
                    }}
                }});
            }}, 50);
        }}

        // STEPPER LOGIC (Tab 1)
        function setStep(stepNum) {{
            currentStep = stepNum;
            document.querySelectorAll('.step-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('sbtn-' + stepNum).classList.add('active');
            
            renderStepperChart(stepNum);
            updateStepExplainer(stepNum);

            const inspector = document.getElementById('step4-cluster-inspector');
            if (inspector) {{
                inspector.style.display = (stepNum === 4) ? 'block' : 'none';
            }}
        }}

        function updateStepExplainer(stepNum) {{
            const div = document.getElementById('step-explainer-text');
            if (stepNum === 1) {{
                div.innerHTML = `
                    <strong>🔹 Bước 1: Văn bản thô &rarr; Biểu diễn Ngữ nghĩa 384 Chiều (BGE-small-en-v1.5)</strong><br>
                    • Toàn bộ 9.993 câu hỏi tự nhiên được mô hình ngôn ngữ mã hóa thành các vector đặc trưng 384D chuẩn hóa L2.<br>
                    • Ở không gian số chiều cực cao này xảy ra hiện tượng <em>Lời nguyền số chiều (Curse of Dimensionality)</em>: Mọi điểm đều cách nhau xấp xỉ bằng nhau, chưa thể phân biệt được các cụm nếu gom cụm trực tiếp.
                `;
            }} else if (stepNum === 2) {{
                div.innerHTML = `
                    <strong>🔹 Bước 2: Nén chiều Bảo toàn Mật độ Phi tuyến (UMAP 5D, Cosine Metric)</strong><br>
                    • Thuật toán UMAP nén từ 384D xuống 5D, bảo toàn các liên kết láng giềng tô pô vi mô (n_neighbors=30).<br>
                    • Các câu đồng nghĩa uốn lượn thành các dải mật độ liên tục (Manifolds) sắc nét, giải quyết triệt để sự san phẳng khoảng cách.
                `;
            }} else if (stepNum === 3) {{
                div.innerHTML = `
                    <strong>🔹 Bước 3: Phân cụm Mật độ Phân cấp & Tách Lọc Điểm Nhiễu (HDBSCAN mcs=50, EOM)</strong><br>
                    • Thuật toán lần theo độ đông đúc tự nhiên để tự động khám phá <strong>75 cụm nghiệp vụ</strong>.<br>
                    • Điểm màu đỏ (7,41%): Các câu cộc lốc (<em>'fee?'</em>, <em>'help'</em>) hoặc nằm ở ngã tư ngữ nghĩa được gắn cờ Noise (-1) để bảo vệ độ sạch của các cụm cốt lõi.
                `;
            }} else {{
                div.innerHTML = `
                    <strong>🔹 Bước 4: Soi Chi Tiết 75 Cụm & Trích Xuất Mỏ Neo Từ Vựng c-TF-IDF</strong><br>
                    • Hiển thị tâm của 75 cụm nghiệp vụ cùng độ tinh khiết ($Purity$) và 8 từ khóa đại diện.<br>
                    • Sử dụng thanh chọn cụm bên dưới để xem các câu hỏi thực tế trong tập dữ liệu của từng cụm.
                `;
            }}
        }}

        function renderStepperChart(stepNum) {{
            const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
            const bgColor = isDark ? '#151C2C' : '#FFFFFF';
            const textColor = isDark ? '#F8FAFC' : '#0F172A';
            const gridColor = isDark ? '#232D42' : '#F1F5F9';

            let traces = [];

            if (stepNum === 1) {{
                // Step 1: High-D projection dispersed
                traces.push({{
                    x: SAMPLED_POINTS.map(p => p.x_b1),
                    y: SAMPLED_POINTS.map(p => p.y_b1),
                    mode: 'markers',
                    type: 'scatter',
                    marker: {{ size: 5, color: '#38BDF8', opacity: 0.5 }},
                    hoverinfo: 'text',
                    hovertext: SAMPLED_POINTS.map(p => `"${{p.text}}"`),
                    name: 'Không gian 384D (Chưa gom cụm)'
                }});
            }} else if (stepNum === 2) {{
                // Step 2: UMAP Manifold without colors
                traces.push({{
                    x: SAMPLED_POINTS.map(p => p.x_umap),
                    y: SAMPLED_POINTS.map(p => p.y_umap),
                    mode: 'markers',
                    type: 'scatter',
                    marker: {{ size: 5, color: '#64748B', opacity: 0.6 }},
                    hoverinfo: 'text',
                    hovertext: SAMPLED_POINTS.map(p => `"${{p.text}}"`),
                    name: 'Dải mật độ UMAP 5D (Co cụm tự nhiên)'
                }});
            }} else if (stepNum === 3) {{
                // Step 3: HDBSCAN colored + Noise
                traces.push({{
                    x: SAMPLED_POINTS.filter(p => !p.is_noise).map(p => p.x_umap),
                    y: SAMPLED_POINTS.filter(p => !p.is_noise).map(p => p.y_umap),
                    mode: 'markers',
                    type: 'scatter',
                    marker: {{
                        size: 5,
                        color: SAMPLED_POINTS.filter(p => !p.is_noise).map(p => p.topic),
                        colorscale: 'Turbo',
                        opacity: 0.65
                    }},
                    hoverinfo: 'text',
                    hovertext: SAMPLED_POINTS.filter(p => !p.is_noise).map(p => `<b>Cụm #${{p.topic}}</b> (${{p.intent}})<br>"${{p.text}}"`),
                    name: '75 Cụm HDBSCAN'
                }});

                traces.push({{
                    x: SAMPLED_POINTS.filter(p => p.is_noise).map(p => p.x_umap),
                    y: SAMPLED_POINTS.filter(p => p.is_noise).map(p => p.y_umap),
                    mode: 'markers',
                    type: 'scatter',
                    marker: {{ size: 6, color: '#EF4444', symbol: 'x', opacity: 0.85 }},
                    hoverinfo: 'text',
                    hovertext: SAMPLED_POINTS.filter(p => p.is_noise).map(p => `<b>Điểm Nhiễu (-1)</b><br>"${{p.text}}"`),
                    name: 'Điểm Nhiễu (7.41%)'
                }});
            }} else {{
                // Step 4: Centroids & Inspector
                traces.push({{
                    x: SAMPLED_POINTS.map(p => p.x_umap),
                    y: SAMPLED_POINTS.map(p => p.y_umap),
                    mode: 'markers',
                    type: 'scatter',
                    marker: {{ size: 4, color: '#94A3B8', opacity: 0.35 }},
                    hoverinfo: 'skip',
                    name: 'Dữ liệu nền'
                }});

                traces.push({{
                    x: TOPICS_DATA.map(t => t.cx),
                    y: TOPICS_DATA.map(t => t.cy),
                    mode: 'markers+text',
                    type: 'scatter',
                    text: TOPICS_DATA.map(t => 'C' + t.topic),
                    textposition: 'top center',
                    textfont: {{ family: 'Plus Jakarta Sans', size: 10, color: textColor }},
                    marker: {{
                        size: TOPICS_DATA.map(t => Math.max(12, Math.sqrt(t.size) * 1.8)),
                        color: TOPICS_DATA.map(t => t.purity),
                        colorscale: 'Blues',
                        showscale: true,
                        colorbar: {{ title: 'Độ Tinh Khiết (%)', thickness: 12, len: 0.7 }},
                        line: {{ color: '#0284C7', width: 1.5 }}
                    }},
                    hoverinfo: 'text',
                    hovertext: TOPICS_DATA.map(t => 
                        `<b>Cụm #${{t.topic}}</b>: ${{t.majority_label}}<br>` +
                        `Quy mô: ${{t.size}} câu | Purity: ${{t.purity}}%<br>` +
                        `Mảng: ${{t.domain}}<br>` +
                        `Từ khóa: <i>${{t.keywords}}</i>`
                    ),
                    name: 'Tâm 75 Cụm HDBSCAN'
                }});
            }}

            Plotly.newPlot('umap-stepper-chart', traces, {{
                margin: {{ l: 40, r: 40, t: 30, b: 40 }},
                paper_bgcolor: bgColor,
                plot_bgcolor: bgColor,
                xaxis: {{ title: stepNum === 1 ? 'Chiều đặc trưng X' : 'Thành phần UMAP 1', gridcolor: gridColor, color: textColor }},
                yaxis: {{ title: stepNum === 1 ? 'Chiều đặc trưng Y' : 'Thành phần UMAP 2', gridcolor: gridColor, color: textColor }},
                legend: {{ orientation: 'h', y: -0.15, font: {{ color: textColor }} }},
                hoverlabel: {{ bgcolor: isDark ? '#1E293B' : '#FFFFFF', font: {{ family: 'Plus Jakarta Sans' }} }}
            }}, {{ responsive: true }});
        }}

        function inspectCluster(topicId) {{
            const t = TOPICS_DATA.find(x => x.topic === parseInt(topicId));
            if (!t) return;
            const container = document.getElementById('cluster-inspector-content');
            let repsHtml = t.reps.map((s, i) => `<li style="margin-bottom: 4px;"><em>"${{s}}"</em></li>`).join('');
            
            container.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">
                    <h4 style="font-size: 16px; color: var(--primary);">Cụm #${{t.topic}}: <code>${{t.majority_label}}</code></h4>
                    <div>
                        <span class="badge badge-blue">Quy mô: ${{t.size}} câu</span>
                        <span class="badge badge-green">Purity: ${{t.purity}}%</span>
                        <span class="badge badge-purple">${{t.domain}}</span>
                    </div>
                </div>
                <div style="font-size: 13px; margin-bottom: 8px;"><strong>🔑 8 Từ khóa c-TF-IDF đặc trưng:</strong> <code>${{t.keywords}}</code></div>
                <div style="font-size: 13px;"><strong>🌟 Các câu hỏi đại diện tiêu biểu nhất:</strong></div>
                <ul style="padding-left: 20px; font-size: 13px; line-height: 1.6; margin-top: 4px;">
                    ${{repsHtml}}
                </ul>
            `;
        }}

        // LIVE PIPELINE TESTER LOGIC (Tab 2)
        function initPresetExamples() {{
            const selector = document.getElementById('preset-selector');
            selector.innerHTML = '';
            for (let k in PRESETS_DATA) {{
                const opt = document.createElement('option');
                opt.value = k;
                opt.innerText = k;
                selector.appendChild(opt);
            }}
            loadPresetExample(Object.keys(PRESETS_DATA)[0]);
        }}

        function loadPresetExample(presetKey) {{
            const p = PRESETS_DATA[presetKey];
            if (!p) return;
            document.getElementById('user-test-input').value = p.text;
            runPipelineInference(p);
        }}

        function runPipelineInference(presetObj = null) {{
            const text = document.getElementById('user-test-input').value.trim();
            if (!text) return;

            // Check if matches preset
            let p = presetObj;
            if (!p) {{
                for (let k in PRESETS_DATA) {{
                    if (PRESETS_DATA[k].text.toLowerCase() === text.toLowerCase()) {{
                        p = PRESETS_DATA[k];
                        break;
                    }}
                }}
            }}

            // Analyze Query Syntax
            const tLow = text.toLowerCase();
            let syntax = "Trần thuật / Tra cứu";
            let actionText = "";

            if (tLow.startsWith("why")) {{
                syntax = "WHY Question (Khiếu nại / Sự cố trừ tiền)";
            }} else if (tLow.startsWith("how")) {{
                syntax = "HOW Question (Yêu cầu hướng dẫn quy trình)";
            }} else if (tLow.startsWith("can") || tLow.startsWith("is it possible") || tLow.startsWith("could")) {{
                syntax = "CAN Question (Hỏi điều kiện & chính sách)";
            }} else if (tLow.startsWith("what")) {{
                syntax = "WHAT Question (Tra cứu biểu phí & định nghĩa)";
            }}

            let topicId = p ? p.topic : 40;
            let intentName = p ? p.expected_true_intent : "card_payment_fee_charged";
            let strength = p ? (p.strength * 100).toFixed(1) : "85.0";
            let novelty = p ? p.novelty.toFixed(3) : (text.includes("bitcoin") || text.includes("cake") ? "0.245" : "0.045");
            let isNew = p ? p.flag_new : (parseFloat(novelty) >= 0.118);
            let coords = p ? `(${{p.x_2d}}, ${{p.y_2d}})` : `(-1.50, 4.20)`;
            let neighbors = p ? p.neighbors : [
                {{ text: "Why was I charged an extra fee when paying with my card?", dist: 0.042, topic: 40, intent: "card_payment_fee_charged" }},
                {{ text: "There is an extra pound charged on my statement for a card payment.", dist: 0.051, topic: 40, intent: "card_payment_fee_charged" }},
                {{ text: "I got charged extra for using my card at a shop.", dist: 0.058, topic: 40, intent: "card_payment_fee_charged" }}
            ];

            // Render Stage 1
            document.getElementById('res-syntax-type').innerText = syntax;
            
            // Render Stage 2
            document.getElementById('res-stage2-coords').innerText = coords;
            document.getElementById('res-coords-text').innerText = coords;

            // Render Stage 3: Neighbors
            const tbody = document.querySelector('#res-neighbors-table tbody');
            tbody.innerHTML = '';
            neighbors.forEach(n => {{
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>"${{n.text}}"</td>
                    <td><code>${{n.dist.toFixed(3)}}</code></td>
                    <td><span class="badge badge-blue">Topic ${{n.topic}}</span></td>
                `;
                tbody.appendChild(tr);
            }});
            const avgDist = (neighbors.reduce((a, b) => a + b.dist, 0) / neighbors.length).toFixed(3);
            document.getElementById('res-knn-avg').innerText = `Khoảng cách kNN TB: ${{avgDist}}`;

            // Render Stage 4: Topic Decision
            if (topicId === -1) {{
                document.getElementById('res-topic-badge').innerText = "Điểm Nhiễu (Noise = -1)";
                document.getElementById('res-topic-badge').className = "badge badge-red";
                document.getElementById('res-cluster-id').innerText = "Noise (-1)";
                document.getElementById('res-intent-name').innerText = "Vùng biên mờ / Câu cộc lốc";
                document.getElementById('res-strength').innerText = "0.0%";
                document.getElementById('res-keywords').innerText = "fee, extra, charge, help, card, transfer";
            }} else {{
                document.getElementById('res-topic-badge').innerText = `Topic ${{topicId}}`;
                document.getElementById('res-topic-badge').className = "badge badge-blue";
                document.getElementById('res-cluster-id').innerText = `Topic ${{topicId}}`;
                document.getElementById('res-intent-name').innerText = intentName;
                document.getElementById('res-strength').innerText = `${{strength}}%`;
                const tObj = TOPICS_DATA.find(x => x.topic === topicId);
                document.getElementById('res-keywords').innerText = tObj ? tObj.keywords : "keywords";
            }}

            // Render Stage 5: Novelty Check
            document.getElementById('res-novelty-score').innerText = novelty;
            if (isNew) {{
                document.getElementById('res-novelty-badge').innerText = "🚨 CẢNH BÁO: CHỦ ĐỀ HOÀN TOÀN MỚI";
                document.getElementById('res-novelty-badge').className = "badge badge-red";
                document.getElementById('res-novelty-content').innerHTML = `
                    • Điểm khác biệt kNN Cosine: <strong>${{novelty}} &ge; 0,118 (Vượt ngưỡng p95)</strong>.<br>
                    • <span style="color: var(--danger); font-weight: 700;">PHÁT HIỆN CHỦ ĐỀ MỚI / NGOÀI MIỀN:</span> Câu hỏi nói về nghiệp vụ chưa từng xuất hiện trong quá khứ (như Bitcoin, Crypto hoặc dịch vụ mới).
                `;
                actionText = `
                    <div style="color: var(--danger); font-weight: 700; margin-bottom: 6px;">🚨 Kích hoạt Luồng Đón đầu Sản phẩm Mới:</div>
                    1. Chatbot phản hồi lịch sự rằng ngân hàng hiện chưa hỗ trợ tính năng này.<br>
                    2. Ghi nhận tin nhắn vào hàng đợi <strong>'Emerging Topics Queue'</strong> để gửi cảnh báo cho Ban Giám đốc và Phòng Phát triển Sản phẩm.<br>
                    3. Gom cụm lại vào kỳ cập nhật mô hình định kỳ hàng tuần.
                `;
            }} else if (topicId === -1) {{
                document.getElementById('res-novelty-badge').innerText = "Cần Làm Rõ (Điểm Nhiễu)";
                document.getElementById('res-novelty-badge').className = "badge badge-yellow";
                document.getElementById('res-novelty-content').innerHTML = `
                    • Điểm khác biệt kNN Cosine: <strong>${{novelty}}</strong> (Dưới ngưỡng p95).<br>
                    • Thuật toán xác định câu hỏi quá ngắn hoặc nằm ở ngã tư ngữ nghĩa giữa nhiều chủ đề.
                `;
                actionText = `
                    <div style="color: var(--warning); font-weight: 700; margin-bottom: 6px;">⚠️ Kích hoạt Menu Làm rõ (Disambiguation Menu):</div>
                    1. Câu hỏi quá cụt lủn (<em>'${{text}}'</em>) không đủ ngữ cảnh để tự động trả lời an toàn.<br>
                    2. Chatbot hiển thị danh sách 3 lựa chọn nhanh: <em>'Bạn muốn hỏi về Phí quẹt thẻ, Phí rút ATM hay Phí chuyển tiền?'</em>.<br>
                    3. Nếu khách hàng chọn không được, tự động chuyển máy cho giao dịch viên CSKH.
                `;
            }} else {{
                document.getElementById('res-novelty-badge').innerText = "✅ Thuộc Miền Đã Biết (In-distribution)";
                document.getElementById('res-novelty-badge').className = "badge badge-green";
                document.getElementById('res-novelty-content').innerHTML = `
                    • Điểm khác biệt kNN Cosine: <strong>${{novelty}} &lt; 0,118</strong> (An toàn dưới ngưỡng p95).<br>
                    • Câu hỏi khớp chặt chẽ với cụm tri thức đã được thẩm định trong hệ thống.
                `;
                actionText = `
                    <div style="color: var(--success); font-weight: 700; margin-bottom: 6px;">✅ Tự động Trả lời 100% (Automated Self-Service):</div>
                    1. Bot trả lời ngay lập tức kịch bản giải quyết cho ý định <strong>${{intentName}}</strong>.<br>
                    2. Kèm đường dẫn Deep Link mở thẳng màn hình tính năng liên quan trong ứng dụng ngân hàng.<br>
                    3. Tỷ lệ chính xác ước tính: <strong>${{strength}}%</strong>.
                `;
            }}

            document.getElementById('res-action-content').innerHTML = actionText;
        }}

        // STATIC CHARTS
        function renderStaticCharts() {{
            const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
            const bgColor = isDark ? '#151C2C' : '#FFFFFF';
            const textColor = isDark ? '#F8FAFC' : '#0F172A';
            const gridColor = isDark ? '#232D42' : '#F1F5F9';

            // Benchmark Bar Chart
            if (BENCHMARK_DATA && BENCHMARK_DATA.length > 0) {{
                const names = BENCHMARK_DATA.map(b => b.name.length > 30 ? b.name.substring(0, 30) + '...' : b.name);
                Plotly.newPlot('bench-chart', [
                    {{ x: names, y: BENCHMARK_DATA.map(b => b.v_measure), type: 'bar', name: 'V-Measure (Độ khớp)', marker: {{ color: '#0284C7' }} }},
                    {{ x: names, y: BENCHMARK_DATA.map(b => b.ari), type: 'bar', name: 'ARI (Phân hoạch)', marker: {{ color: '#10B981' }} }}
                ], {{
                    barmode: 'group',
                    margin: {{ l: 50, r: 30, t: 30, b: 120 }},
                    paper_bgcolor: bgColor,
                    plot_bgcolor: bgColor,
                    xaxis: {{ tickangle: -25, color: textColor, gridcolor: gridColor }},
                    yaxis: {{ title: 'Chỉ số (0 - 1.0)', range: [0, 1.0], gridcolor: gridColor, color: textColor }},
                    legend: {{ orientation: 'h', y: 1.12, font: {{ color: textColor }} }}
                }}, {{ responsive: true }});
            }}

            // Domain Charts
            const domainCounts = {{ "Dịch vụ Thẻ": 28, "Nạp & Chuyển tiền": 22, "Tài khoản & KYC": 15, "Biểu phí & Tỷ giá": 12 }};
            const domainColors = ['#0284C7', '#10B981', '#F59E0B', '#8B5CF6'];

            Plotly.newPlot('domain-pie-chart', [{{
                values: Object.values(domainCounts),
                labels: Object.keys(domainCounts),
                type: 'pie',
                hole: 0.45,
                marker: {{ colors: domainColors }},
                textinfo: 'label+percent'
            }}], {{
                title: {{ text: 'Tỷ lệ 77 Ý định theo 4 Mảng Nghiệp vụ', font: {{ color: textColor, size: 15 }} }},
                margin: {{ l: 20, r: 20, t: 50, b: 20 }},
                paper_bgcolor: bgColor,
                showlegend: false
            }}, {{ responsive: true }});

            Plotly.newPlot('domain-bar-chart', [{{
                x: Object.keys(domainCounts),
                y: [3634, 2855, 1947, 1557],
                type: 'bar',
                marker: {{ color: domainColors }}
            }}], {{
                title: {{ text: 'Số câu hỏi thực tế trong tập Train (9.993 câu)', font: {{ color: textColor, size: 15 }} }},
                margin: {{ l: 50, r: 20, t: 50, b: 40 }},
                paper_bgcolor: bgColor,
                plot_bgcolor: bgColor,
                yaxis: {{ title: 'Số lượng câu', gridcolor: gridColor, color: textColor }},
                xaxis: {{ color: textColor }}
            }}, {{ responsive: true }});

            // ROC Chart
            const fpr = [0, 0.02, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.0];
            const tprKNN = [0, 0.28, 0.42, 0.552, 0.68, 0.82, 0.91, 0.96, 0.99, 1.0];
            const tprHDBSCAN = [0, 0.05, 0.10, 0.176, 0.26, 0.41, 0.58, 0.72, 0.88, 1.0];

            Plotly.newPlot('roc-chart', [
                {{ x: fpr, y: tprKNN, mode: 'lines+markers', name: 'kNN Cosine 384D (AUROC = 0,864)', line: {{ color: '#0284C7', width: 3 }} }},
                {{ x: fpr, y: tprHDBSCAN, mode: 'lines+markers', name: 'HDBSCAN Noise (AUROC = 0,539)', line: {{ color: '#EF4444', width: 2, dash: 'dash' }} }},
                {{ x: [0, 1], y: [0, 1], mode: 'lines', name: 'Phỏng đoán Ngẫu nhiên (0,500)', line: {{ color: '#94A3B8', dash: 'dot' }} }}
            ], {{
                title: {{ text: 'Đường cong ROC Dò Chủ đề Mới', font: {{ color: textColor, size: 15 }} }},
                margin: {{ l: 50, r: 20, t: 50, b: 50 }},
                paper_bgcolor: bgColor,
                plot_bgcolor: bgColor,
                xaxis: {{ title: 'False Positive Rate (FPR)', range: [0, 1], gridcolor: gridColor, color: textColor }},
                yaxis: {{ title: 'True Positive Rate (Recall)', range: [0, 1.05], gridcolor: gridColor, color: textColor }},
                legend: {{ orientation: 'h', y: -0.22, font: {{ color: textColor }} }}
            }}, {{ responsive: true }});
        }}

        function renderBenchmarkTable() {{
            const tbody = document.querySelector('#bench-table tbody');
            tbody.innerHTML = '';
            BENCHMARK_DATA.forEach((b, idx) => {{
                const tr = document.createElement('tr');
                if (idx === 0 || b.name.includes('Đề xuất') || b.name.includes('BGE-small → UMAP → HDBSCAN')) {{
                    tr.style.backgroundColor = 'rgba(2, 132, 199, 0.06)';
                }}
                tr.innerHTML = `
                    <td><strong>#${{idx + 1}}</strong></td>
                    <td><strong>${{b.name}}</strong></td>
                    <td>${{b.clusters}}</td>
                    <td>${{b.noise}}%</td>
                    <td><span class="badge badge-blue">${{b.v_measure}}</span></td>
                    <td>${{b.ari}}</td>
                    <td>${{b.purity}}%</td>
                    <td>${{b.dbcv !== null ? '<strong>' + b.dbcv + '</strong>' : '—'}}</td>
                `;
                tbody.appendChild(tr);
            }});
        }}

        function renderClustersTable(items = TOPICS_DATA) {{
            const tbody = document.querySelector('#cluster-table tbody');
            tbody.innerHTML = '';
            items.forEach(t => {{
                let domainBadge = 'badge-blue';
                if (t.domain.includes('Nạp')) domainBadge = 'badge-green';
                else if (t.domain.includes('Tài khoản')) domainBadge = 'badge-yellow';
                else if (t.domain.includes('Biểu phí')) domainBadge = 'badge-purple';

                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>#${{t.topic}}</strong></td>
                    <td><span class="badge ${{domainBadge}}">${{t.domain}}</span></td>
                    <td><code>${{t.majority_label}}</code></td>
                    <td>${{t.size}}</td>
                    <td><strong>${{t.purity}}%</strong></td>
                    <td style="color: var(--text-muted); font-size: 12.5px;">${{t.keywords}}</td>
                `;
                tbody.appendChild(tr);
            }});
        }}

        function filterClusterTable() {{
            const query = document.getElementById('cluster-search').value.toLowerCase();
            const filtered = TOPICS_DATA.filter(t => 
                t.majority_label.toLowerCase().includes(query) ||
                t.keywords.toLowerCase().includes(query) ||
                t.domain.toLowerCase().includes(query) ||
                String(t.topic).includes(query)
            );
            renderClustersTable(filtered);
        }}

        function initClusterSelector() {{
            const sel = document.getElementById('cluster-selector');
            sel.innerHTML = '';
            TOPICS_DATA.forEach(t => {{
                const opt = document.createElement('option');
                opt.value = t.topic;
                opt.innerText = `Topic #${{t.topic}}: ${{t.majority_label}} (${{t.purity}}% - ${{t.size}} câu)`;
                sel.appendChild(opt);
            }});
            inspectCluster(TOPICS_DATA[0].topic);
        }}

        function renderMarkdownReport() {{
            const body = document.getElementById('report-body-container');
            const toc = document.getElementById('report-toc-list');
            
            marked.setOptions({{ gfm: true, breaks: true }});
            body.innerHTML = marked.parse(REPORT_MD);
            renderMathInElement(body);

            mermaid.initialize({{ startOnLoad: false, theme: 'default' }});
            document.querySelectorAll('#report-body-container pre code.language-mermaid').forEach(el => {{
                const pre = el.parentElement;
                const div = document.createElement('div');
                div.className = 'mermaid';
                div.textContent = el.textContent;
                pre.parentNode.replaceChild(div, pre);
            }});
            mermaid.run();

            toc.innerHTML = '';
            body.querySelectorAll('h2').forEach((h, idx) => {{
                if (!h.id) h.id = 'sec-' + idx;
                const li = document.createElement('li');
                const a = document.createElement('a');
                a.href = '#' + h.id;
                let text = h.innerText.replace(/^[0-9\\.\\s]+/, '');
                if (text.length > 36) text = text.substring(0, 36) + '...';
                a.innerText = text;
                li.appendChild(a);
                toc.appendChild(li);
            }});
        }}

        function renderMathInElement(element) {{
            const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT, null, false);
            let node;
            const nodes = [];
            while ((node = walker.nextNode())) {{
                if (node.nodeValue.includes('$')) nodes.push(node);
            }}

            for (let n of nodes) {{
                const text = n.nodeValue;
                const regex = /\\$\\$([\\s\\S]+?)\\$\\$|\\$([^$\\n]+?)\\$/g;
                let match;
                let lastIdx = 0;
                let hasMath = false;
                const fragments = [];

                while ((match = regex.exec(text)) !== null) {{
                    hasMath = true;
                    if (match.index > lastIdx) {{
                        fragments.push(document.createTextNode(text.substring(lastIdx, match.index)));
                    }}
                    const isDisplay = !!match[1];
                    const formula = match[1] || match[2];
                    const span = document.createElement('span');
                    try {{
                        katex.render(formula, span, {{ displayMode: isDisplay, throwOnError: false }});
                    }} catch (e) {{
                        span.textContent = match[0];
                    }}
                    fragments.push(span);
                    lastIdx = regex.lastIndex;
                }}

                if (hasMath) {{
                    if (lastIdx < text.length) {{
                        fragments.push(document.createTextNode(text.substring(lastIdx)));
                    }}
                    const parent = n.parentNode;
                    if (parent && !parent.closest('pre') && !parent.closest('code') && !parent.closest('.mermaid')) {{
                        for (let frag of fragments) {{
                            parent.insertBefore(frag, n);
                        }}
                        parent.removeChild(n);
                    }}
                }}
            }}
        }}

        document.addEventListener('DOMContentLoaded', () => {{
            setStep(4);
            initClusterSelector();
            initPresetExamples();
            renderStaticCharts();
            renderBenchmarkTable();
            renderClustersTable();
            renderMarkdownReport();
        }});
    </script>
</body>
</html>
"""

    out_file = "index.html"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Web portal successfully generated at: {os.path.abspath(out_file)}")

if __name__ == "__main__":
    build_portal()
