# build_github_pages.py
# -*- coding: utf-8 -*-
"""
Script to generate a self-contained, interactive Web Portal (index.html)
for GitHub Pages hosting. Includes Plotly.js charts, 75 clusters explorer,
benchmark visualizations, and the full 6-chapter scientific report.
"""

import os
import json
import pandas as pd
import numpy as np

def build_portal():
    print("Building GitHub Pages Web Portal...")

    # 1. Load Topics
    topics_path = "artifacts/topics_v3.csv"
    if os.path.exists(topics_path):
        df_topics = pd.read_csv(topics_path)
    else:
        df_topics = pd.DataFrame()

    # Determine Domain for each topic based on intent keyword
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

    # 2. Sample UMAP 2D points for interactive scatter plot
    umap_path = "artifacts/umap_2d_states.csv"
    sampled_points = []
    if os.path.exists(umap_path):
        df_umap = pd.read_csv(umap_path)
        # Sample ~1200 points stratified by topic to keep index.html lightweight and responsive
        sample_size = min(1200, len(df_umap))
        df_sample = df_umap.sample(n=sample_size, random_state=42)
        for _, row in df_sample.iterrows():
            sampled_points.append({
                "x": round(float(row["x_umap"]), 3),
                "y": round(float(row["y_umap"]), 3),
                "topic": int(row["topic"]),
                "is_noise": bool(row["is_noise"]),
                "intent": str(row["true_intent"]),
                "text": str(row["text"])[:90] + "..." if len(str(row["text"])) > 90 else str(row["text"])
            })

    # 3. Load Benchmark
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

    # 4. Load Markdown Report
    report_path = "BAO_CAO_TOAN_DIEN_DU_AN_BANKING77.md"
    report_md = ""
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_md = f.read()

    # Convert topics to list of dicts for JSON embedding
    topics_list = []
    if not df_topics.empty:
        for _, r in df_topics.iterrows():
            topics_list.append({
                "topic": int(r["topic"]),
                "size": int(r["size"]),
                "purity": round(float(r["purity"]) * 100, 1),
                "majority_label": str(r["majority_label"]),
                "keywords": str(r["keywords"]),
                "cx": round(float(r["cx"]), 3),
                "cy": round(float(r["cy"]), 3),
                "domain": str(r["domain"])
            })

    # Prepare JSON structures
    topics_json = json.dumps(topics_list, ensure_ascii=False)
    sampled_points_json = json.dumps(sampled_points, ensure_ascii=False)
    benchmarks_json = json.dumps(benchmarks, ensure_ascii=False)
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
            --bg-nav: rgba(255, 255, 255, 0.9);
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
            --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -4px rgba(0,0,0,0.04);
        }}

        [data-theme="dark"] {{
            --bg-page: #0B0F19;
            --bg-card: #151C2C;
            --bg-nav: rgba(21, 28, 44, 0.9);
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
            --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.5);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-page);
            color: var(--text-main);
            line-height: 1.6;
            transition: background-color 0.25s, color 0.25s;
        }}

        /* Header Navigation */
        header {{
            position: sticky;
            top: 0;
            z-index: 1000;
            backdrop-filter: blur(12px);
            background-color: var(--bg-nav);
            border-bottom: 1px solid var(--border-color);
        }}

        .navbar {{
            max-width: 1380px;
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

        .brand-text h1 {{
            font-size: 16px;
            font-weight: 800;
            letter-spacing: -0.3px;
            line-height: 1.2;
        }}

        .brand-text span {{
            font-size: 12px;
            color: var(--text-muted);
            font-weight: 500;
        }}

        .nav-actions {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

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

        .btn:hover {{
            border-color: var(--primary);
            color: var(--primary);
            transform: translateY(-1px);
        }}

        .btn-primary {{
            background: linear-gradient(135deg, #0284C7, #2563EB);
            color: #FFFFFF;
            border: none;
            box-shadow: 0 4px 12px rgba(2, 132, 199, 0.25);
        }}

        .btn-primary:hover {{
            color: #FFFFFF;
            opacity: 0.95;
            box-shadow: 0 6px 16px rgba(2, 132, 199, 0.35);
        }}

        /* Hero Banner */
        .hero {{
            max-width: 1380px;
            margin: 28px auto 20px;
            padding: 0 24px;
        }}

        .hero-card {{
            background: linear-gradient(135deg, rgba(2, 132, 199, 0.08) 0%, rgba(37, 99, 235, 0.04) 100%);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 32px 36px;
            position: relative;
            overflow: hidden;
        }}

        .hero-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            background-color: var(--primary-glow);
            color: var(--primary);
            border-radius: 9999px;
            font-size: 12.5px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }}

        .hero-title {{
            font-size: 26px;
            font-weight: 800;
            letter-spacing: -0.5px;
            line-height: 1.3;
            margin-bottom: 12px;
            max-width: 950px;
        }}

        .hero-desc {{
            font-size: 15px;
            color: var(--text-muted);
            max-width: 900px;
            line-height: 1.6;
            margin-bottom: 24px;
        }}

        /* Metrics Row */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 16px;
        }}

        .kpi-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 18px 20px;
            box-shadow: var(--shadow-sm);
            transition: all 0.2s ease;
        }}

        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
            border-color: var(--primary);
        }}

        .kpi-label {{
            font-size: 12.5px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.4px;
            margin-bottom: 6px;
        }}

        .kpi-value {{
            font-size: 26px;
            font-weight: 800;
            color: var(--text-main);
            letter-spacing: -0.5px;
        }}

        .kpi-subtext {{
            font-size: 12px;
            color: var(--success);
            font-weight: 600;
            margin-top: 4px;
            display: flex;
            align-items: center;
            gap: 4px;
        }}

        /* Main Container & Tabs */
        .main-container {{
            max-width: 1380px;
            margin: 0 auto 60px;
            padding: 0 24px;
        }}

        .tab-nav {{
            display: flex;
            gap: 8px;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 24px;
            overflow-x: auto;
            padding-bottom: 4px;
        }}

        .tab-btn {{
            padding: 12px 20px;
            background: none;
            border: none;
            border-bottom: 2px solid transparent;
            color: var(--text-muted);
            font-size: 14.5px;
            font-weight: 600;
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .tab-btn:hover {{
            color: var(--primary);
        }}

        .tab-btn.active {{
            color: var(--primary);
            border-bottom-color: var(--primary);
        }}

        .tab-content {{
            display: none;
            animation: fadeIn 0.3s ease;
        }}

        .tab-content.active {{
            display: block;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(6px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* Card panels */
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
            margin-bottom: 20px;
            flex-wrap: wrap;
            gap: 12px;
        }}

        .panel-title {{
            font-size: 18px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .panel-subtitle {{
            font-size: 13.5px;
            color: var(--text-muted);
            margin-top: 3px;
        }}

        /* Plot Containers */
        .chart-box {{
            width: 100%;
            min-height: 520px;
            border-radius: 10px;
            overflow: hidden;
        }}

        /* Tables */
        .table-responsive {{
            overflow-x: auto;
            border: 1px solid var(--border-color);
            border-radius: 10px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13.5px;
            text-align: left;
        }}

        th, td {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
        }}

        th {{
            background-color: rgba(2, 132, 199, 0.05);
            font-weight: 700;
            color: var(--primary);
            white-space: nowrap;
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        tr:hover td {{
            background-color: rgba(2, 132, 199, 0.02);
        }}

        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 11.5px;
            font-weight: 700;
        }}

        .badge-blue {{ background-color: rgba(2, 132, 199, 0.12); color: #0284C7; }}
        .badge-green {{ background-color: rgba(16, 185, 129, 0.12); color: #10B981; }}
        .badge-yellow {{ background-color: rgba(245, 158, 11, 0.12); color: #D97706; }}
        .badge-purple {{ background-color: rgba(147, 51, 234, 0.12); color: #9333EA; }}

        /* Search input */
        .search-box {{
            padding: 8px 14px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            background-color: var(--bg-page);
            color: var(--text-main);
            font-size: 13.5px;
            outline: none;
            width: 260px;
            transition: all 0.2s;
        }}

        .search-box:focus {{
            border-color: var(--primary);
            box-shadow: 0 0 0 3px var(--primary-glow);
        }}

        /* Markdown Report Reader */
        .report-reader {{
            display: flex;
            gap: 32px;
            align-items: flex-start;
        }}

        .report-toc {{
            width: 280px;
            flex-shrink: 0;
            position: sticky;
            top: 80px;
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 18px;
            max-height: calc(100vh - 100px);
            overflow-y: auto;
        }}

        .report-toc h4 {{
            font-size: 13px;
            text-transform: uppercase;
            color: var(--text-muted);
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }}

        .toc-list {{
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}

        .toc-list a {{
            color: var(--text-muted);
            text-decoration: none;
            font-size: 13px;
            padding: 6px 10px;
            border-radius: 6px;
            display: block;
            transition: all 0.2s;
        }}

        .toc-list a:hover, .toc-list a.active {{
            color: var(--primary);
            background-color: var(--primary-glow);
            font-weight: 600;
        }}

        .report-body {{
            flex-grow: 1;
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 36px 44px;
            max-width: 980px;
            overflow-x: hidden;
        }}

        .report-body h1 {{ font-size: 26px; color: var(--primary); margin-bottom: 16px; }}
        .report-body h2 {{ font-size: 21px; margin-top: 36px; margin-bottom: 14px; border-bottom: 1px solid var(--border-color); padding-bottom: 8px; }}
        .report-body h3 {{ font-size: 17px; margin-top: 24px; margin-bottom: 10px; color: var(--accent); }}
        .report-body h4 {{ font-size: 15px; margin-top: 18px; margin-bottom: 8px; color: var(--warning); }}
        .report-body p {{ margin-bottom: 14px; line-height: 1.7; }}
        .report-body ul, .report-body ol {{ margin-bottom: 14px; padding-left: 22px; }}
        .report-body li {{ margin-bottom: 5px; }}
        .report-body code {{
            background-color: var(--code-bg);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            color: var(--primary);
        }}
        .report-body pre {{
            background-color: var(--code-bg);
            padding: 16px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            overflow-x: auto;
            margin: 18px 0;
        }}
        .report-body pre code {{
            background: none;
            padding: 0;
            color: var(--text-main);
        }}
        .report-body blockquote {{
            border-left: 4px solid var(--primary);
            background-color: var(--primary-glow);
            padding: 12px 18px;
            border-radius: 0 8px 8px 0;
            margin: 18px 0;
        }}

        /* Responsive */
        @media (max-width: 900px) {{
            .report-reader {{ flex-direction: column; }}
            .report-toc {{ width: 100%; position: static; max-height: none; }}
            .report-body {{ padding: 20px; }}
            .hero-card {{ padding: 20px; }}
            .hero-title {{ font-size: 20px; }}
        }}
    </style>
</head>
<body>

    <!-- Header Navigation -->
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

    <!-- Hero Banner with Key Metrics -->
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
                    <div class="kpi-label">Quy mô Dữ liệu Huấn luyện</div>
                    <div class="kpi-value">9.993</div>
                    <div class="kpi-subtext">✓ 13.073 tổng tin nhắn chuẩn</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Số Cụm Tự Động Khám Phá</div>
                    <div class="kpi-value">75 Cụm</div>
                    <div class="kpi-subtext">✓ Tiệm cận 77 nhãn chuyên gia</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Độ Khớp Nhãn V-Measure</div>
                    <div class="kpi-value">0,833</div>
                    <div class="kpi-subtext">✓ 0,837 trên Test (No Overfitting)</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Chỉ số Mật độ Nội tại DBCV</div>
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

    <!-- Main Content Tabs -->
    <main class="main-container">
        <nav class="tab-nav">
            <button class="tab-btn active" onclick="switchTab('tab-umap', event)">🗺️ Bản đồ 2D UMAP</button>
            <button class="tab-btn" onclick="switchTab('tab-bench', event)">📊 Đối chuẩn 9 Thuật toán</button>
            <button class="tab-btn" onclick="switchTab('tab-domains', event)">🧩 4 Mảng Nghiệp vụ & 75 Cụm</button>
            <button class="tab-btn" onclick="switchTab('tab-novelty', event)">🚨 Dò Chủ đề Mới (Novelty)</button>
            <button class="tab-btn" onclick="switchTab('tab-report', event)">📖 Toàn văn Báo cáo (6 Chương)</button>
        </nav>

        <!-- Tab 1: Interactive UMAP Plot -->
        <section id="tab-umap" class="tab-content active">
            <div class="panel">
                <div class="panel-header">
                    <div>
                        <h3 class="panel-title">🗺️ Bản đồ Không gian Ngữ nghĩa 2D (Interactive UMAP Projection)</h3>
                        <p class="panel-subtitle">Rê chuột vào các điểm để xem câu hỏi, Intent gốc, Cụm HDBSCAN và Từ khóa c-TF-IDF đại diện.</p>
                    </div>
                    <div>
                        <span class="badge badge-blue">75 Cụm Mật độ Cao</span>
                        <span class="badge badge-yellow">7,41% Điểm Nhiễu (Noise)</span>
                    </div>
                </div>
                <div id="umap-chart" class="chart-box"></div>
            </div>
        </section>

        <!-- Tab 2: Benchmark Charts -->
        <section id="tab-bench" class="tab-content">
            <div class="panel">
                <div class="panel-header">
                    <div>
                        <h3 class="panel-title">📊 Bảng Đối chuẩn Định lượng 9 Phương pháp (Benchmark Baselines)</h3>
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
                        <tbody>
                            <!-- Populated by JS -->
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- Tab 3: Domains & 75 Clusters Explorer -->
        <section id="tab-domains" class="tab-content">
            <div class="panel">
                <div class="panel-header">
                    <div>
                        <h3 class="panel-title">🧩 Cơ cấu 4 Mảng Nghiệp vụ Ngân hàng & Khám phá 75 Cụm</h3>
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
                    <div style="display: flex; gap: 10px;">
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
                        <tbody>
                            <!-- Populated by JS -->
                        </tbody>
                    </table>
                </div>
            </div>
        </section>

        <!-- Tab 4: Novelty Detection -->
        <section id="tab-novelty" class="tab-content">
            <div class="panel">
                <div class="panel-header">
                    <div>
                        <h3 class="panel-title">🚨 Đánh giá Bộ Dò Chủ đề Mới (Emerging Topic Detection)</h3>
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

        <!-- Tab 5: Full Scientific Report Reader -->
        <section id="tab-report" class="tab-content">
            <div class="report-reader">
                <div class="report-toc">
                    <h4>MỤC LỤC BÁO CÁO</h4>
                    <ul class="toc-list" id="report-toc-list">
                        <!-- Populated dynamically -->
                    </ul>
                    <div style="margin-top: 18px; pt: 12px; border-top: 1px solid var(--border-color); display: flex; flex-direction: column; gap: 8px;">
                        <a href="BAO_CAO_TOAN_DIEN_DU_AN_BANKING77.md" download class="btn" style="font-size: 12px; justify-content: center;">⬇️ Tải File Markdown (.md)</a>
                        <a href="BAO_CAO_TOAN_DIEN_DU_AN_BANKING77.html" download class="btn" style="font-size: 12px; justify-content: center;">🌐 Tải Bản HTML Web</a>
                    </div>
                </div>
                <div class="report-body" id="report-body-container">
                    <!-- Rendered Markdown goes here -->
                </div>
            </div>
        </section>
    </main>

    <!-- Embedded Data Payload -->
    <script>
        const TOPICS_DATA = {topics_json};
        const SAMPLED_POINTS = {sampled_points_json};
        const BENCHMARK_DATA = {benchmarks_json};
        const REPORT_MD = {report_md_json};

        // Theme Manager
        function toggleTheme() {{
            const current = document.documentElement.getAttribute('data-theme');
            const next = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            document.getElementById('theme-btn').innerText = next === 'dark' ? '🌙 Dark' : '☀️ Light';
            // Redraw charts with matching theme
            renderAllCharts(next);
        }}

        // Tab Navigation
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

            // Trigger resize on Plotly charts after tab is made visible
            setTimeout(() => {{
                window.dispatchEvent(new Event('resize'));
                const chartIds = ['umap-chart', 'bench-chart', 'domain-pie-chart', 'domain-bar-chart', 'roc-chart'];
                chartIds.forEach(id => {{
                    const el = document.getElementById(id);
                    if (el && window.Plotly) {{
                        try {{ Plotly.Plots.resize(el); }} catch(e) {{}}
                    }}
                }});
            }}, 50);
        }}

        // Render Plotly Charts
        function renderAllCharts(theme = 'light') {{
            const isDark = theme === 'dark';
            const bgColor = isDark ? '#151C2C' : '#FFFFFF';
            const textColor = isDark ? '#F8FAFC' : '#0F172A';
            const gridColor = isDark ? '#232D42' : '#F1F5F9';

            // 1. UMAP 2D Plot
            if (TOPICS_DATA && TOPICS_DATA.length > 0) {{
                // Centers trace
                const traceCenters = {{
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
                        `Quy mô: ${{t.size}} câu | Độ tinh khiết: ${{t.purity}}%<br>` +
                        `Mảng: ${{t.domain}}<br>` +
                        `Từ khóa: <i>${{t.keywords}}</i>`
                    ),
                    name: 'Tâm 75 Cụm HDBSCAN'
                }};

                // Sample points trace
                const tracePoints = {{
                    x: SAMPLED_POINTS.filter(p => !p.is_noise).map(p => p.x),
                    y: SAMPLED_POINTS.filter(p => !p.is_noise).map(p => p.y),
                    mode: 'markers',
                    type: 'scatter',
                    marker: {{
                        size: 5,
                        color: '#94A3B8',
                        opacity: 0.45
                    }},
                    hoverinfo: 'text',
                    hovertext: SAMPLED_POINTS.filter(p => !p.is_noise).map(p => 
                        `<b>Cụm #${{p.topic}}</b> (${{p.intent}})<br>"${{p.text}}"`
                    ),
                    name: 'Điểm dữ liệu chuẩn'
                }};

                // Noise trace
                const traceNoise = {{
                    x: SAMPLED_POINTS.filter(p => p.is_noise).map(p => p.x),
                    y: SAMPLED_POINTS.filter(p => p.is_noise).map(p => p.y),
                    mode: 'markers',
                    type: 'scatter',
                    marker: {{
                        size: 6,
                        color: '#EF4444',
                        symbol: 'x',
                        opacity: 0.75
                    }},
                    hoverinfo: 'text',
                    hovertext: SAMPLED_POINTS.filter(p => p.is_noise).map(p => 
                        `<b>Điểm Nhiễu (Noise = -1)</b><br>"${{p.text}}"`
                    ),
                    name: 'Điểm Nhiễu (Noise 7.41%)'
                }};

                Plotly.newPlot('umap-chart', [tracePoints, traceNoise, traceCenters], {{
                    margin: {{ l: 40, r: 40, t: 30, b: 40 }},
                    paper_bgcolor: bgColor,
                    plot_bgcolor: bgColor,
                    xaxis: {{ title: 'Thành phần UMAP 1', gridcolor: gridColor, color: textColor }},
                    yaxis: {{ title: 'Thành phần UMAP 2', gridcolor: gridColor, color: textColor }},
                    legend: {{ orientation: 'h', y: -0.15, font: {{ color: textColor }} }},
                    hoverlabel: {{ bgcolor: isDark ? '#1E293B' : '#FFFFFF', font: {{ family: 'Plus Jakarta Sans' }} }}
                }}, {{ responsive: true, displayModeBar: true }});
            }}

            // 2. Benchmark Bar Chart
            if (BENCHMARK_DATA && BENCHMARK_DATA.length > 0) {{
                const names = BENCHMARK_DATA.map(b => b.name.length > 32 ? b.name.substring(0, 32) + '...' : b.name);
                const traceV = {{
                    x: names,
                    y: BENCHMARK_DATA.map(b => b.v_measure),
                    type: 'bar',
                    name: 'V-Measure (Độ khớp nhãn)',
                    marker: {{ color: '#0284C7' }}
                }};
                const traceARI = {{
                    x: names,
                    y: BENCHMARK_DATA.map(b => b.ari),
                    type: 'bar',
                    name: 'ARI (Tương đồng phân hoạch)',
                    marker: {{ color: '#10B981' }}
                }};
                Plotly.newPlot('bench-chart', [traceV, traceARI], {{
                    barmode: 'group',
                    margin: {{ l: 50, r: 30, t: 30, b: 120 }},
                    paper_bgcolor: bgColor,
                    plot_bgcolor: bgColor,
                    xaxis: {{ tickangle: -25, color: textColor, gridcolor: gridColor }},
                    yaxis: {{ title: 'Chỉ số (0 - 1.0)', range: [0, 1.0], gridcolor: gridColor, color: textColor }},
                    legend: {{ orientation: 'h', y: 1.12, font: {{ color: textColor }} }}
                }}, {{ responsive: true }});
            }}

            // 3. Domain Charts
            const domainCounts = {{
                "Dịch vụ Thẻ": 28,
                "Nạp & Chuyển tiền": 22,
                "Tài khoản & KYC": 15,
                "Biểu phí & Tỷ giá": 12
            }};
            const domainColors = ['#0284C7', '#10B981', '#F59E0B', '#8B5CF6'];

            Plotly.newPlot('domain-pie-chart', [{{
                values: Object.values(domainCounts),
                labels: Object.keys(domainCounts),
                type: 'pie',
                hole: 0.45,
                marker: {{ colors: domainColors }},
                textinfo: 'label+percent',
                insidetextorientation: 'radial'
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

            // 4. ROC Curve Chart
            const fpr = [0, 0.02, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.0];
            const tprKNN = [0, 0.28, 0.42, 0.552, 0.68, 0.82, 0.91, 0.96, 0.99, 1.0];
            const tprHDBSCAN = [0, 0.05, 0.10, 0.176, 0.26, 0.41, 0.58, 0.72, 0.88, 1.0];

            Plotly.newPlot('roc-chart', [
                {{
                    x: fpr, y: tprKNN, mode: 'lines+markers', name: 'kNN Cosine 384D (AUROC = 0,864)',
                    line: {{ color: '#0284C7', width: 3 }}
                }},
                {{
                    x: fpr, y: tprHDBSCAN, mode: 'lines+markers', name: 'HDBSCAN Noise (AUROC = 0,539)',
                    line: {{ color: '#EF4444', width: 2, dash: 'dash' }}
                }},
                {{
                    x: [0, 1], y: [0, 1], mode: 'lines', name: 'Phỏng đoán Ngẫu nhiên (0,500)',
                    line: {{ color: '#94A3B8', dash: 'dot' }}
                }}
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

        // Populate Benchmark Table
        function renderBenchmarkTable() {{
            const tbody = document.querySelector('#bench-table tbody');
            tbody.innerHTML = '';
            BENCHMARK_DATA.forEach((b, idx) => {{
                const tr = document.createElement('tr');
                const isWinner = b.name.includes('Đề xuất') || idx === 0 || b.name.includes('BGE-small → UMAP → HDBSCAN');
                if (isWinner) tr.style.backgroundColor = 'rgba(2, 132, 199, 0.06)';
                
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

        // Populate 75 Clusters Table
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

        // Render Markdown Scientific Report with KaTeX and TOC
        function renderMarkdownReport() {{
            const body = document.getElementById('report-body-container');
            const toc = document.getElementById('report-toc-list');
            
            // Configure marked options
            marked.setOptions({{
                gfm: true,
                breaks: true
            }});

            body.innerHTML = marked.parse(REPORT_MD);

            // Render LaTeX formulas
            renderMathInElement(body);

            // Re-render Mermaid diagrams
            mermaid.initialize({{ startOnLoad: false, theme: 'default' }});
            document.querySelectorAll('#report-body-container pre code.language-mermaid').forEach(el => {{
                const pre = el.parentElement;
                const div = document.createElement('div');
                div.className = 'mermaid';
                div.textContent = el.textContent;
                pre.parentNode.replaceChild(div, pre);
            }});
            mermaid.run();

            // Build Table of Contents
            toc.innerHTML = '';
            const headings = body.querySelectorAll('h2');
            headings.forEach((h, idx) => {{
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

        // Simple KaTeX DOM Walker
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

        // Init on Load
        document.addEventListener('DOMContentLoaded', () => {{
            renderAllCharts('light');
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

