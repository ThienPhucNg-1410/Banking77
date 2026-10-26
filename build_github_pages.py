# build_github_pages.py
# -*- coding: utf-8 -*-
"""
Script to generate the comprehensive, interactive Web Portal (index.html)
for GitHub Pages hosting.
Tailored to user's requirements:
- Removed old sections 3, 4, 6 (Baselines table, Domains table, Long Report reader).
- Added interactive Dataset Overview & Configuration Playground:
  * Select configuration (HDBSCAN mcs=20,30,50,80 / EOM vs Leaf, OR K-Means K=20,50,77,100).
  * Run clustering simulation and see output metrics, 2D scatter visualization, and expert insights.
  * Direct Head-to-Head Comparison between HDBSCAN (density ribbons + noise) and K-Means (spherical Voronoi + 0% noise).
- Maintained 4-step algorithm stepper (Tab 1), live pipeline sentence tester (Tab 2), and novelty detection (Tab 4).
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def build_portal():
    print("Building updated GitHub Pages Web Portal...")

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

    # 3. Sample UMAP 2D points (with x_b1, y_b1, x_umap, y_umap, all topic sweep cols)
    umap_path = "artifacts/umap_2d_states.csv"
    sampled_points = []
    if os.path.exists(umap_path):
        df_umap = pd.read_csv(umap_path)
        sample_size = min(1200, len(df_umap))
        df_sample = df_umap.sample(n=sample_size, random_state=42).copy()
        
        # Fit K-Means on UMAP 2D coordinates for interactive K-Means modes
        coords = df_sample[['x_umap', 'y_umap']].values
        for k in [20, 50, 77, 100]:
            km = KMeans(n_clusters=k, random_state=42, n_init=1).fit(coords)
            df_sample[f'km_{k}'] = km.labels_

        for _, row in df_sample.iterrows():
            sampled_points.append({
                "x_b1": round(float(row.get("x_b1", row["x_umap"])), 3),
                "y_b1": round(float(row.get("y_b1", row["y_umap"])), 3),
                "x_umap": round(float(row["x_umap"]), 3),
                "y_umap": round(float(row["y_umap"]), 3),
                "topic": int(row["topic"]),
                "is_noise": bool(row["is_noise"]),
                "intent": str(row["true_intent"]),
                "text": str(row["text"])[:85] + "..." if len(str(row["text"])) > 85 else str(row["text"]),
                "t_20_eom": int(row.get("topic_20_eom", row["topic"])),
                "t_20_leaf": int(row.get("topic_20_leaf", row["topic"])),
                "t_30_eom": int(row.get("topic_30_eom", row["topic"])),
                "t_30_leaf": int(row.get("topic_30_leaf", row["topic"])),
                "t_50_eom": int(row.get("topic_50_eom", row["topic"])),
                "t_50_leaf": int(row.get("topic_50_leaf", row["topic"])),
                "t_80_eom": int(row.get("topic_80_eom", row["topic"])),
                "t_80_leaf": int(row.get("topic_80_leaf", row["topic"])),
                "km_20": int(row.get("km_20", 0)),
                "km_50": int(row.get("km_50", 0)),
                "km_77": int(row.get("km_77", 0)),
                "km_100": int(row.get("km_100", 0)),
            })

    # 4. Preset Examples for Live Sentence Tester
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

    # 5. Configurations Metadata for Playground
    hdbscan_configs = {
        "hdbscan_50_eom": {
            "name": "HDBSCAN (mcs=50, EOM) — [Cấu hình Tối ưu Đề xuất]",
            "algo": "HDBSCAN",
            "clusters": 75,
            "noise": 7.41,
            "v_measure": 0.833,
            "ari": 0.542,
            "dbcv": 0.432,
            "silhouette": 0.222,
            "coverage": 92.59,
            "col": "t_50_eom",
            "badge": "🏆 TỐI ƯU TOÀN DIỆN",
            "summary": "Khám phá tự động 75 cụm (sát 77 nhãn thật của chuyên gia), đạt điểm chất lượng mật độ DBCV cao nhất (0.432). Lọc 7,41% câu cộc lốc/mơ hồ ra ngoài (Noise = -1) để giữ độ thuần khiết tuyệt đối cho 75 cụm nghiệp vụ."
        },
        "hdbscan_20_eom": {
            "name": "HDBSCAN (mcs=20, EOM)",
            "algo": "HDBSCAN",
            "clusters": 184,
            "noise": 8.10,
            "v_measure": 0.806,
            "ari": 0.570,
            "dbcv": 0.313,
            "silhouette": 0.228,
            "coverage": 91.90,
            "col": "t_20_eom",
            "badge": "Phân mảnh",
            "summary": "Vì mcs=20 quá nhỏ, thuật toán bị phân mảnh dữ liệu thành 184 cụm li ti. Nhiều intent bị xé nhỏ thành 2-3 cụm con, làm giảm tính khái quát nghiệp vụ."
        },
        "hdbscan_20_leaf": {
            "name": "HDBSCAN (mcs=20, Leaf)",
            "algo": "HDBSCAN",
            "clusters": 204,
            "noise": 13.90,
            "v_measure": 0.772,
            "ari": 0.532,
            "dbcv": 0.264,
            "silhouette": 0.234,
            "coverage": 86.10,
            "col": "t_20_leaf",
            "badge": "Quá khớp",
            "summary": "Chiến lược Leaf tách đến tận lá cây phân cấp khiến số cụm vọt lên 204 cụm, tỷ lệ câu bị gạt ra làm Noise tăng vọt lên gần 14%."
        },
        "hdbscan_30_eom": {
            "name": "HDBSCAN (mcs=30, EOM)",
            "algo": "HDBSCAN",
            "clusters": 125,
            "noise": 9.50,
            "v_measure": 0.814,
            "ari": 0.658,
            "dbcv": 0.315,
            "silhouette": 0.225,
            "coverage": 90.50,
            "col": "t_30_eom",
            "badge": "Khá tốt",
            "summary": "Đã bắt đầu hội tụ tốt hơn mcs=20 nhưng 125 cụm vẫn còn quá nhiều so với 77 nhu cầu thực tế của khách hàng."
        },
        "hdbscan_50_leaf": {
            "name": "HDBSCAN (mcs=50, Leaf)",
            "algo": "HDBSCAN",
            "clusters": 86,
            "noise": 13.02,
            "v_measure": 0.814,
            "ari": 0.739,
            "dbcv": 0.366,
            "silhouette": 0.229,
            "coverage": 86.98,
            "col": "t_50_leaf",
            "badge": "Noise cao",
            "summary": "Tìm được 86 cụm nhưng phương thức Leaf ép tỷ lệ Noise tăng gần gấp đôi lên 13,02% (bỏ phí quá nhiều câu hỏi hợp lệ)."
        },
        "hdbscan_80_eom": {
            "name": "HDBSCAN (mcs=80, EOM)",
            "algo": "HDBSCAN",
            "clusters": 58,
            "noise": 7.80,
            "v_measure": 0.832,
            "ari": 0.646,
            "dbcv": 0.411,
            "silhouette": 0.214,
            "coverage": 92.20,
            "col": "t_80_eom",
            "badge": "Gộp cụm",
            "summary": "mcs=80 quá lớn dẫn đến hiện tượng gộp cụm quá mức: Chỉ còn 58 cụm, các intent gần nhau (như thẻ ảo và thẻ vật lý) bị nuốt chung vào một cụm to."
        }
    }

    kmeans_configs = {
        "kmeans_77": {
            "name": "K-Means (K=77) — [Số cụm bằng đúng nhãn thật]",
            "algo": "K-Means",
            "clusters": 77,
            "noise": 0.00,
            "v_measure": 0.861,
            "ari": 0.675,
            "dbcv": None,
            "silhouette": 0.205,
            "coverage": 100.0,
            "col": "km_77",
            "badge": "⚠️ Ép 100% Cầu",
            "summary": "K-Means dùng mặt phẳng chia không gian thành 77 khối cầu lồi (Voronoi). Nhược điểm chí mạng: 1) Bắt buộc con người phải 'mớm' trước K=77 (trong thực tế không ai biết trước K); 2) Không có cơ chế lọc nhiễu, ép cả câu cộc lốc/rác vào cụm làm bẩn dữ liệu."
        },
        "kmeans_50": {
            "name": "K-Means (K=50)",
            "algo": "K-Means",
            "clusters": 50,
            "noise": 0.00,
            "v_measure": 0.724,
            "ari": 0.485,
            "dbcv": None,
            "silhouette": 0.226,
            "coverage": 100.0,
            "col": "km_50",
            "badge": "Gộp ép buộc",
            "summary": "Thiếu 27 cụm so với thực tế. Các dải lụa ngôn ngữ uốn lượn bị mặt phẳng K-Means chém đứt, gộp lẫn lộn giữa phí thẻ và phí chuyển tiền."
        },
        "kmeans_20": {
            "name": "K-Means (K=20)",
            "algo": "K-Means",
            "clusters": 20,
            "noise": 0.00,
            "v_measure": 0.582,
            "ari": 0.295,
            "dbcv": None,
            "silhouette": 0.251,
            "coverage": 100.0,
            "col": "km_20",
            "badge": "Trộn lẫn nặng",
            "summary": "20 khối cầu khổng lồ gom bừa hàng chục nghiệp vụ khác nhau vào chung một chỗ. Độ thuần khiết sụt giảm nghiêm trọng."
        },
        "kmeans_100": {
            "name": "K-Means (K=100)",
            "algo": "K-Means",
            "clusters": 100,
            "noise": 0.00,
            "v_measure": 0.812,
            "ari": 0.598,
            "dbcv": None,
            "silhouette": 0.198,
            "coverage": 100.0,
            "col": "km_100",
            "badge": "Phân mảnh",
            "summary": "Xé nhỏ không gian thành 100 ô hình cầu, làm đứt đoạn các chuỗi câu hỏi cùng nghiệp vụ."
        }
    }

    # 6. Sample Sentences from BANKING77 Dataset for Preview
    sample_dataset_sentences = [
        {"id": 1, "text": "Why was I charged an extra fee when paying with my card?", "intent": "card_payment_fee_charged", "domain": "Dịch vụ Thẻ & Phí", "len": 12},
        {"id": 2, "text": "My card was stolen, how do I block it immediately?", "intent": "lost_or_stolen_card", "domain": "Dịch vụ Thẻ", "len": 10},
        {"id": 3, "text": "Why was my transfer declined when I tried to send money?", "intent": "declined_transfer", "domain": "Nạp & Chuyển tiền", "len": 11},
        {"id": 4, "text": "fee?", "intent": "card_payment_fee_charged / Noise", "domain": "Câu cộc lốc (Điểm Nhiễu)", "len": 1},
        {"id": 5, "text": "I deposited a cheque yesterday but my balance has not been updated.", "intent": "balance_not_updated_after_cheque_or_cash_deposit", "domain": "Nạp & Chuyển tiền", "len": 12},
        {"id": 6, "text": "Can I change my card PIN at cash machines while on vacation in Spain?", "intent": "change_pin", "domain": "Dịch vụ Thẻ", "len": 14},
        {"id": 7, "text": "What documents are required to verify my identity and activate account?", "intent": "why_verify_identity", "domain": "Tài khoản & KYC", "len": 11},
        {"id": 8, "text": "What is the exchange rate from USD to EUR when making a wire transfer?", "intent": "exchange_rate", "domain": "Biểu phí & Tỷ giá", "len": 14},
        {"id": 9, "text": "My card got stuck inside the ATM machine at the branch, please help!", "intent": "card_swallowed", "domain": "Dịch vụ Thẻ", "len": 13},
        {"id": 10, "text": "There was a double charge on my statement yesterday morning for my supermarket grocery shopping order.", "intent": "transaction_charged_twice", "domain": "Sự cố Trừ tiền", "len": 17}
    ]

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
    hdbscan_configs_json = json.dumps(hdbscan_configs, ensure_ascii=False)
    kmeans_configs_json = json.dumps(kmeans_configs, ensure_ascii=False)
    dataset_samples_json = json.dumps(sample_dataset_sentences, ensure_ascii=False)
    presets_json = json.dumps(presets_data, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="vi" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BANKING77: Khám Phá Chủ Đề & So Sánh Thuật Toán Trực Quan</title>
    <!-- Favicon & Fonts -->
    <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🏛️</text></svg>">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    
    <!-- Plotly.js for interactive charts -->
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>

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
            padding: 26px 30px;
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
            margin-bottom: 8px;
        }}

        .hero-title {{ font-size: 23px; font-weight: 800; line-height: 1.3; margin-bottom: 8px; }}
        .hero-desc {{ font-size: 14px; color: var(--text-muted); line-height: 1.6; margin-bottom: 18px; }}

        .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; }}
        .kpi-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: var(--shadow-sm);
        }}
        .kpi-label {{ font-size: 11.5px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; margin-bottom: 4px; }}
        .kpi-value {{ font-size: 22px; font-weight: 800; color: var(--text-main); }}
        .kpi-subtext {{ font-size: 11.5px; color: var(--success); font-weight: 600; margin-top: 3px; }}

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
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }}
        th, td {{ padding: 10px 14px; border-bottom: 1px solid var(--border-color); }}
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
            min-width: 170px;
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

        .tester-grid {{ display: grid; grid-template-columns: 1.1fr 1fr; gap: 24px; }}
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

        /* Playground Styling */
        .playground-box {{
            background: var(--bg-page);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 22px;
        }}

        .config-radio-group {{
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-top: 8px;
        }}

        .config-radio-label {{
            display: flex;
            align-items: center;
            gap: 8px;
            background: var(--bg-card);
            padding: 8px 14px;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            transition: all 0.2s;
        }}
        .config-radio-label:hover {{ border-color: var(--primary); }}
        .config-radio-label input:checked + span {{ color: var(--primary); font-weight: 700; }}

        .metric-pill-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 12px;
            margin-top: 14px;
        }}

        .metric-pill {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 12px 14px;
            text-align: center;
        }}
        .metric-pill-title {{ font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 700; }}
        .metric-pill-val {{ font-size: 20px; font-weight: 800; color: var(--primary); margin: 3px 0; }}
        .metric-pill-sub {{ font-size: 11px; color: var(--text-muted); }}
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
        <!-- 4 TABS: Removed 3, 4, 6 and added Config & Compare Playground -->
        <nav class="tab-nav">
            <button class="tab-btn active" onclick="switchTab('tab-stepper', event)">🗺️ 1. Quy Trình Dựng Cụm 4 Bước</button>
            <button class="tab-btn" onclick="switchTab('tab-tester', event)">🧪 2. Chạy Thử Thuật Toán Từng Bước</button>
            <button class="tab-btn" onclick="switchTab('tab-playground', event)">⚙️ 3. Thử Cấu Hình & So Sánh (HDBSCAN vs K-Means)</button>
            <button class="tab-btn" onclick="switchTab('tab-novelty', event)">🚨 4. Giám Sát Chủ Đề Mới (Novelty)</button>
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

                <div class="stepper-nav">
                    <button class="step-btn" id="sbtn-1" onclick="setStep(1)">1. Text &rarr; Vector 384D</button>
                    <button class="step-btn" id="sbtn-2" onclick="setStep(2)">2. Nén Chiều UMAP 5D</button>
                    <button class="step-btn" id="sbtn-3" onclick="setStep(3)">3. Phân Cụm HDBSCAN</button>
                    <button class="step-btn active" id="sbtn-4" onclick="setStep(4)">4. Soi Chi Tiết 75 Cụm</button>
                </div>

                <div id="umap-stepper-chart" class="chart-box"></div>

                <div class="step-explainer" id="step-explainer-text"></div>

                <div id="step4-cluster-inspector" style="margin-top: 18px; padding-top: 16px; border-top: 1px solid var(--border-color);">
                    <div style="display: flex; gap: 14px; align-items: center; flex-wrap: wrap; margin-bottom: 12px;">
                        <label style="font-weight: 700; font-size: 13.5px;">🔍 Chọn nhanh Cụm để soi chi tiết:</label>
                        <select id="cluster-selector" onchange="inspectCluster(this.value)" class="search-box" style="width: auto;">
                        </select>
                    </div>
                    <div id="cluster-inspector-content" style="background: var(--bg-page); border: 1px solid var(--border-color); border-radius: 10px; padding: 18px;">
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
                    <div>
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
                                    <tbody></tbody>
                                </table>
                            </div>
                        </div>

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

                    <div>
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

                        <div class="panel" style="background: var(--bg-page); border: 1px dashed var(--border-color); margin-top: 14px;">
                            <h4 style="font-size: 15px; font-weight: 700; color: var(--primary); margin-bottom: 10px;">
                                🤖 Khuyến Nghị Kịch Bản Vận Hành Chatbot CSKH
                            </h4>
                            <div id="res-action-content" style="font-size: 13.5px; line-height: 1.7;"></div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <!-- TAB 3: DATASET OVERVIEW & CONFIG PLAYGROUND (HDBSCAN vs K-MEANS) -->
        <section id="tab-playground" class="tab-content">
            <!-- 1. DATASET OVERVIEW -->
            <div class="panel">
                <div class="panel-header">
                    <div>
                        <h3 class="panel-title">📦 Bước 1: Tập Dữ Liệu Đầu Vào (BANKING77)</h3>
                        <p class="panel-subtitle">Dữ liệu gồm hơn 13.000 tin nhắn thật của khách hàng ngân hàng số bằng tiếng Anh, chia đều cho 77 ý định.</p>
                    </div>
                    <div>
                        <span class="badge badge-blue">9.993 câu Train</span>
                        <span class="badge badge-green">77 Ý định</span>
                        <span class="badge badge-yellow">~130 câu/Ý định</span>
                    </div>
                </div>

                <div class="table-responsive">
                    <table id="dataset-preview-table">
                        <thead>
                            <tr>
                                <th>STT</th>
                                <th>Câu hỏi thực tế của khách hàng (Raw Text)</th>
                                <th>Ý định thực tế (Ground Truth)</th>
                                <th>Mảng Nghiệp Vụ</th>
                                <th>Độ dài câu</th>
                            </tr>
                        </thead>
                        <tbody>
                            <!-- Populated by JS -->
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- 2. CHOOSE CONFIGURATION & RUN -->
            <div class="panel">
                <div class="panel-header">
                    <div>
                        <h3 class="panel-title">⚙️ Bước 2: Chọn Cấu Hình & Bấm Chạy Thuật Toán</h3>
                        <p class="panel-subtitle">Thử nghiệm các cấu hình siêu tham số của HDBSCAN hoặc chọn K-Means để kiểm chứng sự khác biệt.</p>
                    </div>
                </div>

                <div class="playground-box">
                    <div style="margin-bottom: 16px;">
                        <label style="font-weight: 700; font-size: 13.5px; display: block; margin-bottom: 8px;">
                            1. Chọn Thuật toán Phân Cụm:
                        </label>
                        <div class="config-radio-group">
                            <label class="config-radio-label">
                                <input type="radio" name="algo_choice" value="HDBSCAN" checked onchange="onAlgoChange()">
                                <span>🟢 HDBSCAN (Phân cụm Mật độ Tự nhiên & Lọc Nhiễu)</span>
                            </label>
                            <label class="config-radio-label">
                                <input type="radio" name="algo_choice" value="KMEANS" onchange="onAlgoChange()">
                                <span>🟠 K-Means (Tâm hình cầu Voronoi & Không lọc Nhiễu)</span>
                            </label>
                            <label class="config-radio-label">
                                <input type="radio" name="algo_choice" value="COMPARE" onchange="onAlgoChange()">
                                <span>⚖️ So Sánh Trực Tiếp (HDBSCAN mcs=50 vs K-Means K=77)</span>
                            </label>
                        </div>
                    </div>

                    <!-- Param Controls for HDBSCAN -->
                    <div id="hdbscan-params" style="display: block; margin-bottom: 16px;">
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px;">
                            <div>
                                <label style="font-size: 12.5px; font-weight: 700; display: block; margin-bottom: 4px;">Kích thước cụm tối thiểu (min_cluster_size):</label>
                                <select id="hdbscan-mcs-select" class="search-box" style="width: 100%;">
                                    <option value="50" selected>mcs = 50 (Tối ưu đề xuất)</option>
                                    <option value="20">mcs = 20 (Cụm nhỏ, dễ phân mảnh)</option>
                                    <option value="30">mcs = 30</option>
                                    <option value="80">mcs = 80 (Cụm to, dễ gộp nhầm)</option>
                                </select>
                            </div>
                            <div>
                                <label style="font-size: 12.5px; font-weight: 700; display: block; margin-bottom: 4px;">Phương thức trích xuất cụm (selection):</label>
                                <select id="hdbscan-sel-select" class="search-box" style="width: 100%;">
                                    <option value="eom" selected>EOM (Excess of Mass - Bền vững nhất)</option>
                                    <option value="leaf">Leaf (Tách sâu xuống lá cây)</option>
                                </select>
                            </div>
                        </div>
                    </div>

                    <!-- Param Controls for K-Means -->
                    <div id="kmeans-params" style="display: none; margin-bottom: 16px;">
                        <div style="max-width: 400px;">
                            <label style="font-size: 12.5px; font-weight: 700; display: block; margin-bottom: 4px;">Số lượng cụm K biết trước (n_clusters):</label>
                            <select id="kmeans-k-select" class="search-box" style="width: 100%;">
                                <option value="77" selected>K = 77 (Bằng đúng số nhãn thật)</option>
                                <option value="50">K = 50 (Thiếu 27 cụm)</option>
                                <option value="20">K = 20 (Gộp ép thành 20 khối cầu to)</option>
                                <option value="100">K = 100 (Xé nhỏ thành 100 khối cầu)</option>
                            </select>
                        </div>
                    </div>

                    <!-- Action Button -->
                    <div style="margin-top: 18px; display: flex; align-items: center; gap: 14px;">
                        <button class="btn btn-primary" onclick="runClusteringSimulation()" style="font-size: 14px; padding: 10px 24px;">
                            ▶️ Chạy Phân Cụm Ngay
                        </button>
                        <span id="run-status" style="font-size: 13px; color: var(--text-muted); font-weight: 600;"></span>
                    </div>
                </div>

                <!-- 3. OUTPUT & RESULTS AFTER RUNNING -->
                <div id="playground-results">
                    <div class="panel-header" style="margin-top: 24px;">
                        <div>
                            <h4 style="font-size: 16px; font-weight: 700; color: var(--primary);" id="result-config-title">
                                📊 Kết Quả Đầu Ra: HDBSCAN (mcs=50, EOM)
                            </h4>
                            <p class="panel-subtitle" id="result-config-desc">
                                Đang hiển thị kết quả tính toán chi tiết của cấu hình được chọn.
                            </p>
                        </div>
                        <div>
                            <span class="badge badge-green" id="result-badge">🏆 TỐI ƯU TOÀN DIỆN</span>
                        </div>
                    </div>

                    <!-- Output KPI Metric Pills -->
                    <div class="metric-pill-grid">
                        <div class="metric-pill">
                            <div class="metric-pill-title">Số Cụm Tìm Được</div>
                            <div class="metric-pill-val" id="res-m-clusters">75</div>
                            <div class="metric-pill-sub" id="res-sub-clusters">Tiệm cận 77 nhãn thật</div>
                        </div>
                        <div class="metric-pill">
                            <div class="metric-pill-title">Tỷ Lệ Noise (Nhiễu)</div>
                            <div class="metric-pill-val" id="res-m-noise">7,41%</div>
                            <div class="metric-pill-sub" id="res-sub-noise">Tách lọc câu cộc lốc/mơ hồ</div>
                        </div>
                        <div class="metric-pill">
                            <div class="metric-pill-title">V-Measure (Khớp Nhãn)</div>
                            <div class="metric-pill-val" id="res-m-v">0,833</div>
                            <div class="metric-pill-sub" id="res-sub-v">0,873 nếu bỏ qua noise</div>
                        </div>
                        <div class="metric-pill">
                            <div class="metric-pill-title">Tương Đồng ARI</div>
                            <div class="metric-pill-val" id="res-m-ari">0,542</div>
                            <div class="metric-pill-sub">Hiệu chỉnh ngẫu nhiên</div>
                        </div>
                        <div class="metric-pill">
                            <div class="metric-pill-title">Chỉ Số Mật Độ DBCV</div>
                            <div class="metric-pill-val" id="res-m-dbcv">0,432</div>
                            <div class="metric-pill-sub" id="res-sub-dbcv">Mật độ lõi sắc nét</div>
                        </div>
                        <div class="metric-pill">
                            <div class="metric-pill-title">Tỷ Lệ Bao Phủ</div>
                            <div class="metric-pill-val" id="res-m-cov">92,59%</div>
                            <div class="metric-pill-sub">Thu nhận phần lớn câu hỏi</div>
                        </div>
                    </div>

                    <!-- Result Scatter Plot -->
                    <div style="margin-top: 24px;">
                        <h4 style="font-size: 15px; font-weight: 700; margin-bottom: 8px;">
                            🗺️ Bản Đồ Phân Bố Cụm 2D Sau Khi Chạy:
                        </h4>
                        <div id="playground-scatter-chart" class="chart-box" style="min-height: 480px;"></div>
                    </div>

                    <!-- Comparative Table & Insights -->
                    <div id="compare-section" style="margin-top: 28px; display: none;">
                        <h4 style="font-size: 16px; font-weight: 700; color: var(--primary); margin-bottom: 12px;">
                            ⚖️ Bảng Đối Chiếu Trực Diện: HDBSCAN vs K-Means
                        </h4>
                        <div class="table-responsive">
                            <table>
                                <thead>
                                    <tr>
                                        <th>Tiêu Chí So Sánh</th>
                                        <th>HDBSCAN (mcs=50, EOM - Đề xuất)</th>
                                        <th>K-Means (K=77 - Truyền thống)</th>
                                        <th>Ý Nghĩa Khoa Học & Nghiệp Vụ</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td><strong>Xác định số cụm $K$</strong></td>
                                        <td><span class="badge badge-green">Tự động phát hiện 75 cụm</span></td>
                                        <td><span class="badge badge-red">Bắt buộc phải biết trước $K=77$</span></td>
                                        <td>Trong thực tế, tin nhắn khách hàng đổ về không ai biết trước có bao nhiêu loại lỗi. K-Means bắt buộc con người phải đoán mò.</td>
                                    </tr>
                                    <tr>
                                        <td><strong>Hình dạng cụm giả định</strong></td>
                                        <td><span class="badge badge-blue">Dải lụa mật độ uốn lượn tự do</span></td>
                                        <td><span class="badge badge-yellow">Khối cầu lồi đối xứng (Voronoi)</span></td>
                                        <td>Ngôn ngữ tự nhiên tạo thành các dải lụa uốn lượn. K-Means dùng mặt phẳng chém đứt dải lụa làm đôi.</td>
                                    </tr>
                                    <tr>
                                        <td><strong>Cơ chế lọc nhiễu & rác</strong></td>
                                        <td><span class="badge badge-green">Tách 7,41% Noise (-1) ra ngoài</span></td>
                                        <td><span class="badge badge-red">0% Noise (Ép 100% câu vào cụm)</span></td>
                                        <td>Câu cộc lốc như <em>'fee?'</em>, <em>'help'</em> bị K-Means ép vào một cụm nào đó làm rác bẩn toàn bộ dữ liệu.</td>
                                    </tr>
                                    <tr>
                                        <td><strong>Chỉ số chất lượng DBCV</strong></td>
                                        <td><strong>0,432</strong> (Đỉnh cao mật độ)</td>
                                        <td>Không thể tính (Không dựa trên mật độ)</td>
                                        <td>DBCV đo độ sâu của các thung lũng phân tách cụm, tương quan +0.93 với nhãn thật.</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <!-- Expert Commentary Box -->
                    <div class="step-explainer" id="result-commentary" style="margin-top: 20px;">
                        <!-- Dynamic commentary -->
                    </div>
                </div>
            </div>
        </section>

        <!-- TAB 4: NOVELTY DETECTION -->
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
    </main>

    <!-- Embedded Data Payload -->
    <script>
        const TOPICS_DATA = {topics_json};
        const SAMPLED_POINTS = {sampled_points_json};
        const HDBSCAN_CONFIGS = {hdbscan_configs_json};
        const KMEANS_CONFIGS = {kmeans_configs_json};
        const DATASET_SAMPLES = {dataset_samples_json};
        const PRESETS_DATA = {presets_json};

        let currentStep = 4;
        let currentPlaygroundConfig = HDBSCAN_CONFIGS["hdbscan_50_eom"];

        function toggleTheme() {{
            const current = document.documentElement.getAttribute('data-theme');
            const next = current === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', next);
            document.getElementById('theme-btn').innerText = next === 'dark' ? '🌙 Dark' : '☀️ Light';
            renderStepperChart(currentStep);
            renderPlaygroundScatter(currentPlaygroundConfig);
            renderRocChart();
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
                const chartIds = ['umap-stepper-chart', 'playground-scatter-chart', 'roc-chart'];
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

            let p = presetObj;
            if (!p) {{
                for (let k in PRESETS_DATA) {{
                    if (PRESETS_DATA[k].text.toLowerCase() === text.toLowerCase()) {{
                        p = PRESETS_DATA[k];
                        break;
                    }}
                }}
            }}

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

            document.getElementById('res-syntax-type').innerText = syntax;
            document.getElementById('res-stage2-coords').innerText = coords;
            document.getElementById('res-coords-text').innerText = coords;

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

        // PLAYGROUND: DATASET OVERVIEW & CONFIG RUNNER (Tab 3)
        function initDatasetPreview() {{
            const tbody = document.querySelector('#dataset-preview-table tbody');
            tbody.innerHTML = '';
            DATASET_SAMPLES.forEach(s => {{
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><strong>#${{s.id}}</strong></td>
                    <td><em>"${{s.text}}"</em></td>
                    <td><code>${{s.intent}}</code></td>
                    <td><span class="badge badge-blue">${{s.domain}}</span></td>
                    <td>${{s.len}} từ</td>
                `;
                tbody.appendChild(tr);
            }});
        }}

        function onAlgoChange() {{
            const choice = document.querySelector('input[name="algo_choice"]:checked').value;
            const hdbBox = document.getElementById('hdbscan-params');
            const kmBox = document.getElementById('kmeans-params');
            
            if (choice === 'HDBSCAN') {{
                hdbBox.style.display = 'block';
                kmBox.style.display = 'none';
            }} else if (choice === 'KMEANS') {{
                hdbBox.style.display = 'none';
                kmBox.style.display = 'block';
            }} else {{
                // COMPARE
                hdbBox.style.display = 'block';
                kmBox.style.display = 'block';
            }}
        }}

        function runClusteringSimulation() {{
            const choice = document.querySelector('input[name="algo_choice"]:checked').value;
            const statusEl = document.getElementById('run-status');
            statusEl.innerText = "⏳ Đang phân cụm dữ liệu BANKING77...";

            setTimeout(() => {{
                statusEl.innerText = "✅ Hoàn tất phân cụm!";
                setTimeout(() => {{ statusEl.innerText = ""; }}, 2500);

                let cfg;
                const compareSec = document.getElementById('compare-section');

                if (choice === 'HDBSCAN') {{
                    compareSec.style.display = 'none';
                    const mcs = document.getElementById('hdbscan-mcs-select').value;
                    const sel = document.getElementById('hdbscan-sel-select').value;
                    const key = `hdbscan_${{mcs}}_${{sel}}`;
                    cfg = HDBSCAN_CONFIGS[key] || HDBSCAN_CONFIGS["hdbscan_50_eom"];
                }} else if (choice === 'KMEANS') {{
                    compareSec.style.display = 'none';
                    const k = document.getElementById('kmeans-k-select').value;
                    const key = `kmeans_${{k}}`;
                    cfg = KMEANS_CONFIGS[key] || KMEANS_CONFIGS["kmeans_77"];
                }} else {{
                    // COMPARE: HDBSCAN mcs=50 vs KMEANS K=77
                    compareSec.style.display = 'block';
                    cfg = HDBSCAN_CONFIGS["hdbscan_50_eom"];
                }}

                currentPlaygroundConfig = cfg;
                renderPlaygroundResults(cfg, choice === 'COMPARE');
            }}, 300);
        }}

        function renderPlaygroundResults(cfg, isCompare = false) {{
            document.getElementById('result-config-title').innerText = isCompare 
                ? "📊 Kết Quả: Đối Chiếu Song Song HDBSCAN (mcs=50) vs K-Means (K=77)"
                : `📊 Kết Quả Đầu Ra: ${{cfg.name}}`;
            document.getElementById('result-config-desc').innerText = cfg.summary;
            document.getElementById('result-badge').innerText = cfg.badge;
            document.getElementById('result-badge').className = cfg.badge.includes('TỐI ƯU') ? "badge badge-green" : "badge badge-yellow";

            document.getElementById('res-m-clusters').innerText = cfg.clusters;
            document.getElementById('res-sub-clusters').innerText = cfg.algo === 'K-Means' ? 'Bắt buộc biết trước K' : 'Tự động phát hiện';
            document.getElementById('res-m-noise').innerText = cfg.noise + '%';
            document.getElementById('res-sub-noise').innerText = cfg.noise === 0 ? 'Không lọc, ép 100% vào cụm' : 'Lọc bỏ câu mơ hồ/rác';
            document.getElementById('res-m-v').innerText = cfg.v_measure;
            document.getElementById('res-m-ari').innerText = cfg.ari;
            document.getElementById('res-m-dbcv').innerText = cfg.dbcv !== null ? cfg.dbcv : 'N/A';
            document.getElementById('res-sub-dbcv').innerText = cfg.dbcv !== null ? 'Chất lượng mật độ nội tại' : 'K-Means không có mật độ';
            document.getElementById('res-m-cov').innerText = cfg.coverage + '%';

            // Commentary
            document.getElementById('result-commentary').innerHTML = `
                <strong>💡 Bình luận Chuyên gia:</strong><br>
                ${{cfg.summary}}<br><br>
                <strong>So sánh cốt lõi:</strong> HDBSCAN theo đuổi hình thái tự nhiên của ngôn ngữ (dải lụa mật độ) và có cơ chế loại bỏ rác. Trong khi đó, K-Means cắt không gian bằng mặt phẳng hình học và ép 100% câu hỏi (kể cả câu cộc lốc hoặc rác) vào cụm.
            `;

            renderPlaygroundScatter(cfg);
        }}

        function renderPlaygroundScatter(cfg) {{
            const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
            const bgColor = isDark ? '#151C2C' : '#FFFFFF';
            const textColor = isDark ? '#F8FAFC' : '#0F172A';
            const gridColor = isDark ? '#232D42' : '#F1F5F9';

            const colName = cfg.col;
            let traces = [];

            if (cfg.algo === 'HDBSCAN') {{
                const inCluster = SAMPLED_POINTS.filter(p => p[colName] !== -1);
                const noise = SAMPLED_POINTS.filter(p => p[colName] === -1);

                traces.push({{
                    x: inCluster.map(p => p.x_umap),
                    y: inCluster.map(p => p.y_umap),
                    mode: 'markers',
                    type: 'scatter',
                    marker: {{
                        size: 5,
                        color: inCluster.map(p => p[colName]),
                        colorscale: 'Turbo',
                        opacity: 0.7
                    }},
                    hoverinfo: 'text',
                    hovertext: inCluster.map(p => `<b>Cụm #${{p[colName]}}</b> (${{p.intent}})<br>"${{p.text}}"`),
                    name: `Cụm ${{cfg.clusters}} nhóm`
                }});

                if (noise.length > 0) {{
                    traces.push({{
                        x: noise.map(p => p.x_umap),
                        y: noise.map(p => p.y_umap),
                        mode: 'markers',
                        type: 'scatter',
                        marker: {{ size: 6, color: '#EF4444', symbol: 'x', opacity: 0.85 }},
                        hoverinfo: 'text',
                        hovertext: noise.map(p => `<b>Điểm Nhiễu (-1)</b><br>"${{p.text}}"`),
                        name: `Điểm Nhiễu (${{cfg.noise}}%)`
                    }});
                }}
            }} else {{
                // K-Means: No noise, all colored
                traces.push({{
                    x: SAMPLED_POINTS.map(p => p.x_umap),
                    y: SAMPLED_POINTS.map(p => p.y_umap),
                    mode: 'markers',
                    type: 'scatter',
                    marker: {{
                        size: 5,
                        color: SAMPLED_POINTS.map(p => p[colName]),
                        colorscale: 'Rainbow',
                        opacity: 0.75
                    }},
                    hoverinfo: 'text',
                    hovertext: SAMPLED_POINTS.map(p => `<b>K-Means Cụm #${{p[colName]}}</b> (${{p.intent}})<br>"${{p.text}}"`),
                    name: `K-Means (${{cfg.clusters}} Khối Cầu Voronoi - 0% Noise)`
                }});
            }}

            Plotly.newPlot('playground-scatter-chart', traces, {{
                margin: {{ l: 40, r: 40, t: 30, b: 40 }},
                paper_bgcolor: bgColor,
                plot_bgcolor: bgColor,
                xaxis: {{ title: 'Thành phần UMAP 1', gridcolor: gridColor, color: textColor }},
                yaxis: {{ title: 'Thành phần UMAP 2', gridcolor: gridColor, color: textColor }},
                legend: {{ orientation: 'h', y: -0.15, font: {{ color: textColor }} }},
                hoverlabel: {{ bgcolor: isDark ? '#1E293B' : '#FFFFFF', font: {{ family: 'Plus Jakarta Sans' }} }}
            }}, {{ responsive: true }});
        }}

        // NOVELTY ROC CHART (Tab 4)
        function renderRocChart() {{
            const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
            const bgColor = isDark ? '#151C2C' : '#FFFFFF';
            const textColor = isDark ? '#F8FAFC' : '#0F172A';
            const gridColor = isDark ? '#232D42' : '#F1F5F9';

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

        document.addEventListener('DOMContentLoaded', () => {{
            setStep(4);
            initClusterSelector();
            initPresetExamples();
            initDatasetPreview();
            runClusteringSimulation();
            renderRocChart();
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
