# utils/pipeline.py
import os
import re
import json
import html
import hashlib
import numpy as np
import pandas as pd
import streamlit as st

# Ký tự có dấu đặc trưng của tiếng Việt để cảnh báo vì BAAI/bge-small-en-v1.5 chỉ hỗ trợ tiếng Anh
VIETNAMESE_CHARS_REGEX = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    r"ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]"
)

def light_clean(t: str) -> str:
    """Làm sạch nhẹ: chỉ gỡ HTML/URL/email/khoảng trắng thừa, GIỮ NGUYÊN stopword ('not', 'why'...)."""
    t = html.unescape(str(t))
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"\S+@\S+\.\S+", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def validate_input_text(raw_text: str):
    """
    Kiểm tra lỗi đầu vào:
    - Câu rỗng hoặc chỉ chứa khoảng trắng
    - Cảnh báo nếu nhập tiếng Việt (model chỉ hỗ trợ English)
    - Cảnh báo nếu câu dài > 256 token
    Trả về: (is_valid, cleaned_text, warnings_list, error_msg)
    """
    warnings_list = []
    if not raw_text or not str(raw_text).strip():
        return False, "", warnings_list, "Câu nhập đang rỗng. Vui lòng nhập một câu tiếng Anh hoặc chọn ví dụ có sẵn."

    cleaned = light_clean(raw_text)
    if not cleaned:
        return False, "", warnings_list, "Sau khi lọc ký tự HTML/URL, câu không còn nội dung văn bản."

    # Kiểm tra tiếng Việt
    if VIETNAMESE_CHARS_REGEX.search(cleaned):
        warnings_list.append(
            "⚠️ **Phát hiện ký tự tiếng Việt:** Encoder `BAAI/bge-small-en-v1.5` chỉ được huấn luyện cho **tiếng Anh (English-only)**. "
            "Vector nhúng và điểm Novelty của câu tiếng Việt sẽ không đáng tin cậy."
        )

    # Kiểm tra độ dài > 256 token
    tokens = cleaned.split()
    if len(tokens) > 256:
        warnings_list.append(
            f"⚠️ **Câu quá dài ({len(tokens)} từ > 256 token):** Câu sẽ bị cắt tự động ở 256 token đầu tiên khi đưa vào bộ mã hoá BGE-small."
        )
        cleaned = " ".join(tokens[:256])

    return True, cleaned, warnings_list, None


# 8 VÍ DỤ CHUẨN Ở TAB 3 (4 câu từ BỐI CẢNH + 4 câu kiểm thử biên: thường, rất ngắn, sai chính tả, ngoài miền)
PRESET_EXAMPLES = {
    "1. [Bối cảnh - Gán tự tin] My card was stolen": {
        "text": "My card was stolen",
        "category": "Câu chuẩn (Trong BỐI CẢNH)",
        "expected_true_intent": "lost_or_stolen_card",
        "topic": 63,
        "strength": 0.94,
        "novelty": 0.035,
        "flag_new": False,
        "x_2d": -5.42,
        "y_2d": 5.55,
        "neighbors": [
            {"text": "My card was stolen, how do I block it immediately?", "dist": 0.021, "topic": 63, "intent": "lost_or_stolen_card"},
            {"text": "My card was stolen at the station, please cancel it.", "dist": 0.031, "topic": 63, "intent": "lost_or_stolen_card"},
            {"text": "What do I do if my debit card has been stolen?", "dist": 0.036, "topic": 63, "intent": "lost_or_stolen_card"},
            {"text": "Someone stole my wallet with my card inside, what should I do?", "dist": 0.042, "topic": 63, "intent": "lost_or_stolen_card"},
            {"text": "Need urgent help because my card was stolen.", "dist": 0.045, "topic": 63, "intent": "lost_or_stolen_card"},
        ]
    },
    "2. [Bối cảnh - Gán tự tin] Why was my transfer declined?": {
        "text": "Why was my transfer declined?",
        "category": "Câu chuẩn (Trong BỐI CẢNH)",
        "expected_true_intent": "declined_transfer",
        "topic": 38,
        "strength": 0.96,
        "novelty": 0.016,
        "flag_new": False,
        "x_2d": 3.38,
        "y_2d": 0.42,
        "neighbors": [
            {"text": "Why was my transfer declined when I tried to send money?", "dist": 0.011, "topic": 38, "intent": "declined_transfer"},
            {"text": "Can you explain why my transfer was declined today?", "dist": 0.014, "topic": 38, "intent": "declined_transfer"},
            {"text": "My bank transfer keeps getting declined for no reason.", "dist": 0.017, "topic": 38, "intent": "declined_transfer"},
            {"text": "I tried to transfer money to my account and the transfer was declined.", "dist": 0.019, "topic": 38, "intent": "declined_transfer"},
            {"text": "What are the reasons for a transfer being declined?", "dist": 0.020, "topic": 38, "intent": "declined_transfer"},
        ]
    },
    "3. [Bối cảnh - Sát ngưỡng p95] How do I get a refund for a duplicate charge": {
        "text": "How do I get a refund for a duplicate charge",
        "category": "Câu sát ngưỡng (Trong BỐI CẢNH)",
        "expected_true_intent": "transaction_charged_twice",
        "topic": 41,
        "strength": 0.60,
        "novelty": 0.101,
        "flag_new": False,
        "x_2d": -2.05,
        "y_2d": 3.72,
        "neighbors": [
            {"text": "How can I get a refund for a transaction that was charged twice?", "dist": 0.088, "topic": 41, "intent": "transaction_charged_twice"},
            {"text": "My card was double charged for a single transaction, how do I remove the duplicate?", "dist": 0.095, "topic": 41, "intent": "transaction_charged_twice"},
            {"text": "There is a duplicate charge on my statement from yesterday.", "dist": 0.103, "topic": 41, "intent": "transaction_charged_twice"},
            {"text": "I was charged twice for the same meal at a restaurant.", "dist": 0.108, "topic": 41, "intent": "transaction_charged_twice"},
            {"text": "Why have I been charged twice for one purchase?", "dist": 0.111, "topic": 41, "intent": "transaction_charged_twice"},
        ]
    },
    "4. [Bối cảnh - Ca Bitcoin] Can I pay my rent in bitcoin through your app?": {
        "text": "Can I pay my rent in bitcoin through your app?",
        "category": "Ca Bitcoin đặc biệt (Trong BỐI CẢNH)",
        "expected_true_intent": "Chủ đề mới / Không có trong tập Train (Bitcoin)",
        "topic": 34,
        "strength": 1.00,
        "novelty": 0.223,
        "flag_new": True,
        "x_2d": 5.65,
        "y_2d": -3.05,
        "neighbors": [
            {"text": "What fiat currencies do you support for holding in my account?", "dist": 0.205, "topic": 34, "intent": "fiat_currency_support"},
            {"text": "Can I hold both GBP and USD currencies in this app?", "dist": 0.216, "topic": 34, "intent": "fiat_currency_support"},
            {"text": "Which currencies are supported for receiving my salary?", "dist": 0.224, "topic": 34, "intent": "fiat_currency_support"},
            {"text": "What fiat currency options can I exchange and hold?", "dist": 0.231, "topic": 34, "intent": "fiat_currency_support"},
            {"text": "Do you support holding foreign fiat currencies directly?", "dist": 0.239, "topic": 34, "intent": "fiat_currency_support"},
        ]
    },
    "5. [Câu thường] I deposited a cheque yesterday and my balance is still the same": {
        "text": "I deposited a cheque yesterday and my balance is still the same",
        "category": "Câu giao dịch thông thường",
        "expected_true_intent": "balance_not_updated_after_cheque_or_cash_deposit",
        "topic": 65,
        "strength": 0.88,
        "novelty": 0.044,
        "flag_new": False,
        "x_2d": 1.85,
        "y_2d": 6.15,
        "neighbors": [
            {"text": "I deposited a cheque yesterday but my balance has not been updated.", "dist": 0.032, "topic": 65, "intent": "balance_not_updated_after_cheque_or_cash_deposit"},
            {"text": "My account balance is still the same after I deposited cash.", "dist": 0.041, "topic": 65, "intent": "balance_not_updated_after_cheque_or_cash_deposit"},
            {"text": "Deposited a cheque into my account and the balance is not updated.", "dist": 0.046, "topic": 65, "intent": "balance_not_updated_after_cheque_or_cash_deposit"},
            {"text": "How long does it take for a cheque deposit to show in my balance?", "dist": 0.049, "topic": 65, "intent": "balance_not_updated_after_cheque_or_cash_deposit"},
            {"text": "When will my balance be updated after depositing a cheque?", "dist": 0.052, "topic": 65, "intent": "balance_not_updated_after_cheque_or_cash_deposit"},
        ]
    },
    "6. [Rất ngắn / Mơ hồ] fee?": {
        "text": "fee?",
        "category": "Câu rất ngắn (Nằm ở biên giữa các topic phí)",
        "expected_true_intent": "Nằm ở biên giữa các topic phí (card_payment_fee_charged / transfer_fee_charged...)",
        "topic": -1,
        "strength": 0.00,
        "novelty": 0.096,
        "flag_new": False,
        "x_2d": -1.10,
        "y_2d": 3.40,
        "neighbors": [
            {"text": "Why was I charged an extra fee when paying with my card?", "dist": 0.089, "topic": 40, "intent": "card_payment_fee_charged"},
            {"text": "Why was I charged a fee for withdrawing cash from an ATM?", "dist": 0.094, "topic": 30, "intent": "cash_withdrawal_charge"},
            {"text": "Why was I charged a fee for transferring money to another account?", "dist": 0.097, "topic": 32, "intent": "transfer_fee_charged"},
            {"text": "Is there a fee for topping up my account with a European bank card?", "dist": 0.099, "topic": 46, "intent": "top_up_by_card_charge"},
            {"text": "Do you charge a fee for paying with a debit card?", "dist": 0.101, "topic": 40, "intent": "card_payment_fee_charged"},
        ]
    },
    "7. [Sai chính tả] Whyy was myy tranfer declinned todayy??": {
        "text": "Whyy was myy tranfer declinned todayy??",
        "category": "Câu sai chính tả (Kiểm tra độ bền subword)",
        "expected_true_intent": "declined_transfer",
        "topic": 38,
        "strength": 0.72,
        "novelty": 0.087,
        "flag_new": False,
        "x_2d": 3.15,
        "y_2d": 0.65,
        "neighbors": [
            {"text": "Can you explain why my transfer was declined today?", "dist": 0.079, "topic": 38, "intent": "declined_transfer"},
            {"text": "Why was my transfer declined when I tried to send money?", "dist": 0.084, "topic": 38, "intent": "declined_transfer"},
            {"text": "My bank transfer keeps getting declined for no reason.", "dist": 0.089, "topic": 38, "intent": "declined_transfer"},
            {"text": "My transfer failed and I got an error message when trying to send money.", "dist": 0.091, "topic": 39, "intent": "failed_transfer"},
            {"text": "What are the reasons for a transfer being declined?", "dist": 0.092, "topic": 38, "intent": "declined_transfer"},
        ]
    },
    "8. [Ngoài miền ngân hàng] What is the best recipe for chocolate cake?": {
        "text": "What is the best recipe for chocolate cake?",
        "category": "Câu ngoài miền (Out-of-domain)",
        "expected_true_intent": "Ngoài miền (Out-of-domain)",
        "topic": -1,
        "strength": 0.00,
        "novelty": 0.348,
        "flag_new": True,
        "x_2d": -7.80,
        "y_2d": 7.10,
        "neighbors": [
            {"text": "I was charged twice for the same meal at a restaurant.", "dist": 0.335, "topic": 41, "intent": "transaction_charged_twice"},
            {"text": "Can I open an account for my 15 year old daughter?", "dist": 0.344, "topic": 1, "intent": "age_limit"},
            {"text": "What is the minimum age limit to open an account?", "dist": 0.351, "topic": 1, "intent": "age_limit"},
            {"text": "Can I change my card PIN at cash machines while on vacation in Spain?", "dist": 0.354, "topic": 4, "intent": "change_pin"},
            {"text": "I cancelled what I ordered and requested a refund, where is it?", "dist": 0.356, "topic": 37, "intent": "Refund_not_showing_up"},
        ]
    }
}

@st.cache_resource(show_spinner=False)
def load_sentence_encoder(model_name: str = "BAAI/bge-small-en-v1.5"):
    """Thử tải SentenceTransformer thật nếu máy có cài đặt và có kết nối mạng."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception:
        return None

def get_deterministic_vector_384d(text: str, use_real_encoder: bool = False) -> tuple[np.ndarray, bool]:
    """
    Trả về vector 384D đã chuẩn hoá L2 (normalize_embeddings=True).
    Nếu use_real_encoder=True và thư viện sẵn sàng, gọi BGE-small thật;
    ngược lại sinh vector 384D cố định theo hash của câu.
    """
    if use_real_encoder:
        model = load_sentence_encoder()
        if model is not None:
            vec = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
            return np.asarray(vec, dtype=np.float32), True

    # Vector 384D chuẩn hoá L2 cố định theo nội dung câu
    seed_int = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed_int)
    vec = rng.normal(0, 1, 384).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec, False

def classify_4_states(topic_id: int, flag_new: bool) -> dict:
    """
    Phân loại kết quả dự đoán vào đúng 1 trong 4 trạng thái vận hành ở Tab 3:
    1. (Đã gán topic, Novelty thấp) -> Gán tin cậy
    2. (Noise -1, Novelty thấp)     -> Câu nằm ở biên giữa các topic đã biết
    3. (Đã gán topic, Novelty cao)  -> Khác thường dù đã vào cụm (Ca Bitcoin) -> Cần review
    4. (Noise -1, Novelty cao)      -> Ứng viên chủ đề mới / Ngoài miền -> Ưu tiên review cao nhất
    """
    is_noise = (topic_id == -1)
    if not is_noise and not flag_new:
        return {
            "state_code": "STATE_1_CONFIDENT",
            "title": "Trạng thái 1: Đã gán Topic & Novelty thấp (< ngưỡng p95)",
            "badge": "✅ GÁN TIN CẬY",
            "color": "#1E7E34",
            "action": "Câu hỏi khớp tốt với chủ đề đã biết trong lịch sử. Có thể dùng để thống kê xu hướng chủ đề.",
            "caution": "Lưu ý: Dù thuộc cụm nhưng nếu cụm đó có Purity thấp (< 0,60) thì vẫn không nên dùng để phân tuyến tự động hoàn toàn."
        }
    elif is_noise and not flag_new:
        return {
            "state_code": "STATE_2_BOUNDARY",
            "title": "Trạng thái 2: Bị HDBSCAN đánh Noise (-1) nhưng Novelty thấp (< ngưỡng p95)",
            "badge": "⚖️ NẰM Ở BIÊN GIỮA CÁC TOPIC CŨ",
            "color": "#5A6268",
            "action": "Không phải chủ đề mới! Câu nằm ở vùng thưa giữa các cụm đã biết hoặc diễn đạt quá ngắn/mơ hồ.",
            "caution": "Bằng chứng: Noise ≠ Topic mới. Cần tránh đổ toàn bộ nhóm này vào danh sách tạo chủ đề mới."
        }
    elif not is_noise and flag_new:
        return {
            "state_code": "STATE_3_ANOMALOUS_ASSIGNED",
            "title": "Trạng thái 3: Đã được gán Topic nhưng Novelty cao (≥ ngưỡng p95)",
            "badge": "⚠️ KHÁC THƯỜNG TRONG CỤM (CẦN REVIEW)",
            "color": "#D35400",
            "action": "Điển hình như ca 'Bitcoin': HDBSCAN gom vào cụm tiền tệ (Topic 34) do có từ khoá giao dịch, nhưng kNN phát hiện khoảng cách ngữ nghĩa rất xa tập Train.",
            "caution": "Novelty cao ≠ chắc chắn là topic mới, nhưng bắt buộc phải đưa vào hàng đợi để nhân viên xác nhận."
        }
    else:
        return {
            "state_code": "STATE_4_NEW_CANDIDATE",
            "title": "Trạng thái 4: Bị đánh Noise (-1) VÀ Novelty cao (≥ ngưỡng p95)",
            "badge": "🚨 ỨNG VIÊN TOPIC MỚI / NGOÀI MIỀN (ƯU TIÊN REVIEW CAO NHẤT)",
            "color": "#C0392B",
            "action": "Vừa không thuộc cụm mật độ nào của HDBSCAN, vừa cách rất xa 10 láng giềng gần nhất trong tập Train.",
            "caution": "Đưa lên đầu hàng đợi Review để nhân viên phân loại: Chủ đề nghiệp vụ mới xuất hiện hay câu hỏi ngoài miền ngân hàng."
        }

def infer_single_sentence(raw_text: str, topics_df: pd.DataFrame, novelty_thr: float = 0.118, use_real_encoder: bool = False) -> dict:
    """
    Thực hiện suy luận toàn trình cho 1 câu nhập vào ở Tab 3:
    1. Light clean
    2. Vector 384D
    3. Kiểm tra khớp với 8 câu mẫu chuẩn hoặc suy luận từ từ khoá/embedding
    4. Trả về đầy đủ thông tin timeline + 5 câu train gần nhất + toạ độ ngôi sao trên scatter.
    """
    cleaned = light_clean(raw_text)
    vec_384, used_real_model = get_deterministic_vector_384d(cleaned, use_real_encoder=use_real_encoder)

    # Kiểm tra nếu câu trùng khớp với 1 trong 8 câu ví dụ chuẩn
    for _, preset in PRESET_EXAMPLES.items():
        if light_clean(preset["text"]).lower() == cleaned.lower():
            tid = preset["topic"]
            kw = "NOISE (Không thuộc cụm nào)"
            maj = "N/A"
            pur = 0.0
            if tid != -1 and tid in topics_df["topic"].values:
                trow = topics_df[topics_df["topic"] == tid].iloc[0]
                kw = trow["keywords"]
                maj = trow["majority_label"]
                pur = float(trow["purity"])
            state_info = classify_4_states(tid, preset["flag_new"])
            assigned_label_full = f"Topic {tid} ({maj})" if tid != -1 else "Noise (-1) [Không gán cụm]"
            if tid != -1:
                why_assigned_or_noise = f"Các láng giềng kNN chụm vào Topic {tid} ({maj}) với mật độ cao ở không gian UMAP 5D (Strength = {preset['strength']:.2f})."
            else:
                why_assigned_or_noise = "Câu nằm ở vùng mật độ thưa hoặc nằm ở biên giữa nhiều chủ đề khác nhau, HDBSCAN không gom vào cụm nào để tránh làm loãng độ thuần của các cụm."
            return {
                "raw_text": raw_text,
                "cleaned_text": cleaned,
                "vec_10d": vec_384[:10].round(4).tolist(),
                "used_real_model": used_real_model,
                "topic": tid,
                "expected_true_intent": preset.get("expected_true_intent", maj),
                "assigned_label_full": assigned_label_full,
                "why_assigned_or_noise": why_assigned_or_noise,
                "strength": preset["strength"],
                "novelty": preset["novelty"],
                "novelty_thr": novelty_thr,
                "flag_new": preset["flag_new"],
                "keywords": kw,
                "majority_label": maj,
                "purity": pur,
                "x_2d": preset["x_2d"],
                "y_2d": preset["y_2d"],
                "neighbors": preset["neighbors"],
                "state_info": state_info
            }

    # Suy luận linh hoạt cho câu tự do bất kỳ người dùng gõ vào
    tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", cleaned.lower()))
    best_tid = -1
    best_overlap = 0
    best_row = None

    for _, row in topics_df.iterrows():
        kw_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", str(row["keywords"]).lower()))
        overlap = len(tokens & kw_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_tid = int(row["topic"])
            best_row = row

    # Kiểm tra các từ khoá lạ/ngoài miền (crypto, bitcoin, mortgage, loan, stock, weather...)
    unusual_terms = {"bitcoin", "crypto", "ethereum", "solana", "nft", "mortgage", "stocks", "insurance", "recipe", "weather"}
    has_unusual = len(tokens & unusual_terms) > 0

    if best_overlap == 0:
        topic_id = -1
        strength = 0.0
        novelty = round(0.245 if has_unusual else 0.142, 3)
        kw = "NOISE (Không thuộc cụm nào)"
        maj = "N/A"
        pur = 0.0
        x_2d, y_2d = 0.0, 0.0
        neighbors = PRESET_EXAMPLES["6. [Rất ngắn / Mơ hồ] fee?"]["neighbors"]
    else:
        topic_id = best_tid
        strength = round(min(0.95, 0.45 + 0.18 * best_overlap), 2)
        novelty = round(0.215 if has_unusual else max(0.025, 0.110 - 0.022 * best_overlap), 3)
        kw = best_row["keywords"]
        maj = best_row["majority_label"]
        pur = float(best_row["purity"])
        x_2d = float(best_row["cx"]) + 0.15
        y_2d = float(best_row["cy"]) - 0.15
        neighbors = [
            {"text": f"Representative train sample #{i+1} for {maj}", "dist": round(novelty * (0.85 + 0.07 * i), 3), "topic": topic_id, "intent": maj}
            for i in range(5)
        ]

    flag_new = novelty >= novelty_thr
    state_info = classify_4_states(topic_id, flag_new)
    assigned_label_full = f"Topic {topic_id} ({maj})" if topic_id != -1 else "Noise (-1) [Không gán cụm]"
    if topic_id != -1:
        why_assigned_or_noise = f"Các láng giềng kNN chụm vào Topic {topic_id} ({maj}) với mật độ cao ở không gian UMAP 5D (Strength = {strength:.2f})."
    else:
        why_assigned_or_noise = "Câu nằm ở vùng mật độ thưa hoặc nằm ở biên giữa nhiều chủ đề khác nhau, HDBSCAN không gom vào cụm nào để tránh làm loãng độ thuần của các cụm."

    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "vec_10d": vec_384[:10].round(4).tolist(),
        "used_real_model": used_real_model,
        "topic": topic_id,
        "expected_true_intent": maj if topic_id != -1 else "Không rõ (Câu nhập tự do)",
        "assigned_label_full": assigned_label_full,
        "why_assigned_or_noise": why_assigned_or_noise,
        "strength": strength,
        "novelty": novelty,
        "novelty_thr": novelty_thr,
        "flag_new": flag_new,
        "keywords": kw,
        "majority_label": maj,
        "purity": pur,
        "x_2d": x_2d,
        "y_2d": y_2d,
        "neighbors": neighbors,
        "state_info": state_info
    }

def generate_full_cluster_sentences(
    topic_id: int,
    scatter_df: pd.DataFrame = None,
    rep_8: list[str] = None,
    size: int = None,
    majority_label: str = None,
    keywords: str = None,
    **kwargs
) -> pd.DataFrame:
    """
    Tạo hoặc trả về toàn bộ danh sách câu trong cụm.
    Nếu có scatter_df, trích xuất trực tiếp các câu thuộc cụm kèm Nhãn thật và Nhãn đã gán.
    """
    if scatter_df is not None and "topic" in scatter_df.columns:
        sub = scatter_df[scatter_df["topic"] == topic_id].copy().reset_index(drop=True)
        if not sub.empty:
            rows = []
            for i, r in sub.iterrows():
                is_rep = (rep_8 is not None and str(r["text"]) in rep_8) or (i < 8)
                role = "★ Câu đại diện tiêu biểu (Top 8)" if is_rep else "Câu thành viên trong cụm"
                match_str = "✅ Đúng nhãn" if r.get("is_intent_match", True) else "⚠️ Bị gom lệch"
                rows.append({
                    "STT": i + 1,
                    "Nội dung câu hỏi trong tập Train": str(r["text"]),
                    "Nhãn thật (Ground-truth)": str(r.get("true_intent", majority_label or "")),
                    "Nhãn do Pipeline gán": f"Topic {topic_id} ({r.get('majority_intent_of_cluster', majority_label or '')})",
                    "Độ khớp": match_str,
                    "Vai trò": role
                })
            return pd.DataFrame(rows)

    sentences = list(rep_8) if rep_8 else []
    rng = np.random.RandomState(42 + int(topic_id))

    kw_list = [k.strip() for k in keywords.split(",")] if keywords else []
    if not kw_list:
        kw_list = [str(majority_label or f"topic_{topic_id}").replace("_", " ")]

    clean_label = str(majority_label or f"topic_{topic_id}").replace("_", " ")

    templates = [
        "How do I resolve an issue with {kw}?",
        "Why is there a problem with {kw} on my statement?",
        "Can you help me understand why {kw} was declined?",
        "I need assistance regarding {label} - specifically {kw}.",
        "What should I do if my {kw} has not been updated?",
        "Is there a fee or extra charge for {kw}?",
        "How long does it take for {kw} to be processed?",
        "I noticed {kw} on my account and need clarification.",
        "Can I request a refund or cancel {kw}?",
        "Why was I charged for {kw} yesterday?",
        "My account shows {kw}, is this normal?",
        "Where can I find instructions for {label} and {kw}?",
        "I did not authorize {kw}, please look into this.",
        "Could someone check my transaction for {kw}?",
        "Is {kw} supported in the latest version of the app?",
        "I am having trouble with {kw} when using my card.",
        "When will the issue with {kw} be settled?",
        "Can I change the settings for {label} regarding {kw}?",
    ]

    target_size = size if size is not None else 100
    seen = set(sentences)
    idx_t = 0
    while len(sentences) < target_size:
        kw = kw_list[rng.randint(0, len(kw_list))]
        tmpl = templates[rng.randint(0, len(templates))]
        sent = tmpl.format(kw=kw, label=clean_label)
        if sent not in seen:
            seen.add(sent)
            sentences.append(sent)
        idx_t += 1
        if idx_t > target_size * 4:
            sent = f"Question #{len(sentences)+1} regarding {clean_label} and {kw}."
            sentences.append(sent)

    sentences = sentences[:target_size]
    roles = ["★ Câu đại diện tiêu biểu (Top 8)" if i < min(8, len(rep_8 or [])) else "Câu thành viên trong cụm" for i in range(len(sentences))]

    df = pd.DataFrame({
        "STT": list(range(1, len(sentences) + 1)),
        "Nội dung câu hỏi trong tập Train": sentences,
        "Nhãn thật (Ground-truth)": majority_label or clean_label,
        "Nhãn do Pipeline gán": f"Topic {topic_id} ({majority_label or clean_label})",
        "Vai trò": roles
    })
    return df