# utils/format_vi.py
import math
import numpy as np

def fmt_int_vi(val) -> str:
    """Định dạng số nguyên dùng dấu chấm phân cách hàng nghìn (vd: 9993 -> '9.993')."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    try:
        n = int(round(float(val)))
        return f"{n:,}".replace(",", ".")
    except (ValueError, TypeError):
        return str(val)

def fmt_num_vi(val, decimals: int = 3, signed: bool = False) -> str:
    """
    Định dạng số thực kiểu Việt: dấu chấm ngăn nghìn, dấu phẩy thập phân.
    Ví dụ: 0.833 -> '0,833' | 7604.5 -> '7.604,500'
    """
    if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
        return "N/A"
    try:
        f_val = float(val)
        sign_str = "+" if (signed and f_val > 0) else ""
        formatted = f"{f_val:,.{decimals}f}"
        # Chuyển đổi ',' và '.' sang chuẩn Việt Nam
        formatted = formatted.replace(",", "TEMP").replace(".", ",").replace("TEMP", ".")
        return f"{sign_str}{formatted}"
    except (ValueError, TypeError):
        return str(val)

def fmt_pct_vi(val, decimals: int = 2, is_ratio: bool = True) -> str:
    """
    Định dạng phần trăm kiểu Việt.
    Nếu is_ratio=True: 0.0741 -> '7,41%'. Nếu is_ratio=False: 7.41 -> '7,41%'.
    """
    if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
        return "N/A"
    try:
        f_val = float(val) * 100.0 if is_ratio else float(val)
        return f"{fmt_num_vi(f_val, decimals=decimals)}%"
    except (ValueError, TypeError):
        return str(val)

def fmt_ci_vi(low, high, decimals: int = 4) -> str:
    """Định dạng khoảng tin cậy: [0,0396; 0,0479]."""
    return f"[{fmt_num_vi(low, decimals)}; {fmt_num_vi(high, decimals)}]"