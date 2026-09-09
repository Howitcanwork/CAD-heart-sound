"""
styles.py
=========
All CSS for the app lives here as one function so pages stay clean.
Supports a light and a dark palette, switched at runtime via
`st.session_state["dark_mode"]`.
"""

from config import COLORS


def _palette(dark: bool) -> dict:
    if not dark:
        return {
            "bg": COLORS["bg"],
            "surface": COLORS["surface"],
            "text": COLORS["text"],
            "text_muted": COLORS["text_muted"],
            "border": COLORS["border"],
            "card_shadow": "0 8px 24px rgba(15, 27, 45, 0.06)",
            "hero_grad": f"linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['accent']} 100%)",
        }
    return {
        "bg": "#0B1420",
        "surface": "#121D2E",
        "text": "#EAF1F8",
        "text_muted": "#9FB2C8",
        "border": "#1E2C40",
        "card_shadow": "0 8px 24px rgba(0, 0, 0, 0.35)",
        "hero_grad": f"linear-gradient(135deg, {COLORS['primary_dark']} 0%, #0B5F66 100%)",
    }


def inject_global_css(dark_mode: bool = False):
    import streamlit as st

    p = _palette(dark_mode)
    primary = COLORS["primary"]
    accent = COLORS["accent"]
    accent_soft = COLORS["accent_soft"] if not dark_mode else "#0F2B30"

    # EDIT 1 (v2): dark-mode-only override so the Session Result History
    # table (st.table) stays readable. In light mode this string is empty,
    # so the table keeps exactly the appearance it has today.
    table_dark_css = "" if not dark_mode else f"""
    [data-testid="stTable"] {{
        background: {p['surface']};
    }}
    [data-testid="stTable"] table, table.dataframe {{
        color: {p['text']};
        background: {p['surface']};
    }}
    [data-testid="stTable"] th, [data-testid="stTable"] td,
    table.dataframe th, table.dataframe td {{
        color: {p['text']} !important;
        background: {p['surface']} !important;
        border-color: {p['border']} !important;
    }}
    """

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .stApp {{
        background: {p['bg']};
        color: {p['text']};
    }}

    #MainMenu, footer {{visibility: hidden;}}
    header[data-testid="stHeader"] {{background: transparent;}}

    section.main > div.block-container {{
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }}

    h1, h2, h3, h4 {{
        font-family: 'Poppins', 'Inter', sans-serif;
        color: {p['text']};
        letter-spacing: -0.01em;
    }}

    p, li, span, label, div {{
        color: {p['text']};
    }}

    /* ---------------- Sidebar ---------------- */
    section[data-testid="stSidebar"] {{
        background: {p['surface']};
        border-right: 1px solid {p['border']};
    }}
    section[data-testid="stSidebar"] * {{
        color: {p['text']};
    }}

    /* ---------------- Buttons ---------------- */
    .stButton > button, .stDownloadButton > button {{
        background: linear-gradient(135deg, {primary} 0%, {accent} 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1.4rem;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 6px 16px rgba(11, 94, 215, 0.25);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 22px rgba(11, 94, 215, 0.35);
        color: #FFFFFF;
    }}
    .stButton > button:active {{
        transform: translateY(0px);
    }}

    /* ---------------- Generic card ---------------- */
    .cad-card {{
        background: {p['surface']};
        border: 1px solid {p['border']};
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        box-shadow: {p['card_shadow']};
        margin-bottom: 1.1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .cad-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 14px 30px rgba(11, 94, 215, 0.12);
    }}

    .cad-card h4 {{
        margin-top: 0;
        margin-bottom: 0.4rem;
    }}
    .cad-card p {{
        color: {p['text_muted']};
        font-size: 0.92rem;
        line-height: 1.5;
        margin-bottom: 0;
    }}

    /* ---------------- Hero section ---------------- */
    .cad-hero {{
        background: {p['hero_grad']};
        border-radius: 24px;
        padding: 3rem 2.5rem;
        color: #FFFFFF;
        box-shadow: 0 20px 45px rgba(11, 94, 215, 0.25);
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
        animation: fadeSlideIn 0.7s ease;
    }}
    .cad-hero * {{ color: #FFFFFF !important; }}
    .cad-hero h1 {{
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.6rem;
        max-width: 720px;
    }}
    .cad-hero p {{
        font-size: 1.02rem;
        max-width: 640px;
        opacity: 0.95;
    }}
    .cad-hero .cad-badge {{
        display: inline-block;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.35);
        border-radius: 999px;
        padding: 0.3rem 0.9rem;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-bottom: 0.9rem;
    }}

    /* ---------------- Pills / badges ---------------- */
    .cad-pill {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }}
    .pill-success {{ background: #DCFAE6; color: #067647; }}
    .pill-warning {{ background: #FEF0C7; color: #B54708; }}
    .pill-danger  {{ background: #FEE4E2; color: #B42318; }}
    .pill-info    {{ background: {accent_soft}; color: {primary}; }}

    /* ---------------- Diagnosis result card ---------------- */
    .cad-result {{
        border-radius: 20px;
        padding: 1.8rem 2rem;
        background: {p['surface']};
        border: 1px solid {p['border']};
        box-shadow: {p['card_shadow']};
        animation: fadeSlideIn 0.5s ease;
    }}
    .cad-result .big-label {{
        font-size: 1.7rem;
        font-weight: 800;
        font-family: 'Poppins', sans-serif;
    }}

    /* ---------------- Stat tiles (sidebar quick stats) ---------------- */
    .cad-stat {{
        background: {accent_soft};
        border-radius: 14px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.55rem;
    }}
    .cad-stat .stat-value {{
        font-size: 1.15rem;
        font-weight: 800;
        color: {primary};
    }}
    .cad-stat .stat-label {{
        font-size: 0.74rem;
        color: {p['text_muted']};
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}

    /* ---------------- Team card ---------------- */
    .cad-team-card {{
        text-align: center;
        background: {p['surface']};
        border: 1px solid {p['border']};
        border-radius: 18px;
        padding: 1.6rem 1rem;
        box-shadow: {p['card_shadow']};
        transition: transform 0.2s ease;
    }}
    .cad-team-card:hover {{ transform: translateY(-4px); }}
    .cad-avatar {{
        width: 68px; height: 68px;
        border-radius: 50%;
        margin: 0 auto 0.7rem auto;
        display: flex; align-items: center; justify-content: center;
        background: linear-gradient(135deg, {primary} 0%, {accent} 100%);
        color: white; font-weight: 800; font-size: 1.3rem;
    }}

    /* ------------- Footer (EDIT 8: brand-gradient band) ------------- */
    .cad-footer {{
        text-align: center;
        background: {p['hero_grad']};
        color: #FFFFFF;
        font-size: 0.85rem;
        padding: 1.6rem 1.4rem;
        border-radius: 18px;
        box-shadow: 0 12px 30px rgba(11, 94, 215, 0.18);
        margin-top: 2.2rem;
    }}
    .cad-footer * {{ color: #FFFFFF !important; }}

    /* ---------------- Disclaimer banner ---------------- */
    .cad-disclaimer {{
        background: #FFF6E5;
        border: 1px solid #F7C873;
        color: #7A4A00;
        border-radius: 14px;
        padding: 1rem 1.3rem;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 1rem 0;
    }}

    @keyframes fadeSlideIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* ---------------- File uploader ---------------- */
    [data-testid="stFileUploaderDropzone"] {{
        border-radius: 16px !important;
        border: 2px dashed {accent} !important;
        background: {accent_soft} !important;
    }}

    /* ---------------- Progress bar ---------------- */
    .stProgress > div > div > div > div {{
        background: linear-gradient(135deg, {primary} 0%, {accent} 100%);
    }}

    /* Radio nav in sidebar - larger tap targets */
    section[data-testid="stSidebar"] .stRadio > div {{
        gap: 0.15rem;
    }}

    /* --- Session Result History table - dark mode readability only --- */
    {table_dark_css}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
