# ui/theme.py
import streamlit as st

LIGHT = {
    "--rex-bg": "#FFFFFF",
    "--rex-ink": "#101316",
    "--rex-muted": "#5E6B78",
    "--rex-border": "#E7EAEE",
    "--rex-card": "#FFFFFF",
    "--rex-primary": "#2FD3C6",
    "--rex-success": "#17B26A",
    "--rex-warning": "#F59E0B",
    "--rex-danger": "#EF4444",
    "--rex-accent": "#7C3AED",
}

DARK = {
    "--rex-bg": "#0B0E11",
    "--rex-ink": "#F3F4F6",
    "--rex-muted": "#A3AAB3",
    "--rex-border": "#1E242B",
    "--rex-card": "#12161B",
    "--rex-primary": "#2FD3C6",
    "--rex-success": "#22C55E",
    "--rex-warning": "#F59E0B",
    "--rex-danger": "#EF4444",
    "--rex-accent": "#8B5CF6",
}

def _vars(theme: dict) -> str:
    return "\n".join(f"{k}: {v};" for k, v in theme.items())

def apply_theme(dark: bool = False):
    theme = DARK if dark else LIGHT
    st.markdown(
        f"""
        <style>
        :root {{
            {_vars(theme)}
            --rex-radius: 16px; --rex-pad: 20px;
        }}
        html, body, .stApp {{
            background: var(--rex-bg);
            color: var(--rex-ink);
            font-feature-settings: "tnum" 1, "ss01" 1;
        }}
        /* Cards */
        .stContainer, .stTabs [data-baseweb="tab-list"] + div {{ gap: 12px; }}
        .rex-card {{
            background: var(--rex-card);
            border: 1px solid var(--rex-border);
            border-radius: var(--rex-radius);
            padding: var(--rex-pad);
            box-shadow: 0 1px 2px rgba(16,19,22,.04);
        }}
        /* Metrics */
        .rex-metric h4 {{ margin: 0 0 6px 0; color: var(--rex-muted); font-weight: 600; }}
        .rex-metric p {{ margin: 0; font-size: 22px; font-weight: 700; }}
        /* Tabs pills */
        [data-baseweb="tab"] button {{
            border-radius: 9999px; padding: 6px 14px;
        }}
        /* Ocultar footer y menú */
        footer, #MainMenu {{ display: none; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def theme_toggle(label="🌗 Dark mode"):
    if "rex_dark" not in st.session_state:
        st.session_state["rex_dark"] = False
    st.session_state["rex_dark"] = st.sidebar.toggle(label, value=st.session_state["rex_dark"])
    apply_theme(st.session_state["rex_dark"])
