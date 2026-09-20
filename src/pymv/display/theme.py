"""Shared display styling, kept with the UI code rather than image assets."""

CSS = """
:root { --pymv-burgundy: #8A1538; --pymv-slate: #334155; }
.stApp { color: #334155; background: #FFFFFF; }
[data-testid="stSidebar"] { background: #F8FAFC; border-right: 1px solid #E2E8F0; }
.block-container { max-width: 1440px; padding-top: 2rem; }
h1, h2, h3 { color: #334155; font-weight: 600; letter-spacing: -.02em; }
.stApp [data-testid="stMarkdownContainer"] a { color: #8A1538; }
[data-testid="stExpander"] { border-color: #E2E8F0; border-radius: 6px; }
button[kind="primary"] { background-color: #8A1538; border-color: #8A1538; color: white; }
[data-baseweb="tab-highlight"] { background-color: #8A1538; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #526174; }
.pymv-brand { text-align: center; margin: -.5rem 0 1rem; }
.pymv-brand h1 { margin-bottom: 0; padding-bottom: .2rem; }
.pymv-brand p { color: #64748b; font-size: .85rem; }
[data-testid="stSidebar"] .stButton > button {
  min-height: 2.9rem; border-radius: .65rem; border: 1px solid #e2e8f0;
  transition: background-color .15s ease, border-color .15s ease;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background-color: #f3e8ec; border-color: #8A1538; color: #8A1538;
}
"""
