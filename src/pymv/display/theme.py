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
.st-key-sidebar_navigation { margin-inline: -1rem; width: calc(100% + 2rem); }
.st-key-sidebar_navigation, .st-key-sidebar_navigation [data-testid="stVerticalBlock"] { gap: 0; }
.st-key-sidebar_navigation .stButton > button {
  min-height: 3.1rem; border-radius: 0; border: 0; margin: 0;
  justify-content: flex-start; padding-inline: 1.25rem; box-shadow: none;
  transition: background-color .15s ease, border-color .15s ease;
}
.st-key-sidebar_navigation .stButton > button[kind="secondary"] { background: transparent; }
.st-key-sidebar_navigation .stButton > button:hover {
  background-color: #f3e8ec; color: #8A1538;
}
"""
