"""Light scientific styling using the shared PolyClustX palette."""

from polyclustx.brand import BRAND, PLOT_COLORS, css_variables


def style_figure(figure):
    """Apply the palette to a figure and its exported copy, not global Plotly state."""
    figure.update_layout(
        template="plotly_white", colorway=PLOT_COLORS,
        paper_bgcolor=BRAND["surface"], plot_bgcolor=BRAND["surface"],
        font_color=BRAND["text"],
        legend={"bgcolor": BRAND["surface"], "font_color": BRAND["text"]},
    )
    figure.update_xaxes(gridcolor=BRAND["border"], zerolinecolor=BRAND["border"])
    figure.update_yaxes(gridcolor=BRAND["border"], zerolinecolor=BRAND["border"])
    return figure

CSS = css_variables() + """
.stApp { color: var(--pcx-text); background: var(--pcx-background); }
[data-testid="stSidebar"] { background: var(--pcx-surface); border-right: 1px solid var(--pcx-border); }
.block-container { max-width: 1440px; padding-top: 2rem; }
h1, h2, h3 { color: var(--pcx-text); font-weight: 600; letter-spacing: -.02em; }
.stApp [data-testid="stMarkdownContainer"] a { color: var(--pcx-primary); }
.stApp [data-testid="stMarkdownContainer"] a:hover { color: var(--pcx-primary-dark); }
[data-testid="stExpander"] { background: var(--pcx-surface); border-color: var(--pcx-border); border-radius: 6px; }
button[kind="primary"] { background: var(--pcx-primary); border-color: var(--pcx-primary); color: var(--pcx-surface); }
[data-baseweb="tab-highlight"] { background: var(--pcx-primary); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--pcx-text-muted); }
.polyclustx-brand { text-align: center; margin: -.5rem 0 1rem; }
.polyclustx-brand h1 { margin-bottom: 0; padding-bottom: .2rem; font-size: 1.9rem; color: var(--pcx-primary); }
.polyclustx-brand p { color: var(--pcx-text-muted); font-size: .85rem; }
.st-key-sidebar_navigation { margin-inline: -1rem; width: calc(100% + 2rem); }
.st-key-sidebar_navigation, .st-key-sidebar_navigation [data-testid="stVerticalBlock"] { gap: 0; }
.st-key-sidebar_navigation .stButton > button {
  min-height: 3.1rem; border-radius: 0; border: 0; margin: 0;
  justify-content: flex-start; padding-inline: 1.25rem; box-shadow: none;
  transition: background-color .15s ease, color .15s ease;
}
.st-key-sidebar_navigation .stButton > button[kind="secondary"] { background: transparent; }
.st-key-sidebar_navigation .stButton > button[kind="primary"] {
  background: var(--pcx-brand-tint); color: var(--pcx-primary-dark);
  box-shadow: inset 3px 0 var(--pcx-primary); font-weight: 600;
}
.st-key-sidebar_navigation .stButton > button:hover { background: var(--pcx-surface-alt); color: var(--pcx-primary); }
.st-key-sidebar_navigation .stButton > button:focus-visible {
  outline: 2px solid var(--pcx-accent); outline-offset: -2px;
}
.st-key-affiliations { background: var(--pcx-surface); padding: .6rem; border-radius: 6px; }
.st-key-affiliations [data-testid="stImage"] img { height: 86px; object-fit: contain; }
.st-key-affiliations [data-testid="stMarkdownContainer"] p { font-size: .75rem; text-align: center; }
"""
