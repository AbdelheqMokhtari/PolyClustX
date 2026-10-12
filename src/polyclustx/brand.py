"""Lightweight shared identity; no UI or numerical dependencies."""

NAME = "PolyClustX"
DEFINITION = (
    "A Unified Framework for Multi-View and Multimodal Representation Learning "
    "and Clustering from Feature Views to Attributed Graphs"
)
REPOSITORY_URL = "https://github.com/AbdelheqMokhtari/PolyClustX"
DOCUMENTATION_URL = "https://polyclustx.readthedocs.io/en/latest/"

BRAND = {
    "primary": "#4E4E92",
    "primary_dark": "#3B3D73",
    "primary_light": "#7478AE",
    "secondary": "#634E87",
    "accent": "#557AA3",
    "background": "#F7F8FC",
    "surface": "#FFFFFF",
    "surface_alt": "#EEF0F7",
    "border": "#D7DAE8",
    "text": "#292B3A",
    "text_muted": "#6B6E7D",
    "brand_tint": "#E8E9F3",
}

PLOT_COLORS = [BRAND[key] for key in ("primary", "accent", "secondary", "primary_light", "text_muted")]


def css_variables():
    """CSS custom properties shared by Sphinx and Streamlit."""
    return ":root {" + "".join(f"--pcx-{key.replace('_', '-')}: {value};" for key, value in BRAND.items()) + "}"
