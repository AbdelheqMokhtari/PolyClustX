"""Optional interactive workbench, installed with pymv-clustering[display]."""


def launch():
    """Launch Streamlit; additional CLI arguments are forwarded to Streamlit."""
    import importlib.util
    import subprocess
    import sys
    from pathlib import Path

    if importlib.util.find_spec("streamlit") is None:
        raise SystemExit('Install the interface with: pip install "pymv-clustering[display]"')
    app = Path(__file__).with_name("app.py")
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app),
                "--theme.base=light",
                "--theme.primaryColor=#8A1538",
                "--theme.backgroundColor=#FFFFFF",
                "--theme.secondaryBackgroundColor=#F8FAFC",
                "--theme.textColor=#334155",
                *sys.argv[1:],
            ]
        )
    )
