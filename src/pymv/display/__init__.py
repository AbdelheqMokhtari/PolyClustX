"""Interactive interface included in the standard installation."""


def launch(arguments=None):
    """Launch Streamlit; additional CLI arguments are forwarded to Streamlit."""
    import importlib.util
    import subprocess
    import sys
    from pathlib import Path

    if importlib.util.find_spec("streamlit") is None:
        raise SystemExit('Repair the installation with: python -m pip install --upgrade pymv-clustering')
    app = Path(__file__).with_name("app.py")
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
            str(app),
            "--server.address=127.0.0.1",
            "--browser.gatherUsageStats=false",
                "--theme.base=light",
                "--theme.primaryColor=#8A1538",
                "--theme.backgroundColor=#FFFFFF",
                "--theme.secondaryBackgroundColor=#F8FAFC",
                "--theme.textColor=#334155",
                *(sys.argv[1:] if arguments is None else arguments),
            ]
        )
    )
