"""Interactive interface included in the standard installation."""


def _startup_line(line):
    """Rebrand only known startup text; keep server URLs and diagnostics intact."""
    message = line.strip()
    if message == "You can now view your Streamlit app in your browser.":
        from polyclustx import __version__

        return f"  PolyClustX {__version__} | Interactive interface\n"
    if message.startswith(("URL: ", "Local URL: ")):
        url = message.split(": ", 1)[1]
        return f"  Open in browser: {url}\n  Press Ctrl+C to stop.\n"
    if message == "Help agents write better Streamlit apps?" or message.startswith(
        "Install the official Streamlit skills by running "
    ):
        return ""
    return line


def launch(arguments=None):
    """Launch Streamlit; additional CLI arguments are forwarded to Streamlit."""
    import importlib.util
    import os
    import subprocess
    import sys
    from pathlib import Path

    from polyclustx.brand import BRAND

    if importlib.util.find_spec("streamlit") is None:
        raise SystemExit('Repair the installation with: python -m pip install --upgrade polyclustx')
    app = Path(__file__).with_name("app.py")
    # Small minibatches suffer from oversubscription. Set defaults only in the
    # interface subprocess, leaving embedding applications and explicit settings alone.
    environment = os.environ.copy()
    threads = environment.get("POLYCLUSTX_CPU_THREADS", "2")
    if not threads.isdigit() or int(threads) < 1:
        raise SystemExit("POLYCLUSTX_CPU_THREADS must be a positive integer")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment.setdefault(variable, threads)
    # Stream output promptly even though stdout is piped through our launcher.
    environment["PYTHONUNBUFFERED"] = "1"
    command = [
        sys.executable, "-m", "streamlit", "run", str(app),
        "--server.address=127.0.0.1",
        "--browser.gatherUsageStats=false",
        "--theme.base=light",
        f"--theme.primaryColor={BRAND['primary']}",
        f"--theme.backgroundColor={BRAND['background']}",
        f"--theme.secondaryBackgroundColor={BRAND['surface_alt']}",
        f"--theme.textColor={BRAND['text']}",
        *(sys.argv[1:] if arguments is None else arguments),
    ]
    # Filter only the banner on stdout. Stderr is inherited unchanged so CLI
    # errors, runtime warnings and tracebacks remain visible. Use the URL that
    # Streamlit actually reports, including custom ports and base paths.
    with subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, text=True, bufsize=1) as process:
        try:
            previous_blank = False
            for line in process.stdout:
                rendered = _startup_line(line)
                if not rendered or (not rendered.strip() and previous_blank):
                    continue
                print(rendered, end="", flush=True)
                previous_blank = not rendered.strip()
            code = process.wait()
        except KeyboardInterrupt:
            # Ctrl+C stops the child too; never leave a background server behind.
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            code = 0
    raise SystemExit(code)
