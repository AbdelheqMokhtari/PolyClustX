"""Interactive interface included in the standard installation."""


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
    try:
        code = subprocess.call(command, env=environment)
    except KeyboardInterrupt:
        # Ctrl+C is the documented stop action, not a launch failure.
        code = 0
    raise SystemExit(code)
