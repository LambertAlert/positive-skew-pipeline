"""
run_massive.py — one-command launcher for the positive-skew pipeline on Massive data.

    python run_massive.py               full run: pull data, walk-forward backtest, score next session, dashboard
    python run_massive.py --live-only   daily: score the next session (needs one full run first)
    python run_massive.py --dashboard-only
    python run_massive.py --horizons 5 10 --test-sessions 126     (any pipeline.run option passes through)

The API key comes from MASSIVE_API_KEY or POLYGON_API_KEY; if neither is set you're asked for it
(it is used for this run only, not saved). Results: data/psp_out/dashboard.html
"""

import getpass
import os
import sys
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)                      # paths like data/psp_cache are relative to this folder
sys.path.insert(0, str(HERE))

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--dashboard-only" not in args and "--source" not in args:
        if not (os.environ.get("MASSIVE_API_KEY") or os.environ.get("POLYGON_API_KEY")):
            key = getpass.getpass("Massive API key: ").strip()
            if not key:
                sys.exit("No key given.")
            os.environ["MASSIVE_API_KEY"] = key
    from pipeline.run import main
    from pipeline.massive_client import MassiveError
    try:
        main(args)
    except MassiveError as e:
        sys.exit(f"\nMassive error: {e}")
    out = Path("data/psp_out/dashboard.html")
    if "--out" in args:
        out = Path(args[args.index("--out") + 1]) / "dashboard.html"
    if out.exists():
        print(f"\nDashboard: {out.resolve()}")
        try:
            webbrowser.open(out.resolve().as_uri())
        except Exception:
            pass
