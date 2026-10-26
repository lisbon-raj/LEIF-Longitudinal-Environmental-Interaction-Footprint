"""
LEIF Dashboard Runner.
Launches the interactive web dashboard server for Longitudinal Environmental Interaction Footprint.

Usage:
    python run_dashboard.py
    python run_dashboard.py --port 8050
"""

import sys
import argparse
from leif.dashboard_server import start_server


def main():
    parser = argparse.ArgumentParser(description="LEIF Interactive Dashboard Server")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind server (default: 8050)")
    parser.add_argument("--open", action="store_true", help="Automatically open browser on launch")
    args = parser.parse_args()

    start_server(port=args.port, open_browser=args.open)


if __name__ == "__main__":
    main()
