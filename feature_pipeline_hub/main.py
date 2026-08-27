"""Launcher: runs the Streamlit UI for the FTI Feature Pipeline Hub."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    app_path = Path(__file__).parent / "ui" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    cmd.extend(sys.argv[1:])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
