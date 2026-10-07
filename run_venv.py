#!/usr/bin/env python3
"""Wrapper that runs a command with the venv python to avoid namespace conflicts."""
import subprocess
import sys
from pathlib import Path

VENV_PYTHON = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"

if not VENV_PYTHON.exists():
    print("ERROR: venv python not found at {}".format(VENV_PYTHON))
    sys.exit(1)

cmd = [str(VENV_PYTHON)] + sys.argv[1:]
result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent))
sys.exit(result.returncode)