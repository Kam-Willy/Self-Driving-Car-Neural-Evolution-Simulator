"""
Self-Driving Car Neural Evolution Simulator
==========================================
Entry point – simply launches the PyQt5 application.

Run with:
    python main.py
"""

import sys
import os

# Ensure the project root is on the path so absolute imports work
sys.path.insert(0, os.path.dirname(__file__))

from visualization.renderer import run_app

if __name__ == "__main__":
    run_app()
