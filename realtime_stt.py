#!/usr/bin/env python3
"""
Shim for running the voice assistant from the root directory.
Redirects to src/voiceassistant/main.py
"""
import sys
import os

# Add src to sys.path to allow importing the voiceassistant package
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from voiceassistant.main import main

if __name__ == "__main__":
    main()
