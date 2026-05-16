#!/usr/bin/env python3
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from apps.gui.gui_ljt import main


if __name__ == "__main__":
    main()
