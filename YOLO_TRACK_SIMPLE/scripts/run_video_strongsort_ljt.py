#!/usr/bin/env python3
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.pipeline.pipeline_ljt import main


if __name__ == "__main__":
    if "--tracker" not in sys.argv:
        sys.argv.extend(["--tracker", "strongsort"])
    main()
