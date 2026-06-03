"""Pytest config: put the backend/ directory on sys.path so the
`navigation` package imports as a top-level module from inside tests/.
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
