"""Entry point for ``python -m omen``."""

from __future__ import annotations

import sys

from omen.cli import main

if __name__ == "__main__":
    sys.exit(main())
