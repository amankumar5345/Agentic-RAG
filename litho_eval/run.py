"""CLI entry point: python -m litho_eval.run ..."""

from __future__ import annotations

import sys

from litho_eval.runner import main

if __name__ == "__main__":
    sys.exit(main())
