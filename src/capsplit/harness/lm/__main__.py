"""`python -m capsplit.harness.lm <stage> CONFIG` (see cli.py)."""
import sys

from .cli import main

sys.exit(main())
