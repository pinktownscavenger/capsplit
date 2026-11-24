"""Leave out the language-model tests when the `lm` extra is not installed.

The extra pins transformers 5.23.0, which needs Python 3.11+, so on Python 3.10 the LM code
cannot be imported at all. CI installs the extra on 3.12, where these tests run.
"""
import importlib.util

collect_ignore = [] if all(importlib.util.find_spec(m) for m in ("transformers", "peft")) else ["lm"]
