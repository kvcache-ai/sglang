"""Runner script that discovers and executes all KSE unit tests.

Usage (from the repo root):
    python test/srt/kse/run_all_kse_tests.py
"""

import os
import sys
import unittest

# Ensure sglang package and this directory are importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "python"))
sys.path.insert(0, _HERE)  # for mock_utils


def main():
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=_HERE, pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
