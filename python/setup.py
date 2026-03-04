"""
sglang-kt setup.py — provides version at build time.

Version is read from the SGLANG_KT_VERSION environment variable,
which is set by ktransformers' install.sh and CI workflows
(sourced from ktransformers/version.py).

When building standalone (without the env var), falls back to "0.0.0.dev0".
"""
import os
from setuptools import setup

version = os.environ.get("SGLANG_KT_VERSION", "0.0.0.dev0")

setup(version=version)
