"""
HuggingFace configuration.

Reads the HF_TOKEN from the environment variable HUGGINGFACE_TOKEN (or HF_TOKEN).
Set it before running the application:
    export HF_TOKEN="hf_..."        # Linux / macOS
    $env:HF_TOKEN = "hf_..."        # PowerShell
"""

import os

HF_TOKEN: str | None = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
