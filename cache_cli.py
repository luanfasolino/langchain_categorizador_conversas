#!/usr/bin/env python3
"""
Script de entrada para a Cache CLI.
"""

import sys
from pathlib import Path

# Adiciona src ao path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from cache_cli import main

if __name__ == "__main__":
    main()