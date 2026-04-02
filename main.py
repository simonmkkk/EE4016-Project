#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

# Add project root to Python path to ensure modules are importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.menu import interactive_menu


def main():
    """Main entry point - launches interactive menu."""
    interactive_menu()


if __name__ == "__main__":
    main()

