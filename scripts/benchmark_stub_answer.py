#!/usr/bin/env python3
from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) < 2:
        return 1
    _prompt_file = sys.argv[1]
    print("STUB_ANSWER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
