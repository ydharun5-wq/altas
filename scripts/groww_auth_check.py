"""Retired Groww integration entry point.

ATLAS V1 is paper-trading only. This compatibility stub deliberately does not
read environment variables, access the network, or save credentials.
"""

from __future__ import annotations


def main() -> int:
    print(
        "Groww integration is disabled. ATLAS V1 does not use broker credentials."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
