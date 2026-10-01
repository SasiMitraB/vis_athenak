"""Parser for AthenaK input (athinput) files."""

from __future__ import annotations

import re
from pathlib import Path


def parse_athinput(path: str | Path) -> dict[str, dict[str, str]]:
    """
    Parse an athinput file into ``{section: {key: value}}``.

    Section names are lower-cased, inline ``# comments`` are stripped, and
    values are left as strings.
    """
    params: dict[str, dict[str, str]] = {}
    section = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            m = re.match(r"^<\s*([\w-]+)\s*>", line)
            if m:
                section = m.group(1).lower()
                params.setdefault(section, {})
            elif section is not None and "=" in line:
                key, value = line.split("=", 1)
                params[section][key.strip()] = value.strip()
    return params
