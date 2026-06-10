"""Tiny YAML subset loader, used only when PyYAML isn't installed.

The rule files ship with slurm-doctor and are written in a restricted
subset: a top-level list of flat mappings, one optional nested mapping
level (``match:``), flow lists of plain scalars, and single/double quoted
strings.  Unit tests assert parity with PyYAML on every shipped rule file.
"""

from __future__ import annotations

import re

try:  # pragma: no cover - exercised implicitly
    import yaml as _pyyaml
except ImportError:  # pragma: no cover
    _pyyaml = None


def _scalar(tok: str):
    tok = tok.strip()
    if not tok or tok in ("null", "~", "None"):
        return None
    if tok.startswith("'") and tok.endswith("'") and len(tok) >= 2:
        return tok[1:-1].replace("''", "'")
    if tok.startswith('"') and tok.endswith('"') and len(tok) >= 2:
        return tok[1:-1].encode().decode("unicode_escape")
    low = tok.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if re.fullmatch(r"[+-]?\d+", tok):
        return int(tok)
    if re.fullmatch(r"[+-]?\d*\.\d+(e[+-]?\d+)?", tok, re.I):
        return float(tok)
    if tok.startswith("[") and tok.endswith("]"):
        inner = tok[1:-1].strip()
        if not inner:
            return []
        # split on commas outside quotes
        parts, buf, q = [], "", None
        for ch in inner:
            if q:
                buf += ch
                if ch == q:
                    q = None
            elif ch in "'\"":
                q = ch
                buf += ch
            elif ch == ",":
                parts.append(buf)
                buf = ""
            else:
                buf += ch
        parts.append(buf)
        return [_scalar(p) for p in parts]
    return tok


_LINE = re.compile(r"^(?P<indent>\s*)(?P<dash>-\s+)?(?P<key>[A-Za-z0-9_.-]+):(?:\s+(?P<val>.*))?$")


def safe_load(text: str):
    """Parse the restricted subset. Returns a list of dicts (or {})."""
    if _pyyaml is not None:
        return _pyyaml.safe_load(text)
    items: list = []
    # stack of (indent, container) for nested dicts
    stack: list[tuple[int, dict]] = []
    for raw in text.splitlines():
        line = raw.split(" #", 1)[0].rstrip() if not raw.lstrip().startswith("#") else ""
        if not line.strip():
            continue
        m = _LINE.match(line)
        if not m:
            raise ValueError(f"mini-yaml cannot parse line: {raw!r}")
        indent = len(m.group("indent")) + (len(m.group("dash") or ""))
        key, val = m.group("key"), m.group("val")
        if m.group("dash"):
            current = {}
            items.append(current)
            stack = [(indent, current)]
        else:
            while stack and indent < stack[-1][0]:
                stack.pop()
            if not stack:
                raise ValueError(f"mini-yaml: unexpected top-level key {key!r}")
            # same or deeper indent attaches to the innermost open dict
            current = stack[-1][1]
        if val is None or val == "":
            child: dict = {}
            current[key] = child
            stack.append((indent + 2, child))
        else:
            current[key] = _scalar(val)
    return items
