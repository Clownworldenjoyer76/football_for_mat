#!/usr/bin/env python3
"""Wire the Issue 28 market-exclusion audit into current training entrypoints.

This is an idempotent local source transformer. It inserts the preflight as the
first executable statement in each top-level main() under scripts/train/*.py.
Helper modules without a top-level main() are left unchanged.
"""

from __future__ import annotations

from pathlib import Path
import ast
import json
import os
import tempfile
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import common


_CONFIG_CONTRACT = common.load_config()

MARKER = "# ISSUE28_MARKET_EXCLUSION_PREFLIGHT"
BLOCK = [
    f"    {MARKER}\n",
    "    _issue28_audit = common.prop_root() / \"scripts\" / \"validate\" / \"audit_market_exclusion.py\"\n",
    "    _issue28_result = __import__(\"subprocess\").run(\n",
    "        [__import__(\"sys\").executable, str(_issue28_audit), \"--preflight\"],\n",
    "        check=False,\n",
    "    )\n",
    "    if _issue28_result.returncode != 0:\n",
    "        raise RuntimeError(\"Issue 28 market-exclusion preflight failed.\")\n",
    "\n",
]


def atomic_write(path: Path, text: str) -> None:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temp_path = Path(handle.name)
    try:
        with handle:
            handle.write(text)
        os.replace(temp_path, path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def top_level_main(tree: ast.Module) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    return None


def insertion_line(main_node: ast.FunctionDef) -> int:
    if not main_node.body:
        raise ValueError("main() has no body")
    first = main_node.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        if len(main_node.body) >= 2:
            return int(main_node.body[1].lineno)
        return int(first.end_lineno or first.lineno) + 1
    return int(first.lineno)


def wire_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return "already_wired"
    if "import common" not in text:
        raise ValueError(f"Training entrypoint does not import common: {path}")

    tree = ast.parse(text, filename=str(path))
    main_node = top_level_main(tree)
    if main_node is None:
        return "helper_skipped"

    lines = text.splitlines(keepends=True)
    line_no = insertion_line(main_node)
    lines[line_no - 1:line_no - 1] = BLOCK
    updated = "".join(lines)
    ast.parse(updated, filename=str(path))
    atomic_write(path, updated)
    return "wired"


def main() -> int:
    train_root = common.prop_root() / "scripts" / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(f"Training directory does not exist: {train_root}")

    rows: list[dict[str, str]] = []
    for path in sorted(train_root.glob("*.py")):
        result = wire_file(path)
        rows.append({"file": path.name, "result": result})

    entrypoints = [row for row in rows if row["result"] != "helper_skipped"]
    if not entrypoints:
        raise RuntimeError("No training entrypoints with main() were found.")

    payload = {
        "status": "passed",
        "training_entrypoints": len(entrypoints),
        "wired": sum(row["result"] == "wired" for row in rows),
        "already_wired": sum(row["result"] == "already_wired" for row in rows),
        "files": rows,
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
