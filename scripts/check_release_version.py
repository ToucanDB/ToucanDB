"""Validate release versions without importing ToucanDB or its dependencies."""

from __future__ import annotations

import argparse
import ast
import email
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_versions() -> tuple[str, str]:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project_section = pyproject.split("[project]", 1)[1].split("\n[", 1)[0]
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', project_section, re.M)
    if match is None:
        raise ValueError("pyproject.toml [project].version is missing")
    package_version = match.group(1)

    module = ast.parse((ROOT / "toucandb" / "__init__.py").read_text(encoding="utf-8"))
    for node in module.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__version__"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return package_version, node.value.value
    raise ValueError("toucandb.__version__ is missing or is not a string literal")


def wheel_version(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise ValueError(f"{path} contains {len(metadata_names)} METADATA files")
        message = email.message_from_bytes(archive.read(metadata_names[0]))
    version = message.get("Version")
    if not version:
        raise ValueError(f"{path} has no Version metadata")
    return version


def normalized_tag(value: str) -> str:
    return value[1:] if value.startswith("v") else value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected")
    parser.add_argument("--wheel", type=Path)
    arguments = parser.parse_args()

    package_version, runtime_version = source_versions()
    versions = {
        "pyproject.toml": package_version,
        "toucandb.__version__": runtime_version,
    }
    if arguments.expected:
        versions["release tag"] = normalized_tag(arguments.expected)
    if arguments.wheel:
        versions["wheel metadata"] = wheel_version(arguments.wheel)

    expected = package_version
    mismatches = {name: value for name, value in versions.items() if value != expected}
    if mismatches:
        for name, value in versions.items():
            print(f"{name}: {value}", file=sys.stderr)
        return 1

    print(f"Release version verified: {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
