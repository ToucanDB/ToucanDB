#!/usr/bin/env python3
"""Script to fix linting issues in ToucanDB codebase."""

import re
from pathlib import Path


def fix_typing_imports(content: str) -> str:
    """Replace deprecated typing imports with modern equivalents."""
    # Replace Dict -> dict, List -> list, Set -> set, Tuple -> tuple
    content = re.sub(r'\bDict\[', 'dict[', content)
    content = re.sub(r'\bList\[', 'list[', content)
    content = re.sub(r'\bSet\[', 'set[', content)
    content = re.sub(r'\bTuple\[', 'tuple[', content)
    
    # Update import statements
    content = re.sub(
        r'from typing import ([^\n]+)',
        lambda m: fix_import_line(m.group(1)),
        content
    )
    
    return content


def fix_import_line(imports: str) -> str:
    """Fix a single import line."""
    # Remove Dict, List, Set, Tuple from imports if present
    deprecated = ['Dict', 'List', 'Set', 'Tuple']
    parts = [p.strip() for p in imports.split(',')]
    parts = [p for p in parts if p not in deprecated]
    
    if parts:
        return f'from typing import {", ".join(parts)}'
    else:
        return ''


def fix_trailing_whitespace(content: str) -> str:
    """Remove trailing whitespace from lines."""
    lines = content.split('\n')
    lines = [line.rstrip() for line in lines]
    return '\n'.join(lines)


def add_newline_at_eof(content: str) -> str:
    """Ensure file ends with a newline."""
    if content and not content.endswith('\n'):
        content += '\n'
    return content


def fix_file(filepath: Path) -> None:
    """Fix linting issues in a single file."""
    print(f"Fixing {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # Apply fixes
    content = fix_typing_imports(content)
    content = fix_trailing_whitespace(content)
    content = add_newline_at_eof(content)
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✓ Fixed {filepath}")
    else:
        print(f"  - No changes needed for {filepath}")


def main():
    """Main function."""
    toucandb_dir = Path(__file__).parent / 'toucandb'
    
    py_files = list(toucandb_dir.glob('*.py'))
    
    for filepath in py_files:
        if filepath.name != '__pycache__':
            fix_file(filepath)
    
    print("\n✓ Linting fixes applied!")


if __name__ == '__main__':
    main()
