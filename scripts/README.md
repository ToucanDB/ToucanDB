# Development checks

ToucanDB does not use a self-scored “readiness” script. Release confidence comes
from executable gates and documented limitations.

Run from the repository root:

```bash
ruff check .
black --check toucandb tests examples benchmarks
mypy toucandb --ignore-missing-imports
pytest
python -m build
twine check dist/*
```

Use `benchmarks/benchmark.py` for reproducible local regression measurements.
The GitHub workflows run the supported Python matrix and trusted publishing.
