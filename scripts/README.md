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

## Promo media

Regenerate the README promo video and poster from the repository root:

```bash
pip install '.[media]'
python scripts/generate_promo_video.py
```

The generator also requires `ffmpeg` on `PATH`. It writes the H.264/AAC video
and PNG poster to `assets/promo/`. The background music is synthesized by the
generator itself and contains no samples or third-party audio.
