# Contributing to ToucanDB

Thank you for helping improve ToucanDB. Changes should preserve its embedded
boundary: correctness and measured resource use come before feature breadth.

## Development setup

Prerequisites:

- Python 3.10 or newer;
- Git; and
- a C/C++ compatible platform supported by the `faiss-cpu` wheel.

```bash
git clone https://github.com/ToucanDB/ToucanDB.git
cd ToucanDB
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

## Required local checks

```bash
ruff check .
black --check toucandb tests examples benchmarks scripts
mypy toucandb --ignore-missing-imports
pytest
python -m build
twine check dist/*
```

Run a fresh-wheel import test for packaging changes. CI repeats tests on Python
3.10 through 3.14.

## Repository layout

```text
toucandb/                 public package and engine
toucandb/integrations/    optional-dependency-light adapters
tests/                    unit and integration tests
examples/                 runnable examples
benchmarks/               reproducible regression benchmark
docs/                     architecture, RAG, migration, and tuning guides
.github/workflows/        CI and trusted PyPI publishing
```

## Engineering rules

- Add type annotations to public APIs and keep `py.typed` accurate.
- Keep imports side-effect free. A library must not configure root logging,
  event-loop policy, global threads, or network clients at import time.
- Load large models and optional SDKs lazily.
- Do not add a core dependency when the standard library or an injected adapter
  provides the boundary.
- Validate a logical multi-batch operation before its first durable write.
- Treat SQLite as authoritative and FAISS as rebuildable.
- Preserve database compatibility or provide a documented, tested migration.
- Bound caches and candidate expansion; do not introduce an unbounded queue or
  resident copy.
- Include source attribution and authorization boundaries in retrieval work.
- Never add unverifiable performance claims. Include the benchmark command,
  hardware, versions, dataset, filters, tuning, and recall method.

## Tests expected by change type

- Storage: reopen, interrupted/stale snapshot behavior, backup, ID types, and
  encryption where relevant.
- Index: flat ground truth plus index/metric combinations and small-corpus edge
  cases.
- RAG/integrations: idempotent sync, changed-content replacement, pruning,
  namespace or tenant isolation, and provider failure behavior.
- Async work: prove blocking work is off the event loop or otherwise bounded.
- Packaging: build sdist/wheel, inspect contents, install the wheel in a clean
  environment, and import public APIs.

Coverage is a signal, not a substitute for failure-path assertions.

## Pull requests

1. Open an issue for a large design or on-disk-format change.
2. Branch from `main`.
3. Keep commits focused and explain the user-visible outcome.
4. Add tests and documentation in the same pull request.
5. Run all required checks.
6. Include migration, security, memory, and performance consequences in the PR
   description.

The PR should state:

- the problem and chosen boundary;
- alternatives considered;
- tests and benchmark commands run;
- compatibility or migration impact;
- new dependencies and why they are necessary; and
- remaining limitations.

## Reporting bugs

Use [GitHub Issues](https://github.com/ToucanDB/ToucanDB/issues). Include a
minimal reproduction, ToucanDB/Python/FAISS/NumPy versions, operating system and
architecture, index schema, vector count/dimensions, encryption state, expected
behavior, and the complete error. Never attach an encryption key or sensitive
database.

For a suspected vulnerability, follow [SECURITY.md](SECURITY.md) rather than
opening a public issue.

## Recognition

Material contributors are credited in release notes and, for sustained work,
in [CREDITS.md](CREDITS.md). ToucanDB was created and is maintained by
[Pierre-Henry Soria](https://pierrehenry.dev).
