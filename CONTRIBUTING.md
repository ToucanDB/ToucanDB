# Contributing to ToucanDB

Thank you for your interest in contributing to ToucanDB! We welcome contributions from the community and are excited to see what you'll build with us.

## 🌟 Ways to Contribute

- **Code**: New features, bug fixes, performance improvements
- **Documentation**: Improve guides, add examples, fix typos
- **Testing**: Write tests, report bugs, improve coverage
- **Design**: UI/UX improvements, logos, diagrams
- **Community**: Help others, answer questions, write tutorials

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment (recommended)

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/pH-7/toucandb.git
   cd toucandb
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run tests to verify setup**
   ```bash
   pytest tests/ -v
   ```

5. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

### Project Structure

```
toucandb/
├── toucandb/           # Core library code
│   ├── __init__.py     # Main API
│   ├── types.py        # Type definitions
│   ├── exceptions.py   # Custom exceptions
│   ├── schema.py       # Schema management
│   └── vector_engine.py # Vector operations
├── tests/              # Test suite
├── examples/           # Usage examples
├── docs/               # Documentation
└── benchmarks/         # Performance tests
```

## 📝 Development Guidelines

### Code Style

- **Python**: Follow PEP 8, use Black for formatting
- **Type hints**: All public APIs must have type annotations
- **Docstrings**: Use Google-style docstrings
- **Imports**: Use isort for import organization

### Example Code Style

```python
from typing import List, Optional, Dict, Any
from datetime import datetime

class VectorCollection:
    """A collection of vectors with search capabilities.
    
    Args:
        name: Collection name
        schema: Vector schema definition
        storage_path: Path for persistent storage
        
    Raises:
        InvalidSchemaError: If schema is invalid
        StorageError: If storage initialization fails
    """
    
    def __init__(
        self, 
        name: str, 
        schema: VectorSchema,
        storage_path: Path
    ) -> None:
        self.name = name
        self.schema = schema
        self._validate_schema()
    
    async def search(
        self, 
        query: SearchQuery
    ) -> OperationResult[List[SearchResult]]:
        """Search for similar vectors.
        
        Args:
            query: Search parameters and vector
            
        Returns:
            Operation result with search results
        """
        # Implementation here
        pass
```

### Commit Messages

Use conventional commits format:

```
type(scope): description

body (optional)

footer (optional)
```

Types:
- `feat`: New feature
- `fix`: Bug fix  
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `perf`: Performance improvements
- `ci`: CI/CD changes

Examples:
- `feat(search): add metadata filtering support`
- `fix(storage): resolve memory leak in vector cache`
- `docs(readme): update installation instructions`

### Testing

- **Unit tests**: Test individual components
- **Integration tests**: Test component interactions
- **Performance tests**: Benchmark critical paths
- **Coverage**: Aim for >90% code coverage

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=toucandb --cov-report=html

# Run specific test file
pytest tests/test_vector_engine.py -v

# Run performance tests
pytest benchmarks/ -k "performance"
```

### Documentation

- **API docs**: Auto-generated from docstrings
- **Tutorials**: Step-by-step guides in `docs/`
- **Examples**: Practical code examples in `examples/`
- **README**: Keep updated with new features

## 🐛 Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. **Environment**:
   - Python version
   - ToucanDB version
   - Operating system
   - Hardware specs (for performance issues)

2. **Reproduction steps**:
   - Minimal code example
   - Expected vs actual behavior
   - Error messages and stack traces

3. **Additional context**:
   - Data size/complexity
   - Performance characteristics
   - Workarounds tried

### Feature Requests

For new features, please provide:

1. **Use case**: What problem does this solve?
2. **Proposed solution**: How should it work?
3. **Alternatives**: Other approaches considered
4. **Impact**: Who would benefit?

## 🔄 Pull Request Process

### Before Submitting

1. **Create an issue** first to discuss the change
2. **Fork the repository** and create a feature branch
3. **Write tests** for new functionality
4. **Update documentation** as needed
5. **Run the full test suite**
6. **Check code style** with pre-commit hooks

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Added unit tests
- [ ] Added integration tests
- [ ] All tests pass
- [ ] Performance benchmarks (if applicable)

## Documentation
- [ ] Updated docstrings
- [ ] Updated README
- [ ] Added examples
- [ ] Updated API docs

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] No breaking changes (or documented)
```

### Review Process

1. **Automated checks**: CI/CD must pass
2. **Code review**: At least one maintainer approval
3. **Testing**: Comprehensive test coverage
4. **Documentation**: Complete and accurate docs

## 🎯 Areas for Contribution

### High Priority

- **Performance optimization**: SIMD, GPU acceleration
- **Storage formats**: Compression, serialization
- **Index algorithms**: New search methods
- **Security features**: Authentication, authorization
- **Cloud integration**: AWS, GCP, Azure support

### Medium Priority

- **Language bindings**: Rust, Go, JavaScript
- **Monitoring**: Metrics, observability
- **Backup/restore**: Data management tools
- **CLI tools**: Command-line interface
- **Web interface**: Management dashboard

### Documentation Needs

- **API reference**: Complete method documentation
- **Tutorials**: Beginner to advanced guides
- **Best practices**: Performance, security tips
- **Deployment guides**: Production setup
- **Migration guides**: Version upgrade help

## 🏆 Recognition

Contributors will be:

- **Listed**: In CONTRIBUTORS.md
- **Credited**: In release notes
- **Featured**: On project website (for major contributions)
- **Invited**: To join the maintainer team (for ongoing contributors)

## 📞 Getting Help

- **Discussions**: GitHub Discussions for questions
- **Discord**: Join our community server
- **Email**: maintainers@toucandb.org
- **Issues**: GitHub Issues for bugs/features

## 🤝 Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

### Our Standards

- **Be respectful**: Treat everyone with kindness
- **Be constructive**: Provide helpful feedback
- **Be patient**: Support newcomers
- **Be inclusive**: Welcome diverse perspectives

## 📋 Development Workflow

### Feature Development

1. **Plan**: Create issue and get feedback
2. **Branch**: Create feature branch from main
3. **Develop**: Write code with tests
4. **Test**: Ensure all tests pass
5. **Document**: Update relevant docs
6. **Review**: Submit PR for review
7. **Merge**: Maintainer merges after approval

### Release Process

1. **Version bump**: Update version numbers
2. **Changelog**: Update CHANGELOG.md
3. **Testing**: Run full test suite
4. **Build**: Create release artifacts
5. **Tag**: Create git tag
6. **Publish**: Deploy to PyPI
7. **Announce**: Update documentation and notify community

## 🛠️ Tools and Resources

### Development Tools

- **IDE**: VSCode with Python extension
- **Linting**: ruff, mypy
- **Formatting**: black, isort
- **Testing**: pytest, pytest-cov
- **Documentation**: mkdocs, mkdocs-material

### Useful Commands

```bash
# Format code
black toucandb/ tests/
isort toucandb/ tests/

# Lint code
ruff check toucandb/ tests/
mypy toucandb/

# Build documentation
mkdocs serve

# Run benchmarks
python benchmarks/run_benchmarks.py
```

Thank you for contributing to **ToucanDB**! Together, we're building the future of vector databases 🦜✨