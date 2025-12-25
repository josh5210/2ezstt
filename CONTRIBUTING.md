# Contributing to EZSTT

Thank you for your interest in contributing to EZSTT! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/ezstt.git`
3. Create a branch: `git checkout -b feature/your-feature`

## Development Setup

### Python Sidecar (STT Server)
```bash
cd ezstt
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### TypeScript Client
```bash
npm install
```

## Code Style

- **Python**: Follow PEP 8, use type hints
- **TypeScript**: Use strict TypeScript, follow existing patterns

## Testing

Run tests before submitting:

```bash
# Python tests
python -m pytest tests/ -v

# TypeScript (if applicable)
npm test
```

## Pull Request Process

1. Ensure tests pass
2. Update documentation if needed
3. Add a clear PR description
4. Wait for review

## Reporting Issues

- Use GitHub Issues
- Include reproduction steps
- Include environment details (OS, Python version, Node version)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
