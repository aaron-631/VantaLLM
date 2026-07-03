# Contributing to VantaLLM

Thank you for your interest in contributing to VantaLLM! This framework is built for experimentation and we welcome pull requests from the community.

## How to Contribute

1. **Fork the Repository**: Create your own fork and branch off `main`.
2. **Setup the Environment**: Follow the setup instructions in the `README.md`.
3. **Make Changes**: Keep your changes focused and atomic.
4. **Testing**: Run local smoke tests before submitting. Use `TINY_MODEL=1` in your environment to ensure the training loop executes properly.
5. **Pull Requests**: Submit a PR with a clear description of your changes, the rationale behind architectural adjustments, and what tests you ran.

## Coding Standards

- **Clarity over Abstraction**: Code should be readable PyTorch, not hidden behind massive class hierarchies.
- **Type Hints**: Please use standard Python type hints for clarity.
- **Docstrings**: Document all new modules or major functions.

## Reporting Bugs

Please use GitHub Issues to report bugs. Include:
- A clear description of the bug.
- Steps to reproduce.
- Your environment details (GPU type, CUDA version, PyTorch version).

We look forward to building better infrastructure with you!
