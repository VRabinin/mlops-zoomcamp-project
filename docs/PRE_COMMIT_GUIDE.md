# Pre-commit Configuration Guide

This document explains the pre-commit hooks configured for the MLOps Platform project.

## Quick Start

1. **Install pre-commit hooks:**
   ```bash
   make precommit-install
   ```

2. **Run hooks manually on all files:**
   ```bash
   make precommit-run
   ```

3. **Update hook versions:**
   ```bash
   make precommit-update
   ```

## Configured Hooks

### 🧹 Code Cleanup
- **trailing-whitespace**: Removes trailing whitespace
- **end-of-file-fixer**: Ensures files end with a newline
- **mixed-line-ending**: Normalizes line endings to LF

### 🔍 File Validation
- **check-yaml**: Validates YAML syntax
- **check-json**: Validates JSON syntax
- **check-toml**: Validates TOML syntax
- **check-merge-conflict**: Detects merge conflict markers
- **check-added-large-files**: Prevents committing large files (>10MB)

### 🐍 Python Code Quality
- **black**: Code formatting (88 character line length)
- **isort**: Import sorting with black compatibility
- **flake8**: Linting with additional plugins:
  - flake8-docstrings (documentation)
  - flake8-builtins (built-in shadowing)
  - flake8-comprehensions (comprehension improvements)
  - flake8-pytest-style (pytest best practices)

### 🔧 Type Checking
- **mypy**: Static type checking with additional type stubs
- Excludes: tests/, notebooks/, .venv/

### 🔒 Security
- **bandit**: Security vulnerability scanner
- **detect-secrets**: Prevents committing secrets
- Creates baseline file: `.secrets.baseline`

### 📓 Notebooks
- **nbqa-black**: Formats code in Jupyter notebooks
- **nbqa-isort**: Sorts imports in notebooks
- **nbqa-flake8**: Lints notebook code

### 🐳 Infrastructure
- **hadolint**: Dockerfile linting
- **docker-compose-check**: Validates docker-compose files
- **terraform_fmt**: Formats Terraform files
- **terraform_validate**: Validates Terraform configuration

### 📝 Documentation
- **prettier**: YAML formatting
- **terraform_docs**: Updates Terraform documentation

## Configuration Details

### Exclusions
Pre-commit automatically excludes:
- Virtual environments (`.venv/`)
- Data directories (`data/`, `models/`, `artifacts/`)
- Generated files (`__pycache__/`, `*.egg-info/`)
- Local environment files (`.env*`)
- Application data (`app_data/`)

### Line Length
- **Black**: 88 characters (Python standard)
- **Flake8**: 88 characters (consistent with Black)
- **MyPy**: Configured for Python 3.11

### Security Baseline
The `.secrets.baseline` file contains known false positives for secret detection. Update this file when you add legitimate strings that might be flagged as secrets.

## Integration with CI/CD

Pre-commit hooks run locally before commits, but the same checks are also run in CI/CD:

```bash
make ci-test  # Runs tests, linting, and security checks
```

The GitHub Actions workflow (`.github/workflows/ci.yml`) includes:
- Linting with flake8
- Type checking with mypy
- Security scanning with bandit and safety
- Test coverage with pytest

## Troubleshooting

### Hook Failures
If a hook fails:
1. Fix the issues reported
2. Stage the fixed files: `git add .`
3. Commit again

### Skip Hooks (Emergency)
To skip pre-commit hooks (not recommended):
```bash
git commit --no-verify -m "commit message"
```

### Update Hooks
Keep hooks up to date:
```bash
make precommit-update
```

### Uninstall Hooks
If needed:
```bash
make precommit-uninstall
```

## Best Practices

1. **Run hooks before committing**: They catch issues early
2. **Fix issues promptly**: Don't accumulate technical debt
3. **Keep hooks updated**: Regular updates include security fixes
4. **Use consistent formatting**: Let black and isort handle formatting
5. **Write type hints**: MyPy helps catch type-related bugs
6. **Document your code**: Flake8-docstrings encourages good documentation

## Performance

Pre-commit hooks run only on staged files, making them fast. The first run after installation may be slower as hooks download dependencies.

## Support

For issues with pre-commit hooks:
1. Check the pre-commit output for specific error messages
2. Review this documentation
3. Run `make precommit-run` to test hooks manually
4. Update hooks with `make precommit-update`

The hooks are configured to balance code quality with development velocity, following MLOps best practices for the platform.
