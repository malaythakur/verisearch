#!/usr/bin/env bash
# Task 17.6: Publish Python SDK to PyPI
# Usage: ./scripts/publish.sh [--test]
#
# Prerequisites:
#   - PYPI_TOKEN environment variable set
#   - poetry installed
#
# Use --test flag to publish to TestPyPI instead

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SDK_DIR"

# Build the package
echo "Building Python SDK..."
poetry build

# Determine target repository
if [[ "${1:-}" == "--test" ]]; then
    echo "Publishing to TestPyPI..."
    poetry config repositories.testpypi https://test.pypi.org/legacy/
    poetry publish --repository testpypi --username __token__ --password "${PYPI_TEST_TOKEN:-$PYPI_TOKEN}"
else
    echo "Publishing to PyPI..."
    poetry publish --username __token__ --password "$PYPI_TOKEN"
fi

echo "Done! Python SDK published successfully."
