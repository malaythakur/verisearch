#!/usr/bin/env bash
# Task 17.6: Publish TypeScript SDK to npm
# Usage: ./scripts/publish.sh [--dry-run]
#
# Prerequisites:
#   - NPM_TOKEN environment variable set (or npm login)
#   - pnpm installed
#
# Use --dry-run flag to simulate publish without actually publishing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SDK_DIR"

# Build the package
echo "Building TypeScript SDK..."
pnpm run build

# Set npm auth token if provided
if [[ -n "${NPM_TOKEN:-}" ]]; then
    echo "//registry.npmjs.org/:_authToken=${NPM_TOKEN}" > .npmrc
fi

# Publish
if [[ "${1:-}" == "--dry-run" ]]; then
    echo "Dry run publish..."
    npm publish --dry-run
else
    echo "Publishing to npm..."
    npm publish --access public
fi

# Clean up .npmrc if we created it
if [[ -n "${NPM_TOKEN:-}" ]]; then
    rm -f .npmrc
fi

echo "Done! TypeScript SDK published successfully."
