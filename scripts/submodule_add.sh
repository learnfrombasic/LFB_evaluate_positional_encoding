#!/bin/bash
# Script to safely add a git submodule directly into the vendors/ directory

set -e

if [ -z "$1" ]; then
    echo "Error: Missing repository URL."
    echo "Usage: $0 <github_repo_url>"
    echo "Example: $0 https://github.com/facebookresearch/ijepa.git"
    exit 1
fi

REPO_URL="$1"

# Extract the repository name from the URL
# Removes any trailing .git and grabs the last part of the URL
REPO_NAME=$(basename -s .git "$REPO_URL")

echo "Adding submodule '$REPO_NAME' to 'vendors/$REPO_NAME'..."

git submodule add -f "$REPO_URL" "vendors/$REPO_NAME"

echo "Successfully added $REPO_NAME to vendors!"