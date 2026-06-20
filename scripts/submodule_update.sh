#!/bin/bash
# Script to initialize and pull the latest code for git submodules,
# and automatically ensure they are placed in the vendors/ directory.

# Exit on any error
set -e

# Navigate to the project root directory (assuming script is in the script/ directory)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Fetching latest changes from the main repository..."
git pull

# Automatically detect and move any submodules that are not inside vendors/
if [ -f .gitmodules ]; then
    echo "Checking submodule paths..."
    # Read all submodule paths
    while read -r path; do
        if [[ -n "$path" && "$path" != vendors/* ]]; then
            echo ">> Found submodule in root: '$path'"
            echo ">> Automatically moving it to 'vendors/$path'..."
            mkdir -p vendors
            
            git submodule update --init "$path" >/dev/null 2>&1 || true
            
            git mv "$path" "vendors/$path"
            echo ">> Moved successfully!"
        fi
    done < <(git config --file .gitmodules --get-regexp 'submodule\..*\.path' | awk '{print $2}' || true)
fi

echo "Initializing and updating submodules to their registered commits..."
git submodule update --init --recursive

echo "Pulling the latest updates from the remote tracking branches of the submodules..."
git submodule update --remote --merge

echo "Submodule pull complete!"