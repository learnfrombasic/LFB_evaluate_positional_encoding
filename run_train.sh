#!/bin/bash

# Configuration file defaults to config.yaml if not provided
CONFIG_FILE=${1:-config.yaml}

# Locate python bin within the environment or fallback to system python
PYTHON_BIN="python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

echo "Using Python executable: $PYTHON_BIN"
echo "Running training pipeline using configuration: $CONFIG_FILE"

# Execute the module
"$PYTHON_BIN" -m src --config "$CONFIG_FILE" --mode train
