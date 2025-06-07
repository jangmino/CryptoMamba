#!/bin/bash

# Define the base directory (equivalent to ${workspaceFolder})
# Assuming this script is run from the project's root directory.
# If not, you might need to specify the absolute path or adjust BASE_DIR.
BASE_DIR=$(pwd)

# Path to the Python script to be executed
PROGRAM="${BASE_DIR}/scripts/training.py"

# Arguments for the Python script
# Note: Arguments are carefully quoted to handle spaces if they were present (not in this case, but good practice).
ARGS=(
    "--config"
    "aiden_lstm_v"
    "--logger_type"
    "wandb"
    "--expname"
    "cmamba_lstm_v"
    "--save_checkpoints"
    "--batch_size=256"
    "--num_workers=8"
)

# --- Execution ---

# Check if the Python script exists
if [ ! -f "$PROGRAM" ]; then
    echo "Error: Python script not found at $PROGRAM"
    exit 1
fi

VENV_PATH="/home/work/vision_data/work/cmamba/.venv/bin/activate"
if [ -f "$VENV_PATH" ]; then
    echo "Activating virtual environment..."
    source "$VENV_PATH"
else
    echo "No virtual environment found at $VENV_PATH. Using system Python or currently active environment."
fi

echo "Running Python script: $PROGRAM with arguments: ${ARGS[*]}"
python "$PROGRAM" "${ARGS[@]}"

# (Optional) Deactivate the virtual environment after execution
deactivate