#!/bin/bash

# Change to script directory
cd "$(dirname "$0")" || error_exit "Failed to change to script directory"
# Remove virtual environment
echo "Remove virtual environment..."
rm -rf venv
echo "Virtual environment removed."
