#!/bin/bash

# Safe exit
error_exit() {
    echo "Error: $1" >&2
    exit 1
}

# Change to script directory
cd "$(dirname "$0")" || error_exit "Failed to change to script directory"

# Check Virtual environment deployment
if [ -d "venv" ]; then
    echo "Virtual environment 'venv' found."
    # launch app
    if venv/bin/python main.py; then
        echo "App is running now."
        exit 0
    else
        # Invalid virtual environment
        echo "Failed to launch main.py with existing virtual environment."
        echo "Please run cleanup.sh and run this script again."
        exit 1
    fi
fi

# Check Python command and version
PYTHON_CMD=""
PYTHON_VERSION=""

if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    error_exit "Python is not installed. Please install Python 3.10 or higher."
fi

PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
MAJOR_VERSION=$($PYTHON_CMD -c "import sys; print(sys.version_info[0])")
MINOR_VERSION=$($PYTHON_CMD -c "import sys; print(sys.version_info[1])")

# Version too low
if [ $MAJOR_VERSION -lt 3 ] || [ $MAJOR_VERSION -eq 3 -a $MINOR_VERSION -lt 10 ]; then
    error_exit "Python version $PYTHON_VERSION is too low. Please install Python 3.10 or higher."
fi

echo "Using Python $PYTHON_VERSION"

# Create virtual environment
echo "Creating virtual environment..."
if ! $PYTHON_CMD -m venv venv; then
    error_exit "Failed to create virtual environment. Make sure venv module is installed."
fi

# Upgrade pip, setuptools and wheel
echo "Upgrading pip, setuptools, wheel..."
if ! venv/bin/python -m pip install --upgrade pip setuptools wheel; then
    error_exit "Failed to upgrade pip, setuptools, wheel."
fi

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing packages from requirements.txt..."
    if ! venv/bin/python -m pip install -r requirements.txt; then
        error_exit "Failed to install packages from requirements.txt."
    fi
else
    echo "requirements.txt not found, skipping package installation."
fi

# OpenCV PyQt Conflict resolve
venv/bin/python -m pip uninstall -y opencv-python
venv/bin/python -m pip install opencv-python-headless==4.12.0.88

# Launch app
echo "Executing main.py..."
venv/bin/python main.py
echo "App is running now."
