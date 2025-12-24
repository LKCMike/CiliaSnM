# CiliaStatisticAndMeasurement
A YOLOv8 and PyQt5 based software for cilia statistic and measurement. Currently it only supports the OME-TIFF format.

# Getting Started
This guide provides instructions for running CiliaStatisticAndMeasurement on different systems.

For 64-bit Windows 7+ Users
It is recommended to download the latest .zip archive from the Releases page. Extract the archive and run the software directly.

For Linux Users
You can run the software from the Python source code using the `run.sh` script provided in this repository.

Note: If you encounter dependency issues, please follow these steps:

- First, use your system's package manager (e.g., apt for Debian/Ubuntu, dnf for RedHat-based systems) to install the necessary development library dependencies.

- Then, run the `cleanup.sh` script provided in this repository to clean up the virtual environment.

- Finally, execute the `run.sh` script again.

For Users Who Prefer to Run Manually from Python
Please refer to the following steps:

```bash
# Create Python virtual environment
python -m venv venv

# Activate Python virtual environment for Linux users
source venv/bin/activate

# Or for Windows users with cmd.exe
.\venv\Scripts\activate.bat
# Or for Windows users with Powershell
.\venv\Scripts\Activate.ps1

# Upgrade pip, setuptools and install requirements
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# There is a conflict between OpenCV and PyQt5 to deal with for Linux users.
pip uninstall -y opencv-python
pip install opencv-python-headless==4.12.0.88

# Launch app.
python main.py
```

And we also provide `main.spec` for packaging using pyinstaller.

# Model Construction
This repository provides a pre-built model for immediate use. For users interested in reproducing or customizing the model, we have included the dataset and related scripts in the `transfer_learning` folder.

To rebuild the model, follow these steps:

Install PyTorch: Please install the version of PyTorch that matches your specific environment (graphics card and CUDA version). You can find the appropriate installation command on the official PyTorch website.

Install YOLOv8: The simplest way is to execute `pip install ultralytics==8.3.235` in a Python virtual environment.

Run the Build Script: Navigate to the `transfer_learning` directory and execute the `build_model.py` script.
