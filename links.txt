# EntrySync - Vehicle Detection & Recognition System

This guide provides complete instructions for setting up and running the EntrySync model, including environment setup, dependency installation, and usage instructions for both Linux/WSL and Windows.

## Prerequisites

- **OS**: Windows 10/11 or Linux (Ubuntu 20.04+ / WSL2)
- **Python**: Version 3.10 or higher (3.12 recommended)
- **Hardware**: NVIDIA GPU with CUDA support is highly recommended for real-time performance.

---

## Installation Guide

### 1. System Preparation (Linux / WSL Only)

If you are using Linux or WSL (Windows Subsystem for Linux), update your system and install necessary system libraries for GUI support.

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install system dependencies for OpenCV and GUI support
sudo apt install libgl1 libglib2.0-0 libgtk-3-dev python3-venv -y
```

### 2. Set Up Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

**Linux / macOS:**
```bash
# Create virtual environment
python3 -m venv <venv_name>

# Activate virtual environment
source <venv_name>/bin/activate
```

**Windows:**
```powershell
# Create virtual environment
python -m venv <venv_name>

# Activate virtual environment
.\<venv_name>\Scripts\Activate.ps1
```

### 3. Install PyTorch

Install the appropriate version of PyTorch based on your hardware.

**Option A: For NVIDIA GPU (Recommended)**
*Check your CUDA version (e.g., 12.1) and adjust the URL if needed.*
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Option B: For CPU Only**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

> **Note:** If you need to switch from CPU to GPU, uninstall the current version first:
> `pip uninstall torch torchvision torchaudio`

### 4. Install Project Dependencies

Install the remaining required libraries for YOLO, OCR, and the GUI.

```bash
pip install ultralytics ncnn easyocr customtkinter pillow opencv-python
```

> **Troubleshooting OpenCV:**
> If you encounter errors related to `opencv-python-headless`, you may need to uninstall it and reinstall the full version:
> ```bash
> pip uninstall opencv-python opencv-python-headless
> pip install opencv-python
> ```

---

## Model Setup

The system uses a YOLO model exported to NCNN format for optimized performance.

1.  **Export Model (If needed):**
    If you have a trained `.pt` model (e.g., `best.pt`), export it to NCNN format:
    ```bash
    yolo export model=best.pt format=ncnn
    ```
    *This will create a `best_ncnn_model` directory.*

2.  **Verify Model Path:**
    Ensure the `best_ncnn_model` folder is present in the root directory of the project.

---

## Usage

### Option 1: Run the Dashboard (GUI)

The main application with a graphical user interface for monitoring multiple streams.

```bash
python build/dashboard.py
```

### Option 2: Run Standalone Detection Script

Run the detection script directly on a specific source (camera, video file, or RTSP stream).

**Basic Camera Usage:**
```bash
python pc_cam_v0.3.py --resolution 640x480
```

**RTSP Stream Usage:**
```bash
python pc_cam_v0.3.py --resolution 640x480 --source "rtsp://admin:admin1234@192.168.0.10:554/cam/realmonitor?channel=1&subtype=1"
```

**Arguments:**
- `--source`: Input source. Defaults to `0` (webcam). Can be a video file path or RTSP URL.
- `--resolution`: Display resolution (e.g., `640x480`).
- `--model`: Path to model (defaults to `best_ncnn_model` in the script).

---

## Troubleshooting

- **WSL GUI Issues:** If the GUI doesn't appear on WSL, ensure you have a strictly compatible X Server installed (like VcXsrv) or use WSLg (included in Windows 11).
- **CUDA Not Detected:** Run `python -c "import torch; print(torch.cuda.is_available())"` to verify PyTorch can see your GPU.
- **EasyOCR Performance:** EasyOCR will automatically use CUDA if available. On first run, it may download necessary model files.

### Advanced: Building OpenCV with CUDA (Windows)

To enable CUDA support in OpenCV for Windows, you must build it from source.

1.  **Prerequisites**:
    - **Visual Studio** (2019+) with C++ tools.
    - **CMake** (GUI version).
    - **CUDA Toolkit & cuDNN** (compatible with your GPU).
    - **OpenCV Source**: Download `opencv` and `opencv_contrib` (same version) from GitHub.

2.  **Configure with CMake**:
    - Set source code and build directories.
    - **Configure**: Select Visual Studio version and `x64`.
    - **Enable**: `WITH_CUDA`, `CUDA_FAST_MATH`, `WITH_CUBLAS`, `BUILD_opencv_world`.
    - **Set**: `OPENCV_EXTRA_MODULES_PATH` to the `modules` folder in `opencv_contrib`.
    - **Python**: Ensure `BUILD_opencv_python3` is checked and paths are correct.
    - **Generate**: Create the Visual Studio solution.

3.  **Build & Install**:
    - Open `OpenCV.sln` in Visual Studio.
    - Set configuration to **Release**.
    - Build **ALL_BUILD**, then build **INSTALL**.

4.  **Setup**:
    - Add `build\install\x64\vc14\bin` to system `PATH`.
    - Verify in Python: `import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())`.


setting up pi

connect wifi
sudo su
sudo apt update
sudo apt upgrade

install vs code
sudo apt install code

install git
sudo apt install git

install python
sudo apt install python3
sudo apt install python3-pip

install virtualenv
sudo apt install virtualenv

CUDA Toolkit Installer	
Installation Instructions:
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/13.1.0/local_installers/cuda-repo-wsl-ubuntu-13-1-local_13.1.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-13-1-local_13.1.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-13-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-1



