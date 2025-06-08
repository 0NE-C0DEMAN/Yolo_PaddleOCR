# Yolo PaddleOCR UI Application

A desktop application for UI element detection and OCR using YOLOv8, PaddleOCR, and Gemini AI, with a modern PyQt6 interface.

## Environment Setup (Windows)

### 1. Install Python 3.10 or 3.11

- Download from [python.org](https://www.python.org/downloads/).
- During installation, **add Python to PATH**.
- Verify installation:
  ```bash
  python --version
  ```

### 2. Install Chocolatey (Windows Package Manager)

- Open **PowerShell as Administrator**.
- Run:
  ```powershell
  Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
  ```
- Verify:
  ```powershell
  choco --version
  ```

### 3. Install ccache (Optional, for faster C/C++ builds)

- In **PowerShell as Administrator**:
  ```powershell
  choco install ccache -y
  ```
- ccache will be installed to `C:\ProgramData\chocolatey\lib\ccache\tools\ccache.exe` and added to PATH automatically.

### 4. Unzip the Project Folder

- Download and unzip the project folder to your desired location (e.g., `D:\Yolo_PaddleOCR`).
- Open a terminal in the unzipped folder.

### 5. Create and Activate a Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 6. Install uv (Recommended)

- Install uv for fast, reliable package management:
  ```bash
  pip install uv
  ```

### 7. Install Python Packages

- **Recommended:** Use [uv](https://github.com/astral-sh/uv) for fast, reliable installs:
  ```bash
  uv pip install -r requirements.lock.txt
  ```
- Or, with pip:
  ```bash
  pip install -r requirements.lock.txt
  ```
- If you want to update dependencies, use `requirements.txt` instead.

### 8. Reactivate the Virtual Environment (IMPORTANT)

- After installing packages, you may need to activate the environment again in a new terminal:
  ```bash
  .venv\Scripts\activate
  ```

### 9. Run the Application

```bash
python src/main.py
```

---

## Notes

- Make sure your Python, pip, and all dependencies are installed in the same environment.
- If you encounter DLL or dependency errors, ensure you are using the correct Python version and all packages are installed in the active virtual environment.
- For GPU support, install CUDA and cuDNN, and ensure compatible versions with torch and paddlepaddle.

## Troubleshooting

- If you see errors about missing DLLs or packages, try reinstalling dependencies in a clean virtual environment.
- For protobuf/onnx errors, ensure `protobuf==3.20.3` is installed (already pinned in lock file).
- For model loading errors, check that model files exist in `models/yolov8m_for_ocr/weights/`.
