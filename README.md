# Yolo_PaddleOCR

A user-friendly desktop app for OCR and AI chat, combining YOLO object detection, PaddleOCR text recognition, and Gemini AI chat, built with PyQt6. Easy setup, secure API key management, and ready for open-source distribution.

---

## Features

- **YOLOv8 Object Detection**: Detects objects in images.
- **PaddleOCR**: Extracts text from detected regions.
- **Gemini AI Chat**: Interact with Gemini AI for advanced chat and analysis.
- **Modern PyQt6 UI**: Simple, clean, and user-friendly interface.
- **Secure API Key Management**: Enter and save your Gemini API key securely.
- **One-click Setup**: Automated environment creation and app launch.

---

## Quick Start (Windows)

### 1. Prerequisites

#### Install Chocolatey (if not already installed)
Chocolatey is a Windows package manager that makes installing software easy.

Open an **Administrator Command Prompt** and run:

```cmd
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

#### Install Python 3.10+ (if not already installed)

After installing Chocolatey, run in **Command Prompt** (not bash):

```cmd
choco install python --version=3.10.11 -y
```

Or download manually from [python.org](https://www.python.org/downloads/).

#### Add Python to PATH
- Make sure to check "Add Python to PATH" during installation, or add it manually if needed.

#### Install Git (if not already installed)

```cmd
choco install git -y
```

---

### 2. Clone the Repository

```bash
git clone https://github.com/0NE-C0DEMAN/Yolo_PaddleOCR.git
cd Yolo_PaddleOCR
```

### 3. One-Click Setup & Run

Just double-click `setup_and_run.bat` in the project folder.  
This will:
- Create a virtual environment (if not present)
- Install all dependencies
- Launch the app

**If you see a message about activating the virtual environment, follow the instructions in the terminal.**

---

## Configuration

- On first run, enter your Gemini API key in the app’s settings panel.
- The key is saved securely in `config/config.json` (this file is ignored by git for your privacy).

---

## File Structure

- `src/` — Main application code
- `models/` — YOLOv8 model weights and related files
- `output/` — Output JSON files (ignored by git)
- `config/config.json` — Stores your Gemini API key (ignored by git)
- `setup_and_run.bat` — One-click setup and launch script
- `requirements.txt` / `requirements.lock.txt` — Dependency lists

---

## Notes

- All sensitive files, model weights, outputs, and your API key are protected by `.gitignore`.
- For advanced users: you can manually activate the virtual environment and run the app with:
  ```bash
  .venv\Scripts\activate
  python src/main.py
  ```

---

## Contributing

Pull requests and issues are welcome!

---

## License

[MIT License](LICENSE)
