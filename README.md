# CS380 Artificial Intelligence - Setup & Usage Guide

This repository contains implementations of various AI algorithms and example implementations.

## Prerequisites

- Python 3.13 or higher installed on your system
- Git for cloning the repository

## Setup

### Option 1: Using VSCode (Recommended)

#### Step 1: Open the Project in VSCode

1. Open VSCode
2. Go to **File → Open Folder**
3. Navigate to the `CS380ArtificialIntelligence` directory and click **Select Folder**

#### Step 2: Create a Virtual Environment

1. Open the integrated terminal in VSCode:
   - **View → Terminal** (or press `Ctrl + ``)
2. Run the following command to create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Wait for the virtual environment to be created (this may take a minute)

#### Step 3: Activate the Virtual Environment

1. In the integrated terminal, activate the virtual environment:
   ```bash
   venv\Scripts\activate
   ```
   - On macOS/Linux, use: `source venv/bin/activate`
   - You should see `(venv)` appear at the beginning of the terminal prompt

2. VSCode may also prompt you to select the Python interpreter:
   - Click **Yes** if prompted, or manually select the interpreter:
   - Open the Command Palette (**Ctrl + Shift + P**)
   - Type `Python: Select Interpreter`
   - Choose the one that shows `./venv` in the path

#### Step 4: Install Required Libraries

1. Install dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
2. Wait for all packages to install

#### Step 5: Run a Program

1. Navigate to the desired script location in the file explorer
   - Example: `Chapter 3 - Solving Problems With Search/BreadthFirstSearch.py`
2. Right-click the file and select **Run Python File in Terminal**
   - Or use the play button (▶) in the top-right corner of the editor
3. The program will execute in the integrated terminal

---

##### Run Using Terminal

```bash
python "Chapter 3 - Solving Problems With Search/BreadthFirstSearch.py"
```

Replace the filename with any script you want to run.

---