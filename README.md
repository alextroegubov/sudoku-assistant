# 🧩 Sudoku Assistant

## Motivation  
Sudoku is a great way to relax and exercise your brain without the distractions of a smartphone. However, solving complex grids can be time-consuming. Spending an hour trying to place a single digit can be frustrating—especially when half the puzzle is already filled in.  

The **Sudoku Assistant** is here to help! It can either solve the entire puzzle for you or suggest the next digit to insert.  
All you need to do is take a clear photo of the puzzle and upload it to the app.

---

## General Description  
The **Sudoku Assistant** is a Python-based application that combines classical computer vision and deep learning to:  
- Detect and extract Sudoku grids from images.  
- Recognize handwritten or printed digits.  
- Solve Sudoku puzzles automatically or provide hints without revealing the full solution.  

---

### 🚀 Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/alextroegubov/sudoku-assistant.git
   cd sudoku-assistant
   ```

2. **Create a virtual environment** with venv, miniconda or any other preferred tool


3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

### 📖 How to Use
1. **Run the Streamlit UI**:
   ```bash
   streamlit run src/gui/app.py
   ```

2. **Upload an image**:  
   - Provide a clear Sudoku puzzle image.
   - The app will process the image and recognize the digits.

3. **Confirm the grid**:  
   - If everything is alright, confirm the grid. Otherwise, remove the grid and upload a new image.

4. **Solve or Get Hints**:  
   - Click "Solve" to compute the full solution.  
   - Click "Show Tip" to reveal the next best move.

---

## ⚙️ Technical Details

Technological stack: OpenCV, PyTorch, streamlit

- **Image Processing**: Uses OpenCV to preprocess images, remove the grid, and extract digits.
- **Digit Classification**: A convolutional neural network (CNN) trained with `timm` classifies digits 1-9.
- **Sudoku Solver**: Implements a backtracking algorithm with optional step-by-step solving.
- **GUI**: Streamlit provides an intuitive web interface.

---

## 🎥 Demo
*Include a GIF or screenshot showcasing the app in action.*

