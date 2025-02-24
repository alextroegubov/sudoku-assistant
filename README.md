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
0. **Run tests**:
    ```bash
    PYTHONPATH=. pytest
    ```

    TBD: one test fails for classificator

1. **Run the Streamlit UI**:
   ```bash
   streamlit run main.py
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

- **Image Processing**: OpenCV

    The aim of preprocessing is to extract and prepare images of digits for classification.
    
    The *first step* is field extraction: grayscale convertion, gaussian blur, adaptive thresholding, selecting the largest rectangular contour, perspective correction, grid lines removal. Finally, morphological operations and binary thresholding are performed.

    The *second step* is digits extraction: splitting field into cells, removing noise for each cell, simple filtering to skip empty or too noise cells.

- **Digit Classification**: timm, PyTorch

    Classification of digits into nine classes (1-9) with CNN (mobilenet v2). Both printed and hand-written (with my hand :)) digits are recognized.

- **Sudoku Solver**: Python, numpy

    TBD
- **GUI**: streamlit

    Simple web-interface for demonstration

---

## Ongoing improvements
* Extend dataset with more hand-written digits
* Add 0 class for an emtpy cell, train new classificator. Check confidence values
* Update and commit training scripts
* Show not only the best next move, but also some tip message to prove the move
* Consider digits classification with classical CV approach (at least for printed digits), maybe stack some models to improve stability
* Add tests for invalid input examples
---

## 🎥 Demo
TBD

