import cv2 as cv

from src.image_preprocess import field_extractor, digits_splitter


import matplotlib.pyplot as plt
import numpy as np


def viz(img, title=""):
    plt.figure()

    if len(img.shape) == 3:
        plt.imshow(img[:, :, ::-1])
    else:
        plt.imshow(img, cmap="gray")

    plt.title(title)
    plt.show()


def plot_sudoku_grid(sudoku_array):
    """Plot a 9×9 Sudoku grid with given numbers."""

    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw the grid
    for i in range(10):  # 9 lines + border
        lw = 2 if i % 3 == 0 else 0.5  # Thicker lines for 3x3 blocks
        ax.axhline(i, color="black", linewidth=lw)
        ax.axvline(i, color="black", linewidth=lw)

    # Fill in numbers
    for row in range(9):
        for col in range(9):
            num = sudoku_array[row, col]
            if num != 0:  # Skip empty cells (0)
                ax.text(
                    col + 0.5,
                    8.5 - row,
                    str(num),
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_frame_on(False)

    plt.show()


sudoku_example = np.array(
    [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ]
)

if __name__ == "__main__":
    files = [
        "/home/user/Documents/data/sudoku-assistant/data/sudoku_1.jpeg",
        "/home/user/Documents/data/sudoku-assistant/data/sudoku_2.jpeg",
        "/home/user/Documents/data/sudoku-assistant/data/sudoku_3.jpeg",
    ]

    ext = field_extractor.FieldExtractor()
    splitter = digits_splitter.DigitsSplitter()

    for file in files:
        img = cv.imread(file)
        viz(img, "original")
        ext_imt = ext(img)

        digits = splitter(ext_imt)
        print(len(digits))

        viz(ext_imt, "processed")

        plot_sudoku_grid(sudoku_example)