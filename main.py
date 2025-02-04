import cv2 as cv

from src.image_preprocess import field_extractor, digits_splitter
from src.classifier import digits_classifier

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
        "data/sudoku_3.jpeg",
        # "data/sudoku_2.jpeg",
        # "data/sudoku_3.jpeg",
    ]

    ext = field_extractor.FieldExtractor()
    splitter = digits_splitter.DigitsSplitter()
    classifier = digits_classifier.DigitsClassifier(
        model_name="mobilenetv2_100.ra_in1k",
        weights_file="model/best_model.pt",
        device='cuda',
    )

    for file in files:
        img = cv.imread(file)
        viz(img, "original")
        ext_imt = ext(img)

        viz(ext_imt, "processed")

        digits = splitter(ext_imt)

        not_none_digits = [d for d in digits if d is not None]

        print(len(not_none_digits))

        # for d in not_none_digits:
        #     viz(d)

        confs, labels = classifier(not_none_digits)

        label_index = 0
        sudoku_array = []
        for digit in digits:
            if digit is None:
                sudoku_array.append(0)
            else:
                sudoku_array.append(labels[label_index] + 1)
                label_index += 1

        # plot_sudoku_grid(sudoku_example)
        plot_sudoku_grid(np.array(sudoku_array).reshape(9, 9))