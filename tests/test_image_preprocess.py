import numpy as np
import cv2 as cv
import pytest
from pathlib import Path

from src.image_preprocess import digits_splitter, field_extractor

# Path to the test cases folder
TEST_CASES_DIR = Path(__file__).parent / "test_image_preprocess_cases"

image_files = [
    TEST_CASES_DIR / "sudoku_book_1.jpg",
    TEST_CASES_DIR / "sudoku_book_2.jpg",
    TEST_CASES_DIR / "sudoku_book_3.jpg",
    TEST_CASES_DIR / "sudoku_book_4.jpeg",
    TEST_CASES_DIR / "sudoku_book_5.jpeg",
    TEST_CASES_DIR / "sudoku_com_expert.png",
    TEST_CASES_DIR / "sudoku_com_extreme.png",
    TEST_CASES_DIR / "sudoku_com_master.png",
    TEST_CASES_DIR / "sudoku_com_hard.png",
]

answers = [
    # sudoku_book_1.jpg
    list(range(81)),
    # sudoku_book_2.jpg
    list(range(81)),
    # sudoku_book_3.jpg
    list(range(81)),
    # sudoku_book_4.jpeg,
    [3, 4, 5, 9, 10, 12, 14, 16, 17, 27, 29, 33, 35, 37, 43, 45, 47, 51, 53, 63, 64, 66, 68, 70, 71, 75, 76, 77],
    # sudoku_book_5.jpeg
    list(range(81)),
    # sudoku_com_expert.png
    [1, 7, 8, 10, 13, 19, 24, 25, 28, 30, 34, 35, 36, 39, 40, 43, 45, 48, 53, 54, 56, 58, 61, 62, 63, 67, 71, 72, 76, 77],
    # sudoku_com_extreme.png
    [0, 4, 5, 6, 11, 14, 25, 30, 34, 35, 36, 37, 47, 52, 53, 55, 58, 59, 65, 68, 70, 75, 78],
    # sudoku_com_master.png
    [3, 4, 6, 9, 11, 19, 21, 27, 31, 41, 43, 46, 49, 52, 54, 59, 62, 63, 68, 69, 70, 72, 74, 78, 80],
    # sudoku_com_hard.png
    [0, 1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 17, 20, 24, 26, 27, 29, 30, 32, 40, 43, 45, 46, 53, 56, 62, 64, 66, 68, 70, 75, 76, 79, 80],
]


@pytest.mark.parametrize("filename, correct_not_nones_idx", zip(image_files, answers))
def test_image_preprocess_file(filename, correct_not_nones_idx):

    splitter = digits_splitter.SimpleSplitter()
    extractor = field_extractor.FieldExtractor()

    img = cv.imread(filename)
    field = extractor(img)
    all_digits = splitter(field, False)

    not_nones_idx = sorted([i for i, d in enumerate(all_digits) if d is not None])

    print(not_nones_idx)

    correct_not_nones_idx = sorted(correct_not_nones_idx)

    unrec = sorted(list(set(correct_not_nones_idx) - set(not_nones_idx)))
    misrec = sorted(list(set(not_nones_idx) - set(correct_not_nones_idx)))

    assert not_nones_idx == correct_not_nones_idx, f"Unrecognized: {unrec}; Misrecognized: {misrec}"
