import numpy as np
import cv2 as cv
import pytest
from pathlib import Path

from src.image_preprocess import digits_splitter, field_extractor

# Path to the test cases folder
TEST_CASES_DIR = Path(__file__).parent / "test_image_preprocess_cases"
image_files = list(TEST_CASES_DIR.glob("*.*"))

answers = [
    # sudoku_2.jpeg
    [3, 4, 5, 9, 10, 12, 14, 16, 17, 27, 29, 33, 35, 37, 43, 45, 47, 51, 53, 63, 64, 66, 68, 70, 71, 75, 76, 77],
    # sudoku_3.jpeg
    [i for i in range(81)],
    # sudoku_4.jpeg,
    [0, 2, 4, 7, 8, 11, 13, 14, 16, 17, 18, 19, 20, 25, 28, 29, 30, 36, 46, 48, 49, 52, 56, 66, 71, 72, 73, 74, 75, 79]
]


@pytest.mark.parametrize("filename, correct_not_nones_idx", zip(image_files, answers))
def test_image_preprocess_file(filename, correct_not_nones_idx):

    splitter = digits_splitter.SimpleSplitter()
    extractor = field_extractor.FieldExtractor()

    img = cv.imread(filename)
    field = extractor(img)
    all_digits = splitter(field, False)

    not_nones_idx = sorted([i for i, d in enumerate(all_digits) if d is not None])
    correct_not_nones_idx = sorted(correct_not_nones_idx)

    unrec = sorted(list(set(correct_not_nones_idx) - set(not_nones_idx)))
    misrec = sorted(list(set(not_nones_idx) - set(correct_not_nones_idx)))

    assert not_nones_idx == correct_not_nones_idx, f"Unrecognized: {unrec}; Misrecognized: {misrec}"
