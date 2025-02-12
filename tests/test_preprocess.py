import numpy as np
import cv2 as cv

from src.image_preprocess import digits_splitter, field_extractor
from src.exceptions import SudokuError


def test_image_preprocess_file(file, correct_not_nones_idx):

    splitter = digits_splitter.SimpleSplitter()
    extractor = field_extractor.FieldExtractor()

    try:
        img = cv.imread(file)
        field = extractor(img)
        all_digits = splitter(field, False)

    except SudokuError as e:
        print(f"Test <test_image_preprocess> with {file} FAILED:", e.message)
    else:
        not_nones_idx = [i for i, d in enumerate(all_digits) if d is not None]
        if not_nones_idx != correct_not_nones_idx:

            unrec = sorted(list(set(correct_not_nones_idx) - set(not_nones_idx)))
            missrec = sorted(list(set(not_nones_idx) - set(correct_not_nones_idx)))

            print(f"Test <test_image_preprocess> with {file} FAILED:")
            print(f"\tUnrecognized cells: {unrec}")
            print(f"\tShould be empty: {missrec}")
        else:
            print(f"Test <test_image_preprocess> with {file} passed")


def test_image_preprocess():
    files = [
        "data/sudoku_2.jpeg",
        "data/sudoku_3.jpeg",
        "data/sudoku_4.jpg",
    ]

    correct = [
        # sudoku_2.jpeg
        [3, 4, 5, 9, 10, 12, 14, 16, 17, 27, 29, 33, 35, 37, 43, 45, 47, 51, 53, 63, 64, 66, 68, 70, 71, 75, 76, 77],
        # sudoku_3.jpeg
        [i for i in range(81)],
        # sudoku_4.jpeg,
        [0, 2, 4, 7, 8, 11, 13, 14, 16, 17, 18, 19, 20, 25, 28, 29, 30, 36, 46, 48, 49, 52, 56, 66, 71, 72, 73, 74, 75, 79]
    ]
    for file, answers in zip(files, correct):
        test_image_preprocess_file(file, answers)
