import numpy as np
from src.solver import sudoku
from src.exceptions import SudokuError


def test_solve(file):
    try:
        solver = sudoku.Sudoku()
        solver.read_from_txt(file)
        solver.solve()
    except SudokuError as e:
        print(f"Test <test_solve> with {file} failed:", e.message)
    else:
        print(f"Test <test_solve> with {file} passed")


def test_solve_one_step(file):
    try:
        solver = sudoku.Sudoku()
        solver.read_from_txt(file)

        sudoku_copy = solver.data.copy()

        while not solver.sudoku_is_solved():
            solver.solve_one_step()
            # exactrly one new digit
            if np.count_nonzero(sudoku_copy - solver.data) != 1:
                print(f"Test <test_solve_one_step> with {file} failed: more than 1 digits changed")
                return

            sudoku_copy = solver.data.copy()

    except SudokuError as e:
        print(f"Test <test_solve_one_step> with {file} failed:", e.message)
    else:
        print(f"Test <test_solve_one_step> with {file} passed")


def test_solver():
    files = [
        "data/sudoku-middle.txt",
        "data/sudoku-hard.txt",
        "data/sudoku-expert.txt",
        "data/sudoku-master.txt",
    ]

    for file in files:
        test_solve(file)
        test_solve_one_step(file)
