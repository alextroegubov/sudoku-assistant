from src.solver import sudoku
from src.exceptions import SudokuError


def test_sudoku_file(file):
    solver = sudoku.Sudoku()

    solver.read_from_txt(file)

    try:
        solver.solve()
    except SudokuError as e:
        print(f"Test with {file} failed:", e.message)
    else:
        print(f"Test with {file} passed")


def test_all_files():
    files = [
        "data/sudoku-middle.txt",
        "data/sudoku-hard.txt",
        "data/sudoku-expert.txt",
        "data/sudoku-master.txt",
    ]

    for file in files:
        test_sudoku_file(file)
