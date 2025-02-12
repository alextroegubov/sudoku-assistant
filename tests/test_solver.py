import numpy as np
import pytest
from pathlib import Path

from src.solver import sudoku

# Path to the test cases folder
TEST_SOLVER_CASES_DIR = Path(__file__).parent / "test_solver_cases"
sudoku_txt_files = list(TEST_SOLVER_CASES_DIR.glob("*.txt"))


@pytest.mark.parametrize("filename", sudoku_txt_files)
def test_solve(filename):

    solver = sudoku.Sudoku()
    solver.read_from_txt(filename)
    solver.solve()

    assert solver.sudoku_is_solved(), f"Sudoku in {filename.name} failed to solve"


@pytest.mark.parametrize("filename", sudoku_txt_files)
def test_solve_one_step(filename):
    solver = sudoku.Sudoku()
    solver.read_from_txt(filename)

    sudoku_copy = solver.data.copy()

    num_steps = len(sudoku_copy[sudoku_copy == 0])

    for step in range(num_steps):
        solver.solve_one_step()
        assert (
            np.count_nonzero(sudoku_copy - solver.data) == 1
        ), f"Inserted more or less than one digit at step {step}"

        sudoku_copy = solver.data.copy()

    assert solver.sudoku_is_solved(), f"Sudoku in {filename.name} failed to solve"
