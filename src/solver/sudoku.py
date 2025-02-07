from enum import Enum
from itertools import combinations
import logging

import numpy as np

from src.exceptions import (
    SudokuError,
    InvalidInputError,
    InvalidDigitsError,
    InvalidFieldError,
    SolverError,
)

logger = logging.getLogger(__name__)
logging.basicConfig(filename="sudoku.log", filemode="w", level=logging.DEBUG, encoding="utf-8")


class Sudoku:
    """Sudoku solver"""

    all_digits = {1, 2, 3, 4, 5, 6, 7, 8, 9}

    class BlockType(Enum):
        """Sudoku block types"""

        ROW = 1
        COL = 2
        SQ = 3

    @staticmethod
    def row_col_to_square(row_n: int, col_n: int) -> int:
        """Convert (row, col) to square number"""
        return (row_n // 3) * 3 + (col_n // 3)

    @staticmethod
    def square_to_row_col(sq_n: int) -> tuple[int, int]:
        """Convert square number to (row, col) of the top left cell of that square"""
        row = (sq_n // 3) * 3
        col = (sq_n % 3) * 3
        return row, col

    @staticmethod
    def row_col_to_idx(row: int, col: int) -> int:
        """Convert (row, col) to idx"""
        return row * 9 + col

    @staticmethod
    def idx_to_row_col(idx: int) -> tuple[int, int]:
        """Convert idx to (row, col)"""
        return divmod(idx, 9)

    @staticmethod
    def idx_to_square(idx: int) -> int:
        """Convert idx to square number"""
        row, col = Sudoku.idx_to_row_col(idx)
        return Sudoku.row_col_to_square(row, col)

    def __init__(self):
        self.data: np.ndarray = np.zeros((9, 9), dtype=np.int32)
        self.possible: dict[int, set[int]] = {i: self.all_digits.copy() for i in range(81)}

    def __str__(self):
        """String representation of the Sudoku grid."""
        lines = []

        # Build the Sudoku grid
        for i, row in enumerate(self.data):
            line = []
            for j, num in enumerate(row):
                cell = str(num) if num else "."
                line.append(cell)
                if j % 3 == 2 and j != 8:
                    line.append("|")
            lines.append(" ".join(line))

            if i % 3 == 2 and i != 8:
                lines.append("-" * 21)

        # Show possible candidates for empty cells
        possible_str = "\nPossible Candidates:\n"
        for idx, candidates in sorted(self.possible.items()):
            if len(candidates) > 0:
                row, col = self.idx_to_row_col(idx)
                possible_str += f"({row}, {col}): {sorted(candidates)}\n"

        return "\n".join(lines) + ("\n" + possible_str if self.possible else "")

    def _get_digits_in_block(self, n: int, block_type: BlockType) -> np.ndarray:
        """Get row/column/square slice of sudoku"""
        if block_type is Sudoku.BlockType.ROW:
            return self.data[n, :]
        elif block_type is Sudoku.BlockType.COL:
            return self.data[:, n]
        else:  # square
            row, col = self.square_to_row_col(n)
            return self.data[row : row + 3, col : col + 3]

    def get_inserted_digits_in_block(self, n: int, block_type: BlockType) -> np.ndarray:
        """Get already inserted in the block digits"""
        all_digits = self._get_digits_in_block(n, block_type).flatten()
        return all_digits[all_digits > 0]

    def get_missed_digits_in_block(self, n: int, block_type: BlockType) -> set[int]:
        """Get missing in the block digits"""
        inserted_digits = self.get_inserted_digits_in_block(n, block_type)
        missed_digits = self.all_digits - set(inserted_digits)
        return missed_digits

    def get_empty_indexes_in_block(self, n: int, block_type: BlockType) -> tuple[int, ...]:
        """Get empty cell indexes in block"""
        if block_type is Sudoku.BlockType.ROW:
            rows = [n] * 9
            cols = list(range(9))

        elif block_type is Sudoku.BlockType.COL:
            rows = list(range(9))
            cols = [n] * 9

        else:
            row, col = Sudoku.square_to_row_col(n)
            rows = [row, row + 1, row + 2] * 3
            cols = [col] * 3 + [col + 1] * 3 + [col + 2] * 3

        indexes = tuple(
            self.row_col_to_idx(row, col)
            for (row, col) in zip(rows, cols)
            if self.data[row, col] == 0
        )
        return indexes

    def block_is_valid(self, n: int, block_type: BlockType) -> bool:
        """Check if the n-th block is valid: all non-zero digits are unique"""
        digits = self._get_digits_in_block(n, block_type).flatten()
        non_zero_count = np.count_nonzero(digits)
        unique_non_zero = np.unique(digits[digits > 0])

        if not set(unique_non_zero).issubset(self.all_digits):
            raise InvalidDigitsError(
                f"{block_type} #{n}: invalid digits: {set(unique_non_zero) - self.all_digits}"
            )

        if non_zero_count != len(unique_non_zero):
            raise InvalidFieldError(f"{block_type} #{n}: invalid block")

    def sudoku_is_valid(self):
        """Check if sudoku is valid: all block are valid"""
        _ = [self.block_is_valid(n, type_) for n in range(9) for type_ in Sudoku.BlockType]

    def block_is_solved(self, n: int, block_type: BlockType) -> bool:
        """Check if n-th sudoku block is solved"""
        empty_cells = self.get_empty_indexes_in_block(n, block_type)
        return len(empty_cells) == 0

    def sudoku_is_solved(self) -> bool:
        """Check if sudoku is solved: all blocks are solved"""
        return all(self.block_is_solved(n, type_) for n in range(9) for type_ in Sudoku.BlockType)

    def is_digit_possible_in_cell(self, cell_idx: int, digit: int) -> bool:
        """Check all cell blocks (row, col and squre) for digit"""
        row, col = Sudoku.idx_to_row_col(cell_idx)
        sq = Sudoku.row_col_to_square(row, col)

        in_row = digit in self.get_inserted_digits_in_block(row, Sudoku.BlockType.ROW)
        in_col = digit in self.get_inserted_digits_in_block(col, Sudoku.BlockType.COL)
        in_square = digit in self.get_inserted_digits_in_block(sq, Sudoku.BlockType.SQ)

        return not (in_row or in_col or in_square)

    def apply_rule_1(self):
        """Fill possible values based on simple checking of digits in rows, cols and squares.
        Applied per cell.
        """
        logger.debug("Start Rule #1")

        for idx in range(81):
            row, col = Sudoku.idx_to_row_col(idx)
            if self.data[row, col] == 0:

                new_possible = {
                    digit
                    for digit in self.possible[idx]
                    if self.is_digit_possible_in_cell(idx, digit)
                }
                if new_possible != self.possible[idx]:
                    self.possible[idx] = new_possible
                    logger.debug("\tUpdate (%s, %s): new candidates %s", row, col, new_possible)

    def apply_rule_2(self):
        """Remove possible values based on rule #2. Applied per block.

        Details:
         1. Find all missing digits in the block
         2. For group size = 1, 2, 3, 4: consider all possible groups of missing digits of this size
         3. If number of empty cells, where these digits are possible, equals group size,
            remove all other candidates from these cells
        """

        logger.debug("Start Rule #2")
        for block_num in range(9):
            for block_type in Sudoku.BlockType:
                self.apply_rule_2_one_block(block_num, block_type)

    def apply_rule_2_one_block(self, block_num: int, block_type: BlockType):
        """Remove possible values based on rule #2 in a single block."""

        missed_digits = self.get_missed_digits_in_block(block_num, block_type)
        missed_idx = self.get_empty_indexes_in_block(block_num, block_type)

        for group_size in [1, 2, 3, 4]:
            for group_digits in combinations(missed_digits, group_size):

                group_digits = set(group_digits)
                group_cells = [idx for idx in missed_idx if len(self.possible[idx] & group_digits)]

                if len(group_cells) == group_size:
                    # update cells in block
                    for idx in group_cells:
                        new = self.possible[idx] & group_digits
                        if new != self.possible[idx]:
                            row_, col_ = self.idx_to_row_col(idx)
                            logger.debug(
                                "\tUpdate (%s, %s): [%s #%s, group %s]: prev = %s, new = %s",
                                row_,
                                col_,
                                block_type,
                                block_num,
                                group_digits,
                                self.possible[idx],
                                new,
                            )
                            self.possible[idx] = new

    def apply_rule_3_one_block(self, block_num: int, bt: BlockType):
        """Remove possible values based on rule #3 in a single block."""
        missed_digits = self.get_missed_digits_in_block(block_num, bt)
        missed_indexes = self.get_empty_indexes_in_block(block_num, bt)

        for digit in missed_digits:
            indexes = [idx for idx in missed_indexes if digit in self.possible[idx]]

            if 2 <= len(indexes) <= 3:
                rows = np.array([self.idx_to_row_col(idx)[0] for idx in indexes])
                cols = np.array([self.idx_to_row_col(idx)[1] for idx in indexes])
                sqs = np.array([self.row_col_to_square(row, col) for row, col in zip(rows, cols)])

                indexes_where_to_remove = []
                if bt is Sudoku.BlockType.SQ and all(rows == rows[0]):
                    # all indexes in the same row, remove others in the same row
                    fixed_row = rows[0]
                    indexes_where_to_remove = [
                        self.row_col_to_idx(fixed_row, col) for col in range(9) if col not in cols
                    ]
                elif bt is Sudoku.BlockType.SQ and all(cols == cols[0]):
                    # all indexes in the same column, remove other in the same columns
                    fixed_col = cols[0]
                    indexes_where_to_remove = [
                        self.row_col_to_idx(row, fixed_col) for row in range(9) if row not in rows
                    ]

                elif (bt in [Sudoku.BlockType.ROW, Sudoku.BlockType.COL]) and all(sqs == sqs[0]):
                    fixed_sq = sqs[0]
                    empty = self.get_empty_indexes_in_block(fixed_sq, Sudoku.BlockType.SQ)
                    indexes_where_to_remove = [idx for idx in empty if idx not in indexes]

                for idx in indexes_where_to_remove:
                    if digit in self.possible[idx]:
                        self.possible[idx].discard(digit)
                        row_, col_ = self.idx_to_row_col(idx)
                        logger.debug(
                            "\tUpdate (%s, %s): [%s #%s]: remove digit %s - new = %s",
                            row_,
                            col_,
                            bt,
                            block_num,
                            digit,
                            self.possible[idx],
                        )

    def apply_rule_3(self):
        """Remove possible values based on rule #3. Applied per block.

        Details:
         1. Find all missing digits in the block
         2. For each missing digits, keep only digits which have two (pair) or three (trio)
         possible cells

        If a pair/trio is in a square and
            * in the same row - remove them from the rest of the row
            * in the same column - remove them from the rest of the column

        If a pair/trio is in a row and
            * in the same square - remove them from the rest of the square

        If a pair/trio is in a column and
            * in the same square - remove them from the rest of the square
        """

        logger.debug("Start Rule #3")
        for block_num in range(9):
            for block_type in Sudoku.BlockType:
                self.apply_rule_3_one_block(block_num, block_type)

    def insert_digit(self, idx: int, digit: int):
        row, col = Sudoku.idx_to_row_col(idx)
        self.data[row, col] = digit
        self.possible[idx].clear()

    def insert_possible_digits(self) -> int:
        count = 0
        for idx in range(81):
            row, col = Sudoku.idx_to_row_col(idx)
            if self.data[row, col] == 0 and len(self.possible[idx]) == 1:
                self.insert_digit(idx, self.possible[idx].pop())
                count += 1

        return count

    def solve(self):
        """Completely solve sudoku"""

        logger.info("Start solving Sudoku")

        max_iter = 100
        it = 0

        while it < max_iter and not self.sudoku_is_solved():
            it += 1

            logger.debug("Iteration: %s\n%s\n", it, str(self))

            count = 1
            # insert as many digits with simple rule as we can
            while count > 0:
                self.apply_rule_1()
                count = self.insert_possible_digits()
                logger.debug("Inserted %d digits with Rule #1", count)

            if self.sudoku_is_solved():
                break

            self.apply_rule_2()
            count = self.insert_possible_digits()
            logger.debug("Inserted %d digits with Rule #2", count)

            if self.sudoku_is_solved():
                break

            self.apply_rule_3()
            count = self.insert_possible_digits()
            logger.debug("Inserted %d digits with Rule #3", count)

        if self.sudoku_is_solved():
            logger.info("Sudoku solved in %s iterations", it)
        else:
            logger.info("Exceed iteration limit")
            raise ValueError

    def read_from_txt(self, input_file: str):
        """Read sudoku from txt"""
        with open(input_file, "r") as f:
            for row, line in enumerate(f):
                digits = [int(d) for d in line.strip().split(" ")]
                self.data[row] = np.array(digits, dtype=np.int32)

                for col in range(9):
                    if self.data[row, col]:
                        self.possible[self.row_col_to_idx(row, col)].clear()

    def read_sudoku_from_numpy(self, np_array: np.ndarray):
        shape = np_array.shape

        if shape == (9, 9):
            self.data = np_array.copy().astype(np.int32)
        elif shape == (81,):
            self.data = np_array.reshape(9, 9).astype(np.int32)
        else:
            raise InvalidInputError(f"Invalid numpy array shape: {np_array.shape}")

        try:
            self.sudoku_is_valid()
        except (InvalidDigitsError, InvalidFieldError) as e:
            raise InvalidInputError(e.message)


solver = Sudoku()

solver.read_from_txt("/home/user/Documents/data/sudoku-assistant/data/sudoku-middle.txt")

solver.solve()
