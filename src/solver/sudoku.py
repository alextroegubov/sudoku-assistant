from enum import Enum
from copy import copy
from itertools import combinations
import logging

import numpy as np

# class MemoryStamp:
#     data: np.ndarray
#     possible: "dict[int, set[int]]"
#     idx: int
#     digit: int

#     def __init__(self, data: np.ndarray, possible: "dict[int, set[int]]", idx: int, digit: int):
#         self.data = np.copy(data)
#         self.possible = deepcopy(possible)
#         self.idx = idx
#         self.digit = digit

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
        self.data: np.ndarray = np.zeros((9, 9), dtype=int)
        self.possible = {i: copy(self.all_digits) for i in range(81)}

        logger.info("Sudoku\n %s", str(self))
        # self.sudoku_stamps = []

    def __str__(self):
        """String representation of the Sudoku grid."""
        lines = []

        # Build the Sudoku grid
        for i, row in enumerate(self.data):
            line = " ".join(str(num) if num != 0 else "." for num in row)
            lines.append(line)
            if i % 3 == 2 and i != 8:  # Add horizontal separator every 3 rows (except last)
                lines.append("-" * 21)

        # Show possible candidates for empty cells
        possible_str = "\nPossible Candidates:\n"
        for idx, candidates in sorted(self.possible.items()):
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

    def get_empty_indexes_in_block(self, n: int, block_type: BlockType) -> tuple[int]:
        """Get empty cell indexes in block"""
        if block_type is Sudoku.BlockType.ROW:
            rows = [n] * 9
            cols = list(range(9))
            # indexes = tuple(self.row_col_to_idx(row=n, col=col) for col in range(9))

        elif block_type is Sudoku.BlockType.COL:
            rows = list(range(9))
            cols = [n] * 9
            # indexes = tuple(self.row_col_to_idx(row=row, col=n) for row in range(9))

        else:
            row, col = Sudoku.square_to_row_col(n)
            rows = [row, row + 1, row + 2] * 3
            cols = [col] * 3 + [col + 1] * 3 + [col + 2] * 3
            # indexes = tuple(
            #     self.row_col_to_idx(row=r, col=c)
            #     for r in range(row, row + 3)
            #     for c in range(col, col + 3)
            # )

        indexes = tuple(self.row_col_to_idx(row, col) for (row, col) in zip(rows, cols) if self.data[row, col] == 0)

        # rows_cols = (self.idx_to_row_col(idx) for idx in indexes)
        # indexes = tuple(idx for idx in indexes if self.data[idx] == 0)
        return indexes

    def block_is_valid(self, n: int, block_type: BlockType) -> bool:
        """Check if the n-th block is valid: all non-zero digits are unique"""
        digits = self._get_digits_in_block(n, block_type).flatten()
        non_zero_count = np.count_nonzero(digits)
        unique_non_zero_count = len(np.unique(digits[digits > 0]))

        return non_zero_count == unique_non_zero_count

    def sudoku_is_valid(self) -> bool:
        """Check if sudoku is valid: all block are valid"""
        return all(self.block_is_valid(n, type_) for n in range(9) for type_ in Sudoku.BlockType)

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
        """Fill possible values based on simple checking of rows, cols and squares.
        Applied per cell.
        """
        for idx in range(81):
            row, col = Sudoku.idx_to_row_col(idx)
            if self.data[row, col] == 0:

                new_possible = {
                    digit
                    for digit in self.possible[idx]
                    if self.is_digit_possible_in_cell(idx, digit)
                }

                self.possible[idx] = new_possible

    def apply_rule_2(self):
        """Remove possible values based on rule #2. Applied per block."""
        for block_num in range(9):
            for block_type in Sudoku.BlockType:
                self.apply_rule_2_one_block(block_num, block_type)

    def apply_rule_2_one_block(self, block_num: int, block_type: BlockType):
        """Remove possible values based on rule #2 in a single block."""
        missed_digits = self.get_missed_digits_in_block(block_num, block_type)
        missed_indexes = self.get_empty_indexes_in_block(block_num, block_type)

        for group_size in [2, 3, 4]:
            for group in combinations(missed_digits, group_size):
                group_set = set(group)

                cells_with_digits_from_group = [
                    idx for idx in missed_indexes if len(self.possible[idx].intersection(group_set))
                ]

                if len(cells_with_digits_from_group) == group_size:
                    # remove these digits from other cells:
                    for idx in missed_indexes:
                        if idx not in cells_with_digits_from_group:
                            self.possible[idx] = self.possible[idx] - group_set

    def apply_rule_3_one_block(self, block_num: int, bt: BlockType):
        missed_digits = self.get_missed_digits_in_block(block_num, bt)
        missed_indexes = self.get_empty_indexes_in_block(block_num, bt)

        for digit in missed_digits:
            indexes = [idx for idx in missed_indexes if digit in self.possible[idx]]

            if len(indexes) == 2 or len(indexes) == 3:
                rows = [self.idx_to_row_col(idx)[0] for idx in indexes]
                cols = [self.idx_to_row_col(idx)[1] for idx in indexes]
                sqs = [self.row_col_to_square(row, col) for row, col in zip(rows, cols)]

                print((sqs == sqs[0]))

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
                    self.possible[idx].discard(digit)

    def apply_rule_3(self):
        """Remove possible values based on rule #3 (applied per block)"""
        for block_num in range(9):
            for block_type in Sudoku.BlockType:
                self.apply_rule_3_one_block(block_num, block_type)

    def insert_digit(self, idx, digit):
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
        it = 0
        while not self.sudoku_is_solved():
            print(f"it = {it}")

            count = 1
            # insert as many digits with rule 1 as we can
            print("Moved to rule 1")
            while count > 0:
                self.apply_rule_1()
                count = self.insert_possible_digits()
                print(f"Inserted {count} digits with rule 1")

            print("Moved to rule 2")
            self.apply_rule_2()
            count = self.insert_possible_digits()
            if count > 0:
                print(f"Inserted {count} digits with rule 2")
                it += 1
                continue

            print("Moved to rule 3")
            self.apply_rule_3()
            count = self.insert_possible_digits()
            if count > 0:
                print(f"Inserted {count} digits with rule 3")
                it += 1
                continue

            it += 1

        self.Print()

        print("Sudoku is solved")

    def Read(self, input_file):
        with open(input_file, "r") as f:
            for i, line in enumerate(f):
                self.data[i] = np.array([int(digit) for digit in line.lstrip().rstrip().split(" ")])

        for i in range(9):
            for j in range(9):
                if self.data[i, j] != 0:
                    self.possible[i * 9 + j] = set()

    def Print(self):
        for row in range(9):
            for col in range(9):
                print(f"{self.data[row, col]} ", end="")
            print("\n")


sudoku = Sudoku()

sudoku.Read("input-middle.txt")
print(str(sudoku))
sudoku.solve()
print(str(sudoku))


