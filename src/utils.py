import numpy as np


def cls_output_to_grid(confs, labels, all_digits):
    """"""
    sudoku_grid = np.zeros(81, dtype=int)
    threshold = 0.8

    label_idx = 0
    for i, digit in enumerate(all_digits):
        if digit is not None:
            if confs[label_idx] > threshold:
                sudoku_grid[i] = labels[label_idx] + 1
            label_idx += 1

    return sudoku_grid.reshape((9, 9))
