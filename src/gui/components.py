""" String messages, constants, etc."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

UPLOAD_MESSAGE = "Upload sudoku image"

IMAGE_HELP_MESSAGE = (
    "Upload a clear, well-lit image of a Sudoku puzzle. "
    "Ensure the grid and digits are contrast, fully visible and not skewed."
)

IMAGE_TYPES = ["png", "jpg", "jpeg"]

BUTTON_SOLVE = "🧩 Solve"
SHOW_TIP = "💡 Show Tip"


GRID_COLOR = "black"
DIGITS_COLOR = "black"
BORDER_COLOR = "black"

UPLOAD_BORDER_COLOR = "blue"
INSERTED_DIGIT_COLOR = "green"
TIP_DIGIT_COLOR = "red"

DIGITS_TXT_PARAMS = {
    "ha": "center",
    "va": "center",
    "fontsize": 16,
    "fontweight": "bold",
}


def plot_sudoku_grid(
    sudoku_grid: np.ndarray,
    orig_sudoku_grid: np.ndarray,
    tip_idx: int,
    border: bool = False,
):
    """Plot a 9x9 Sudoku grid with optional border color."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw the grid with a colored border
    for i in range(10):  # 9 lines + border
        lw = 2 if i % 3 == 0 else 0.5  # Thicker lines for 3x3 blocks
        ax.axhline(i, color=GRID_COLOR, linewidth=lw)
        ax.axvline(i, color=GRID_COLOR, linewidth=lw)

    for idx in range(81):
        row, col = divmod(idx, 9)
        # plot orig_sudoku_grid
        num = orig_sudoku_grid[row, col]
        if num:
            ax.text(col + 0.5, 8.5 - row, str(num), color=DIGITS_COLOR, **DIGITS_TXT_PARAMS)
        # plot delta grid
        num = sudoku_grid[row, col] - orig_sudoku_grid[row, col]
        color = TIP_DIGIT_COLOR if idx == tip_idx else INSERTED_DIGIT_COLOR
        if num:
            ax.text(col + 0.5, 8.5 - row, str(num), color=color, **DIGITS_TXT_PARAMS)

    # Draw outer border manually with correct thickness
    border_color = UPLOAD_BORDER_COLOR if border else BORDER_COLOR
    outer_border = Rectangle((0, 0), 9, 9, linewidth=4, edgecolor=border_color, facecolor="none")
    ax.add_patch(outer_border)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_frame_on(False)

    return fig
