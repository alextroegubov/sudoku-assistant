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

GRID_COLOR = "black"
DIGITS_COLOR = "black"
BORDER_COLOR = "black"

UPLOAD_BORDER_COLOR = "blue"
TIP_DIGITS_COLOR = "green"


def plot_sudoku_grid(
    sudoku_array: np.ndarray,
    tip_indexes: list[int],
    border: bool = False,
):
    """Plot a 9x9 Sudoku grid with optional border color."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw the grid with a colored border
    for i in range(10):  # 9 lines + border
        lw = 2 if i % 3 == 0 else 0.5  # Thicker lines for 3x3 blocks
        ax.axhline(i, color=GRID_COLOR, linewidth=lw)
        ax.axvline(i, color=GRID_COLOR, linewidth=lw)

    # Fill in numbers
    for idx in range(81):
        row, col = divmod(idx, 9)
        num = sudoku_array[row, col]
        color = TIP_DIGITS_COLOR if idx in tip_indexes else DIGITS_COLOR
        if num:  # Skip empty cells (0)
            ax.text(
                col + 0.5,
                8.5 - row,
                str(num),
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color=color,
            )

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
