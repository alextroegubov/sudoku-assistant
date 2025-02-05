import cv2 as cv
import streamlit as st


from src.image_preprocess import field_extractor, digits_splitter
from src.classifier import digits_classifier
from src.utils import cls_output_to_grid

import matplotlib.pyplot as plt
import numpy as np


def plot_sudoku_grid(sudoku_array: np.ndarray, border_color="black"):
    """Plot a 9×9 Sudoku grid with optional border color."""
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw the grid with a colored border
    for i in range(10):  # 9 lines + border
        lw = 2 if i % 3 == 0 else 0.5  # Thicker lines for 3x3 blocks
        ax.axhline(i, color="black", linewidth=lw)
        ax.axvline(i, color="black", linewidth=lw)

    # Fill in numbers
    for row in range(9):
        for col in range(9):
            num = sudoku_array[row, col]
            if num != 0:  # Skip empty cells (0)
                ax.text(
                    col + 0.5,
                    8.5 - row,
                    str(num),
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                )

    # Draw outer border manually with correct thickness
    outer_border = plt.Rectangle(
        (0, 0), 9, 9, linewidth=4, edgecolor=border_color, facecolor="none"
    )
    ax.add_patch(outer_border)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_frame_on(False)

    return fig


@st.cache_resource
def load_model():
    model = digits_classifier.DigitsClassifier(
        model_name="mobilenetv2_100.ra_in1k",
        weights_file="model/best_model.pt",
        device="cpu",
    )

    return model


UPLOAD_MESSAGE = "Upload sudoku image"

IMAGE_HELP_MESSAGE = (
    "Upload a clear, well-lit image of a Sudoku puzzle. "
    "Ensure the grid and digits are contrast, fully visible and not skewed."
)

IMAGE_TYPES = ["png", "jpg", "jpeg"]


def run_app():
    """Runs the Sudoku Assistant UI in Streamlit."""
    st.title("🧩 Sudoku Assistant")

    ext = field_extractor.FieldExtractor()
    splitter = digits_splitter.DigitsSplitter()
    classifier = load_model()

    # Initialize session state variables if not present
    if "grid_to_show" not in st.session_state:
        st.session_state.grid_to_show = None
        st.session_state.uploaded = False
        st.session_state.uploaded_file_id = None
        st.session_state.sudoku_confirmed = False
        st.session_state.sudoku_solved = False

    uploaded_file = st.file_uploader(UPLOAD_MESSAGE, type=IMAGE_TYPES, help=IMAGE_HELP_MESSAGE)

    if uploaded_file and uploaded_file.file_id != st.session_state.uploaded_file_id:
        # Convert image to numpy array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

        st.session_state.uploaded_file_id = uploaded_file.file_id

        try:
            # Extract grid and recognize digits
            image_wo_grid = ext(image)
            all_digits, not_none_digits = splitter(image_wo_grid, with_not_nones=True)

            confs, labels = classifier(not_none_digits)
            sudoku_grid = cls_output_to_grid(confs, labels, all_digits)

            st.session_state.grid_to_show = sudoku_grid
            st.session_state.uploaded = True
            st.session_state.confirmed = False
            st.session_state.sudoku_solved = False

        except ValueError:
            st.error("Error: Please upload a clearer image.")

    if st.session_state.uploaded:
        # Show extracted Sudoku with a **blue border** before confirmation
        border_color = "black" if st.session_state.confirmed else "blue"
        fig = plot_sudoku_grid(st.session_state.grid_to_show, border_color)
        st.pyplot(fig)

        # Buttons for Confirm and Remove
        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Confirm"):
                st.session_state.confirmed = True
                st.rerun()

        with col2:
            if st.button("❌ Remove"):
                st.session_state.grid_to_show = None
                st.session_state.uploaded = False
                st.session_state.confirmed = False
                st.rerun()

    # If confirmed, show solving options
    if st.session_state.confirmed:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🧩 Solve"):
                st.session_state.grid_to_show = np.ones((9, 9), dtype=int) * 8
                st.session_state.sudoku_solved = True
                st.rerun()

        with col2:
            if st.button("💡 Show Tip") and not st.session_state.sudoku_solved:
                st.session_state.grid_to_show = np.ones((9, 9), dtype=int)
                st.rerun()
