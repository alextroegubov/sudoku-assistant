import cv2 as cv
import streamlit as st
import numpy as np
import logging

from src.image_preprocess import field_extractor, digits_splitter
from src.classifier import digits_classifier
from src.utils import cls_output_to_grid
import src.gui.components as guic
from src.gui.components import plot_sudoku_grid
from src.solver import sudoku

from src.exceptions import ImagePreprocessingError, SudokuError

@st.cache_resource
def load_model():
    model = digits_classifier.DigitsClassifier(
        model_name="mobilenetv2_100.ra_in1k",
        weights_file="model/best_model.pt",
        device="cpu",
    )

    return model


def run_app():
    """Runs the Sudoku Assistant UI in Streamlit."""
    st.title("🧩 Sudoku Assistant")

    ext = field_extractor.FieldExtractor()
    splitter = digits_splitter.DigitsSplitter()
    classifier = load_model()
    solver = sudoku.Sudoku()

    # Initialize session state variables if not present
    if "grid_to_show" not in st.session_state:
        st.session_state.grid_to_show = None
        st.session_state.new_digits = []
        st.session_state.uploaded = False
        st.session_state.uploaded_file_id = None
        st.session_state.sudoku_confirmed = False
        st.session_state.sudoku_solved = False

    uploaded_file = st.file_uploader(
        guic.UPLOAD_MESSAGE, type=guic.IMAGE_TYPES, help=guic.IMAGE_HELP_MESSAGE
    )

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
            st.session_state.sudoku_confirmed = False
            st.session_state.sudoku_solved = False

        except ImagePreprocessingError as e:
            st.error(f"Error: {e.message}.\n Please upload a clearer image.")

    if st.session_state.uploaded:
        # Show extracted Sudoku with a **blue border** before confirmation
        fig = plot_sudoku_grid(
            st.session_state.grid_to_show, st.session_state.new_digits, not st.session_state.sudoku_confirmed
        )
        st.pyplot(fig)

        # Buttons for Confirm and Remove
        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ Confirm"):
                st.session_state.sudoku_confirmed = True
                st.rerun()

        with col2:
            if st.button("❌ Remove"):
                st.session_state.grid_to_show = None
                st.session_state.uploaded = False
                st.session_state.sudoku_confirmed = False
                st.rerun()

    # If confirmed, show solving options
    if st.session_state.sudoku_confirmed:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🧩 Solve"):
                solver.read_sudoku_from_numpy(st.session_state.grid_to_show)
                try:
                    solver.solve()
                except SudokuError:
                    st.error("Cannot solve sudoku")

                st.session_state.new_digits = [row * 9 + col for row in range(9) for col in range(9) if solver.data[row, col] != st.session_state.grid_to_show[row, col]]

                st.session_state.grid_to_show = solver.data
                st.session_state.sudoku_solved = True
                st.rerun()

        with col2:
            if st.button("💡 Show Tip") and not st.session_state.sudoku_solved:
                solver.read_sudoku_from_numpy(st.session_state.grid_to_show)
                try:
                    solver.solve_one_step()
                except SudokuError:
                    st.error("Cannot insert any digits")

                st.session_state.new_digits = [row * 9 + col for row in range(9) for col in range(9) if solver.data[row, col] != st.session_state.grid_to_show[row, col]]
                st.session_state.grid_to_show = solver.data
                st.rerun()
