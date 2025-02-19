import cv2 as cv
import streamlit as st
import numpy as np

from src.image_preprocess import field_extractor, digits_splitter
from src.classifier import digits_classifier
from src.utils import cls_output_to_grid
import src.gui.components as guic
from src.gui.components import plot_sudoku_grid
from src.solver import sudoku

from src.exceptions import ImagePreprocessingError, SudokuError


@st.cache_resource
def get_classifier():
    return digits_classifier.DigitsClassifier(
        model_name="mobilenetv2_100.ra_in1k",
        weights_file="model/best_model.pt",
        device="cpu",
    )


@st.cache_resource
def get_extractor():
    return field_extractor.FieldExtractor()


@st.cache_resource
def get_digits_splitter():
    return digits_splitter.SimpleSplitter()


@st.cache_resource
def get_solver():
    return sudoku.Sudoku()


def reset_sudoku_state():
    """Resets the Sudoku session state."""
    st.session_state.grid_to_show = None
    st.session_state.grid_orig = None
    st.session_state.tip_digit = -1
    st.session_state.uploaded = False
    st.session_state.sudoku_confirmed = False
    st.session_state.sudoku_solved = False
    st.rerun()


def initialize_session_state():
    """Initializes session state variables if not already set."""
    st.session_state.setdefault("grid_to_show", None)
    st.session_state.setdefault("grid_orig", None)
    st.session_state.setdefault("tip_digit", -1)
    st.session_state.setdefault("uploaded", False)
    st.session_state.setdefault("uploaded_file_id", None)
    st.session_state.setdefault("sudoku_confirmed", False)
    st.session_state.setdefault("sudoku_solved", False)


def run_app():
    """Runs the Sudoku Assistant UI in Streamlit."""
    st.title("🧩 Sudoku Assistant")

    initialize_session_state()
    handle_file_upload()

    if st.session_state.uploaded:
        display_sudoku_grid()

    if st.session_state.sudoku_confirmed:
        handle_sudoku_solving()


def handle_file_upload():
    """Handles image file upload and preprocessing."""
    uploaded_file = st.file_uploader(
        guic.UPLOAD_MESSAGE, type=guic.IMAGE_TYPES, help=guic.IMAGE_HELP_MESSAGE
    )

    if not uploaded_file or uploaded_file.file_id == st.session_state.uploaded_file_id:
        return

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    st.session_state.uploaded_file_id = uploaded_file.file_id

    extractor = get_extractor()
    splitter = get_digits_splitter()
    classifier = get_classifier()

    try:
        image_wo_grid = extractor(image)
        all_digits, not_none_digits = splitter(image_wo_grid, with_not_nones=True)
        confs, labels = classifier(not_none_digits)
        sudoku_grid = cls_output_to_grid(confs, labels, all_digits)
    except ImagePreprocessingError as e:
        st.error(f"Error: {e.message}. Please upload a clearer image.")
        return

    # Store results in session state
    st.session_state.grid_to_show = sudoku_grid
    st.session_state.grid_orig = sudoku_grid
    st.session_state.uploaded = True
    st.session_state.sudoku_confirmed = False
    st.session_state.sudoku_solved = False


def handle_sudoku_solving():

    solver = sudoku.Sudoku()
    col1, col2 = st.columns(2)

    with col1:
        if st.button(guic.BUTTON_SOLVE):
            solver.read_sudoku_from_numpy(st.session_state.grid_to_show)
            try:
                solver.solve()
            except SudokuError as e:
                st.error(f"Cannot solve sudoku: {e.message}")
            else:
                st.session_state.grid_to_show = solver.data
                st.session_state.sudoku_solved = True
                st.session_state.tip_digit = -1
                st.rerun()

    with col2:
        if st.button(guic.SHOW_TIP) and not st.session_state.sudoku_solved:
            solver.read_sudoku_from_numpy(st.session_state.grid_to_show)
            try:
                solver.solve_one_step()
            except SudokuError as e:
                st.error(f"Cannot insert any digits: {e.message}")
            else:
                row, col = np.where(solver.data != st.session_state.grid_to_show)
                st.session_state.tip_digit = row * 9 + col
                st.session_state.grid_to_show = solver.data
                st.rerun()


def display_sudoku_grid():
    """Displays the Sudoku grid and associated controls."""
    fig = plot_sudoku_grid(
        st.session_state.grid_to_show,
        st.session_state.grid_orig,
        st.session_state.tip_digit,
        not st.session_state.sudoku_confirmed,
    )
    st.pyplot(fig)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Confirm"):
            st.session_state.sudoku_confirmed = True
            st.rerun()

    with col2:
        if st.button("❌ Remove"):
            reset_sudoku_state()
