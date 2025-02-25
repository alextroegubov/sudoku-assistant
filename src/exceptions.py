"""Module with custom exceptions classes"""


class SudokuAssistantError(Exception):
    """Base class for all exceptions in the project."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class ImagePreprocessingError(SudokuAssistantError):
    """Base class for all image preprocessing exceptions."""

    def __init__(self, message: str = "Image Preprocessing Error"):
        super().__init__(message)


class FieldExtractionError(ImagePreprocessingError):
    """Raised on error in field extraction process"""


class InvalidImageError(ImagePreprocessingError):
    """Raised when the input image is invalid."""


class OpenCvError(ImagePreprocessingError):
    """Raised when an OpenCV error occurs."""


class SudokuError(SudokuAssistantError):
    """Base class for exception during sudoku solving"""


class InvalidInputError(SudokuError):
    """"""

    def __init__(self, message="Invalid shape of numpy array"):
        self.message = message
        super().__init__(self.message)


class InvalidDigitsError(SudokuError):
    """"""

    def __init__(self, message="Invalid digits in sudoku field!"):
        self.message = message
        super().__init__(self.message)


class InvalidFieldError(SudokuError):
    """"""

    def __init__(self, message="Invalid sudoku grid!"):
        self.message = message
        super().__init__(self.message)


class SolverError(SudokuError):
    """"""

    def __init__(self, message="Cannot solve puzzle"):
        self.message = message
        super().__init__(self.message)


class ClassificationError(Exception):
    """Base class for exceptions in image processing."""

    pass
