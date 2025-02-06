class ImagePreprocessingError(Exception):
    """Base class for exceptions in image processing."""

    pass


class NoContoursError(ImagePreprocessingError):
    """Raised when no contours are detected in the image."""

    def __init__(self, message="No contours detected in the image."):
        self.message = message
        super().__init__(self.message)


class NoRectContour(ImagePreprocessingError):
    """Raised when no rectangular contour is found."""

    def __init__(self, message="Failed to detect a rectangular Sudoku grid in the image."):
        self.message = message
        super().__init__(self.message)
