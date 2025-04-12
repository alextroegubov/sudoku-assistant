import cv2 as cv
import pytest
from pathlib import Path

from src.classifier import digits_classifier

TEST_CLASSIFIER_CASES_DIR = Path(__file__).parent / "test_classifier_cases"
digits_images = list(TEST_CLASSIFIER_CASES_DIR.glob("*digit*.jpg"))


@pytest.mark.parametrize("filename", digits_images)
def test_classifier(filename):

    image = cv.imread(filename, cv.IMREAD_GRAYSCALE)

    model = digits_classifier.DigitsClassifier(
        weights_file="",
        device="cpu",
    )

    confs, labels = model([image])

    conf = confs[0]
    label = labels[0] + 1
    true_label = int(str(filename.stem)[-3])

    assert conf > 0.75, f"Low confidence ({filename.name}): {conf}"
    assert true_label == label, f"Wrong prediction, {label}"
