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
        weights_file="model/epoch=31-val_acc=1.0000.ckpt",
        device="cpu",
    )

    confs, preds = model([image])

    conf = confs[0]
    pred = preds[0] + 1
    gt = int(str(filename.stem)[-3])

    assert gt == pred, f"Wrong prediction ({filename.name}): {gt=} != {pred=}"
    assert conf > 0.4, f"Too low confidence ({filename.name}): {conf}"
