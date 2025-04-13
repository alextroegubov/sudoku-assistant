import torch
from torch import nn
from torchvision import transforms
import numpy as np

from src.classifier.model import LightningClassifier


class DigitsClassifier:

    NORM_MEAN = 0.083
    NORM_STD = 0.257

    def __init__(self, weights_file: str, device: str):
        self.device = torch.device(device)
        self.model = LightningClassifier.load_from_checkpoint(weights_file)
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, images: list[np.ndarray]):
        images_batch = self.preprocess(images)
        conf, labels = self.apply_model(images_batch)

        return conf.detach().cpu().numpy(), labels.detach().cpu().numpy()

    def preprocess(self, images: list[np.ndarray]):
        # convert to tensor and add batch dim
        inference_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Resize((self.model.input_size, self.model.input_size)),
                transforms.Normalize(mean=(self.NORM_MEAN), std=(self.NORM_STD)),
            ]
        )
        tensors = [inference_transform(img).unsqueeze(0) for img in images]
        images_batch = torch.concat(tensors, dim=0).to(self.device)

        return images_batch

    def apply_model(self, images_batch: torch.Tensor):
        with torch.no_grad():
            logits = self.model(images_batch.to(self.device))
            probs = nn.functional.softmax(logits, dim=-1)
            confidence, labels_idx = torch.max(probs, dim=-1)

        return confidence, labels_idx
