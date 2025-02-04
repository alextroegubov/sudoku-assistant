import torch
from torch import nn
import torchvision
from torchvision import transforms

import timm
import numpy as np

class DigitsClassifier:

    NORM_MEAN = 0.082
    NORM_STD = 0.257

    def __init__(self, model_name: str, weights_file: str, device: str):
        self.device = torch.device(device)
        self.model: nn.Module = timm.create_model(model_name, num_classes=9, in_chans=1)
        self.model.load_state_dict(torch.load(weights_file, weights_only=True))
        self.model.eval()
        self.model.to(self.device)

    def __call__(self, images: list[np.ndarray]):
        """"""
        images_batch = self.preprocess(images)
        conf, labels = self.apply_model(images_batch)

        return conf.detach().cpu().numpy(), labels.detach().cpu().numpy()

    def preprocess(self, images: list[np.ndarray]):
        # convert to tensor and add batch dim
        inference_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(),
            transforms.Resize((50, 50)),
            transforms.Normalize(mean=(self.NORM_MEAN), std=(self.NORM_STD)),
        ])
        tensors = [inference_transform(img).unsqueeze(0) for img in images]
        images_batch = torch.concat(tensors, dim=0).to(self.device)

        return images_batch

    def apply_model(self, images_batch: torch.Tensor):
        with torch.no_grad():
            logits = self.model(images_batch.to(self.device))
            probs = nn.functional.softmax(logits, dim=-1)
            confidence, labels_idx = torch.max(probs, dim=-1)

        return confidence, labels_idx