from pathlib import Path

from src.classifier.train import train

if __name__ == "__main__":
    train(Path("configs/params.yaml"))
