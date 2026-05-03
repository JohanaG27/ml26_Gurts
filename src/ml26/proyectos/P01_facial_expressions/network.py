import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pathlib
from torchvision.models import resnet18, ResNet18_Weights

file_path = pathlib.Path(__file__).parent.absolute()


def build_backbone(model="resnet18", weights="imagenet", freeze=True, last_n_layers=2):
    if model == "resnet18":
        backbone = resnet18(pretrained=weights == "imagenet")
        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False
        return backbone
    else:
        raise Exception(f"Model {model} not supported")


class Network(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # CNN para imágenes grayscale de 48x48
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 48 -> 24 -> 12 -> 6 después de 3 MaxPool
        out_dim = 128 * 6 * 6

        self.fc1 = nn.Linear(out_dim, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, n_classes)

        self.to(self.device)

    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        out_dim = math.floor((in_dim - kernel_size + 2 * padding) / stride) + 1
        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        logits = self.fc2(x)
        proba = F.softmax(logits, dim=1)

        return logits, proba

    def predict(self, x):
        self.eval()
        with torch.inference_mode():
            if x.dim() == 3:
                x = x.unsqueeze(0)
            return self.forward(x)

    def save_model(self, model_name: str):
        """
        Guarda los pesos del modelo.
        """
        models_path = file_path / "models" / model_name
        if not models_path.parent.exists():
            models_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), models_path)

    def load_model(self, model_name: str):
        """
        Carga los pesos del modelo.
        """
        models_path = file_path / "models" / model_name
        self.load_state_dict(torch.load(models_path, map_location=self.device))
        self.eval()