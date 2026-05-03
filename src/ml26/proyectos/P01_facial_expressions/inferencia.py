import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch

from ml26.proyectos.P01_facial_expressions.network import Network
from ml26.proyectos.P01_facial_expressions.utils import (
    to_numpy,
    get_transforms,
    add_img_text,
)
from ml26.proyectos.P01_facial_expressions.dataset import EMOTIONS_MAP
import pathlib

file_path = pathlib.Path(__file__).parent.absolute()


def load_img(path):
    assert os.path.isfile(path), f"El archivo {path} no existe"

    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    val_transforms, unnormalize = get_transforms("test", img_size=48)

    tensor_img = val_transforms(img_rgb)
    denormalized = unnormalize(tensor_img.clone())

    return img, tensor_img, denormalized


def predict(img_title_paths):
    """
    Hace la inferencia de las imagenes
    args:
    - img_title_paths (dict): diccionario con el titulo de la imagen (key) y el path (value)
    """
    # Cargar el modelo
    modelo = Network(48, 7)
    modelo.load_model("modelo_1.pt")
    for path in img_title_paths:
        # Cargar la imagen
        # np.ndarray, torch.Tensor
        im_file = (file_path / path).as_posix()
        original, transformed, denormalized = load_img(im_file)

        # Inferencia
        logits, proba = modelo.predict(transformed)
        pred = torch.argmax(proba, -1).item()
        pred_label = EMOTIONS_MAP[pred]
        confidence = proba[0][pred].item()

        # Original / transformada
        h, w = original.shape[:2]
        resize_value = 300
        img = cv2.resize(original, (w * resize_value // h, resize_value))
        img = add_img_text(img, f"Pred: {pred_label} ({confidence:.2f})")

        # Mostrar la imagen
        denormalized = to_numpy(denormalized)
        denormalized = cv2.resize(denormalized, (resize_value, resize_value))
        cv2.imshow("Predicción - original", img)
        cv2.imshow("Predicción - transformed", denormalized)
        cv2.waitKey(0)

    cv2.destroyAllWindows()



if __name__ == "__main__":
    # Direcciones relativas a este archivo
    img_paths = ["./test_imgs/happy.png"]
    predict(img_paths)
