from torchvision.datasets import FER2013
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch.optim as optim
import torch
import torch.nn as nn
import pandas as pd
import json
from tqdm import tqdm
from ml26.proyectos.P01_facial_expressions.dataset import get_loader
from ml26.proyectos.P01_facial_expressions.network import Network

# Logging
import wandb
from datetime import datetime, timezone


def init_wandb(cfg):
    # Initialize wandb
    now_utc = datetime.now(timezone.utc)
    timestamp = now_utc.strftime("%Y-%m-%d_%H-%M-%S-%f")

    run = wandb.init(
        project="facial_expressions_cnn",
        config=cfg,
        name=f"facial_expressions_cnn_{timestamp}_utc",
    )
    return run


def validation_step(val_loader, net, cost_function):
    val_loss = 0.0
    correct = 0
    total = 0

    net.eval()

    for i, batch in enumerate(val_loader, 0):
        batch_imgs = batch["transformed"]
        batch_labels = batch["label"]
        device = net.device

        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)

        with torch.inference_mode():
            logits, proba = net(batch_imgs)
            loss = cost_function(logits, batch_labels)
            val_loss += loss.item()

            preds = torch.argmax(proba, dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)

    val_loss = val_loss / len(val_loader)
    val_acc = correct / total

    return val_loss, val_acc
 

def train():
    # Hiperparametros
    cfg = {
        "training": {
            "learning_rate": 1e-4,
            "n_epochs": 120,
            "batch_size": 128,
        },
    }
    run = init_wandb(cfg)

    train_cfg = cfg.get("training")
    learning_rate = train_cfg.get("learning_rate")
    n_epochs = train_cfg.get("n_epochs")
    batch_size = train_cfg.get("batch_size")

    # Train, validation, test loaders
    train_dataset, train_loader = get_loader(
        "train", batch_size=batch_size, shuffle=True
    )
    val_dataset, val_loader = get_loader("val", batch_size=batch_size, shuffle=False)
    print(
        f"Cargando datasets --> entrenamiento: {len(train_dataset)}, validacion: {len(val_dataset)}"
    )

    # Instancias red
    modelo = Network(input_dim=48, n_classes=7)

    # Calcula pesos por clase para compensar el desbalance del dataset
    _df = pd.read_csv(train_dataset.root / "data" / "train.csv")

    with open(train_dataset.root / "data" / "split.json") as f:
        _split_ids = json.load(f)["train"]

    _df = _df.iloc[_split_ids]

    _counts = _df["emotion"].value_counts().sort_index().values
    _weights = 1.0 / np.sqrt(_counts)
    _weights = _weights / _weights.sum() * len(_counts)

    class_weights = torch.tensor(_weights, dtype=torch.float).to(modelo.device)

    # Mide el error del modelo dando más peso a clases menos representadas
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    #optimizador, con weight decay pesos grandes son penalizados reducir overfitting ver si jala
    optimizer = optim.Adam(modelo.parameters(), lr=learning_rate, weight_decay=1e-4)

    #bajar learning rate auto si valloss no mejora
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )

    best_epoch_loss = np.inf
    patience = 20
    min_delta = 0.001
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        modelo.train()
        train_loss = 0
        correct = 0
        total = 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch: {epoch}")):
            batch_imgs = batch["transformed"]
            batch_labels = batch["label"]
            #datos a gpu+limpia grads uncs+pred+calc err+calc grads+upd weight
            batch_imgs = batch_imgs.to(modelo.device)
            batch_labels = batch_labels.to(modelo.device)

            optimizer.zero_grad()

            logits, proba = modelo(batch_imgs)
            loss = criterion(logits, batch_labels)

            loss.backward()
            optimizer.step()

            #suma error cada batch
            train_loss += loss.item()
            preds = torch.argmax(proba, dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)

        #promedio loss training epoch tocado y accuracy asi bien fresh
        train_loss = train_loss / len(train_loader)
        train_acc = correct / total

        val_loss, val_acc = validation_step(val_loader, modelo, criterion)
        scheduler.step(val_loss)

        tqdm.write(
            f"Epoch: {epoch}, train_loss: {train_loss:.2f}, val_loss: {val_loss:.2f}, "
            f"train_acc: {train_acc:.2f}, val_acc: {val_acc:.2f}"
        )

        # Guarda el modelo cuando val_loss mejora; si no mejora en 20 epochs, detiene el entrenamiento
        if val_loss < best_epoch_loss - min_delta:
            best_epoch_loss = val_loss
            epochs_without_improvement = 0
            modelo.save_model("modelo_1.pt")
        else:
            epochs_without_improvement += 1
       
        run.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "train/accuracy": train_acc,
                "val/accuracy": val_acc,
                "early_stop/no_improvement": epochs_without_improvement,
            }
        )

        if epochs_without_improvement >= patience:
            tqdm.write(f"Early stopping en epoch {epoch}. Mejor val_loss: {best_epoch_loss:.4f}")
            break

    run.finish()

if __name__ == "__main__":
    train()
