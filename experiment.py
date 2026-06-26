
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import confusion_matrix, classification_report

from model import build_convnext_tiny

LABEL_MAPPING = {
    0: "well-being",
    1: "impaired",
}

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def build_transforms(image_size=224):
    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_tfms, val_tfms

def build_dataloaders(
    train_df,
    val_df,
    dataset_class,
    batch_size=32,
    num_workers=0,
    image_size=224
):
    train_tfms, val_tfms = build_transforms(image_size=image_size)

    train_ds = dataset_class(train_df, transform=train_tfms)
    val_ds = dataset_class(val_df, transform=val_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader

def train_model(
    model,
    train_loader,
    val_loader,
    train_one_epoch_fn,
    evaluate_fn,
    device,
    epochs=10,
    lr=1e-4,
    weight_decay=1e-4,
    target_names=None
):
    if target_names is None:
        target_names = ["well-being", "impaired"]

    model = model.to(device)
    # total = 767 + 631
    # w0 = total / (2 * 767)  # 0.91
    # w1 = total / (2 * 631)  # 1.11
    weights = torch.tensor([
        0.91,
        1.11
    ], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    # criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    history = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        train_loss = train_one_epoch_fn(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        y_true, y_pred = evaluate_fn(
            model,
            val_loader,
            device
        )

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            output_dict=True
        )

        print("Train loss:", train_loss)
        print("\nConfusion matrix:")
        print(cm)

        print("\nClassification report:")
        print(
            classification_report(
                y_true,
                y_pred,
                target_names=target_names
            )
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "confusion_matrix": cm,
            "classification_report": report
        })

    return model, history

def save_model(model, save_path, label_mapping=None):
    if label_mapping is None:
        label_mapping = LABEL_MAPPING

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_mapping": label_mapping
        },
        save_path
    )

    print(f"\nSaved model to {save_path}")

def run_training(
    train_df,
    val_df,
    dataset_class,
    train_one_epoch_fn,
    evaluate_fn,
    save_path="mouse_wellbeing_convnext_tiny.pt",
    epochs=10,
    batch_size=32,
    image_size=224,
    lr=1e-4,
    weight_decay=1e-4,
    num_workers=0,
    pretrained=True
):
    device = get_device()
    print("\nUsing device:", device)

    train_loader, val_loader = build_dataloaders(
        train_df=train_df,
        val_df=val_df,
        dataset_class=dataset_class,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size
    )

    model = build_convnext_tiny(
        num_classes=2,
        pretrained=pretrained
    )

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_one_epoch_fn=train_one_epoch_fn,
        evaluate_fn=evaluate_fn,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        target_names=["well-being", "impaired"]
    )

    save_model(
        model=model,
        save_path=save_path,
        label_mapping={
            0: "well-being",
            1: "impaired"
        }
    )

    return model, history


