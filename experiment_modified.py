import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report

from model import *


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
    # for image transformation
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


def get_primary_metric(report, metric_name="macro_f1"):
    """
    Select the validation metric used for:
        1. ReduceLROnPlateau
        2. saving the best model
        3. early stopping

    For this MGS binary classification task, macro F1 is a good default because
    it treats both classes more fairly than plain accuracy.
    """
    if metric_name == "macro_f1":
        return report["macro avg"]["f1-score"]
    elif metric_name == "weighted_f1":
        return report["weighted avg"]["f1-score"]
    elif metric_name == "impaired_f1":
        return report["impaired"]["f1-score"]
    elif metric_name == "impaired_recall":
        return report["impaired"]["recall"]
    elif metric_name == "accuracy":
        return report["accuracy"]
    else:
        raise ValueError(
            f"Unknown metric_name: {metric_name}. "
            "Use one of: macro_f1, weighted_f1, impaired_f1, "
            "impaired_recall, accuracy."
        )


def save_checkpoint(
    model,
    save_path,
    label_mapping=None,
    epoch=None,
    best_metric=None,
    metric_name=None,
    optimizer=None,
):
    if label_mapping is None:
        label_mapping = LABEL_MAPPING

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "label_mapping": label_mapping,
    }

    if epoch is not None:
        checkpoint["epoch"] = epoch
    if best_metric is not None:
        checkpoint["best_metric"] = best_metric
    if metric_name is not None:
        checkpoint["metric_name"] = metric_name
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, save_path)
    print(f"Saved best model to {save_path}")


def train_model_with_lr_schedule_early_stop_best_model(
    model,
    train_loader,
    val_loader,
    train_one_epoch_fn,
    evaluate_fn,
    device,
    epochs=30,
    lr=1e-4,
    weight_decay=1e-4,
    target_names=None,
    save_path="mouse_wellbeing_convnext_tiny_best.pt",
    metric_name="macro_f1",
    scheduler_factor=0.5,
    scheduler_patience=2,
    early_stopping_patience=5,
    min_delta=1e-4,
):
    """
    Train model with:
        - AdamW optimizer
        - ReduceLROnPlateau learning-rate scheduler
        - best-model saving
        - early stopping

    The default primary metric is macro F1.
    """
    if target_names is None:
        target_names = ["well-being", "impaired"]

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # add ReduceLROnPleateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=scheduler_factor,
        patience=scheduler_patience
    )

    history = []

    # for saving best model
    best_metric = float("-inf")
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        current_epoch = epoch + 1
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {current_epoch}/{epochs}")
        print(f"Current LR: {current_lr:.8f}")

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
            output_dict=True,
            zero_division=0
        )

        current_metric = get_primary_metric(
            report,
            metric_name=metric_name
        )

        print("Train loss:", train_loss)
        print(f"Validation {metric_name}: {current_metric:.4f}")

        print("\nConfusion matrix:")
        print(cm)

        print("\nClassification report:")
        print(
            classification_report(
                y_true,
                y_pred,
                target_names=target_names,
                zero_division=0
            )
        )

        # Reduce LR when the validation metric plateaus.
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(current_metric)
        new_lr = optimizer.param_groups[0]["lr"]

        if new_lr < old_lr:
            print(f"Learning rate reduced: {old_lr:.8f} -> {new_lr:.8f}")

        improved = current_metric > best_metric + min_delta

        if improved:
            best_metric = current_metric
            best_epoch = current_epoch
            epochs_without_improvement = 0

            # Keep a CPU copy in memory and also save it to disk.
            best_state_dict = copy.deepcopy(model.state_dict())

            save_checkpoint(
                model=model,
                save_path=save_path,
                label_mapping=LABEL_MAPPING,
                epoch=current_epoch,
                best_metric=best_metric,
                metric_name=metric_name,
                optimizer=optimizer,
            )

            print(
                f"New best model: epoch={best_epoch}, "
                f"{metric_name}={best_metric:.4f}"
            )
        else:
            epochs_without_improvement += 1
            print(
                "No improvement. "
                f"Early-stopping counter: "
                f"{epochs_without_improvement}/{early_stopping_patience}"
            )

        history.append({
            "epoch": current_epoch,
            "train_loss": train_loss,
            "lr": current_lr,
            "val_metric_name": metric_name,
            "val_metric": current_metric,
            "best_metric_so_far": best_metric,
            "best_epoch_so_far": best_epoch,
            "confusion_matrix": cm,
            "classification_report": report,
        })

        if epochs_without_improvement >= early_stopping_patience:
            print(
                "\nEarly stopping triggered. "
                f"Best epoch: {best_epoch}, "
                f"best {metric_name}: {best_metric:.4f}"
            )
            break

    # Return the best model, not the last-epoch model.
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(
            f"\nLoaded best model from epoch {best_epoch} "
            f"with {metric_name}={best_metric:.4f}"
        )

    return model, history


def save_model(model, save_path, label_mapping=None):
    """
    Kept for compatibility with your old code.
    For normal training, train_model already saves the best model.
    """
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


def run_training_with_lr_schedule_early_stop_best_model(
    train_df,
    val_df,
    dataset_class,
    train_one_epoch_fn,
    evaluate_fn,
    save_path="mouse_wellbeing_convnext_tiny_best.pt",
    epochs=30,
    batch_size=32,
    image_size=224,
    lr=1e-4,
    weight_decay=1e-4,
    num_workers=0,
    pretrained=True,
    metric_name="macro_f1",
    scheduler_factor=0.5,
    scheduler_patience=2,
    early_stopping_patience=5,
    min_delta=1e-4,
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

    model = build_model(
        model_name="convnext_tiny",
        num_classes=2,
        pretrained=pretrained
    )

    model, history = train_model_with_lr_schedule_early_stop_best_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_one_epoch_fn=train_one_epoch_fn,
        evaluate_fn=evaluate_fn,
        device=device,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        target_names=["well-being", "impaired"],
        save_path=save_path,
        metric_name=metric_name,
        scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience,
        early_stopping_patience=early_stopping_patience,
        min_delta=min_delta,
    )

    return model, history
