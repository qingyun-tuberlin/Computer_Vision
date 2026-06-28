from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
)


def get_default_device():
    """
    Return the best available PyTorch device.
    Priority: Apple Silicon MPS -> CUDA -> CPU.
    """
    return torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )


def get_validation_predictions(
    model,
    val_loader,
    evaluate_with_probs_fn,
    device=None,
):
    """
    Run the model on the validation loader and return y_true, y_pred, y_prob.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.
    val_loader : DataLoader
        Validation DataLoader.
    evaluate_with_probs_fn : callable
        Your existing evaluate_with_probs function.
    device : torch.device or str, optional
        Device used for evaluation. If None, automatically selects mps/cuda/cpu.

    Returns
    -------
    y_true, y_pred, y_prob : np.ndarray
        Ground-truth labels, predicted labels, and class probabilities.
    """
    if device is None:
        device = get_default_device()

    model = model.to(device)

    y_true, y_pred, y_prob = evaluate_with_probs_fn(
        model=model,
        loader=val_loader,
        device=device,
    )

    return np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob)


def plot_probability_distribution(
    y_true,
    p_positive,
    output_dir="output",
    filename="probability_distribution.png",
    title="Distribution of P(impaired) on validation set",
    positive_label_name="True impaired",
    negative_label_name="True well-being",
    bins=20,
    show=True,
):
    """
    Plot predicted probability distribution for class 0 and class 1.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / filename

    y_true = np.asarray(y_true)
    p_positive = np.asarray(p_positive)

    p_negative_class = p_positive[y_true == 0]
    p_positive_class = p_positive[y_true == 1]

    plt.figure(figsize=(8, 5))
    plt.hist(p_negative_class, bins=bins, alpha=0.6, label=negative_label_name)
    plt.hist(p_positive_class, bins=bins, alpha=0.6, label=positive_label_name)
    plt.xlabel("Predicted probability: P(impaired)")
    plt.ylabel("Number of images")
    plt.title(title)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def plot_roc_auc(
    y_true,
    p_positive,
    output_dir="output",
    filename="roc_auc.png",
    title="ROC curve",
    show=True,
):
    """
    Plot ROC curve and return AUC.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / filename

    y_true = np.asarray(y_true)
    p_positive = np.asarray(p_positive)

    fpr, tpr, _ = roc_curve(y_true, p_positive)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return roc_auc, save_path


def plot_precision_recall_curve(
    y_true,
    p_positive,
    output_dir="output",
    filename="precision_recall_curve.png",
    title="Precision-Recall curve",
    show=True,
):
    """
    Plot precision-recall curve and return average precision.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / filename

    y_true = np.asarray(y_true)
    p_positive = np.asarray(p_positive)

    precision, recall, _ = precision_recall_curve(y_true, p_positive)
    ap = average_precision_score(y_true, p_positive)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, linewidth=2, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return ap, save_path


def find_best_macro_f1_threshold(
    y_true,
    p_positive,
    threshold_min=0.05,
    threshold_max=0.95,
    step=0.01,
):
    """
    Search for the classification threshold that gives the best macro F1.
    """
    y_true = np.asarray(y_true)
    p_positive = np.asarray(p_positive)

    thresholds = np.arange(threshold_min, threshold_max, step)
    f1_scores = []

    for threshold in thresholds:
        y_pred_threshold = (p_positive >= threshold).astype(int)
        score = f1_score(y_true, y_pred_threshold, average="macro")
        f1_scores.append(score)

    f1_scores = np.asarray(f1_scores)
    best_idx = int(np.argmax(f1_scores))

    return {
        "thresholds": thresholds,
        "macro_f1_scores": f1_scores,
        "best_threshold": float(thresholds[best_idx]),
        "best_macro_f1": float(f1_scores[best_idx]),
    }


def plot_threshold_vs_macro_f1(
    thresholds,
    macro_f1_scores,
    output_dir="output",
    filename="threshold_vs_macro_f1.png",
    title="Macro F1 vs classification threshold",
    show=True,
):
    """
    Plot macro F1 as a function of the classification threshold.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / filename

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, macro_f1_scores)
    plt.xlabel("Threshold")
    plt.ylabel("Macro F1")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return save_path


def visualize_binary_classifier_results(
    model,
    val_loader,
    evaluate_with_probs_fn,
    output_dir="output",
    prefix="muzzle_only",
    title_prefix="Muzzle Only",
    positive_class_index=1,
    device=None,
    show=True,
):
    """
    Complete visualization pipeline after training.

    It creates and saves:
    1. Probability distribution of P(positive class)
    2. ROC-AUC curve
    3. Precision-recall curve
    4. Macro F1 vs threshold curve

    Parameters
    ----------
    model : torch.nn.Module
        Trained model returned by run_training_with_lr_schedule_early_stop_best_model.
    val_loader : DataLoader
        Validation DataLoader.
    evaluate_with_probs_fn : callable
        Your existing evaluate_with_probs function.
    output_dir : str or Path
        Folder for output figures.
    prefix : str
        Prefix used in saved file names.
    title_prefix : str
        Prefix used in plot titles.
    positive_class_index : int
        Probability column for the positive class. For impaired = label 1, use 1.
    device : torch.device or str, optional
        Device used for evaluation. If None, automatically selects mps/cuda/cpu.
    show : bool
        If True, display plots in notebook. If False, save only.

    Returns
    -------
    dict
        Metrics, best threshold, predictions, probabilities, and saved figure paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred, y_prob = get_validation_predictions(
        model=model,
        val_loader=val_loader,
        evaluate_with_probs_fn=evaluate_with_probs_fn,
        device=device,
    )

    p_positive = y_prob[:, positive_class_index]

    prob_path = plot_probability_distribution(
        y_true=y_true,
        p_positive=p_positive,
        output_dir=output_dir,
        filename=f"{prefix}_probability_distribution.png",
        title=f"({title_prefix}) Distribution of P(impaired) on validation set",
        show=show,
    )

    roc_auc, roc_path = plot_roc_auc(
        y_true=y_true,
        p_positive=p_positive,
        output_dir=output_dir,
        filename=f"{prefix}_roc_auc.png",
        title=f"({title_prefix}) ROC curve",
        show=show,
    )

    average_precision, pr_path = plot_precision_recall_curve(
        y_true=y_true,
        p_positive=p_positive,
        output_dir=output_dir,
        filename=f"{prefix}_precision_recall_curve.png",
        title=f"({title_prefix}) Precision-Recall curve",
        show=show,
    )

    threshold_result = find_best_macro_f1_threshold(
        y_true=y_true,
        p_positive=p_positive,
    )

    threshold_path = plot_threshold_vs_macro_f1(
        thresholds=threshold_result["thresholds"],
        macro_f1_scores=threshold_result["macro_f1_scores"],
        output_dir=output_dir,
        filename=f"{prefix}_threshold_vs_macro_f1.png",
        title=f"({title_prefix}) Macro F1 vs classification threshold",
        show=show,
    )

    print(f"AUC: {roc_auc:.4f}")
    print(f"Average precision: {average_precision:.4f}")
    print(f"Best threshold: {threshold_result['best_threshold']:.2f}")
    print(f"Best macro F1: {threshold_result['best_macro_f1']:.4f}")
    print(f"Saved figures to: {output_dir}")

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "p_positive": p_positive,
        "auc": roc_auc,
        "average_precision": average_precision,
        "best_threshold": threshold_result["best_threshold"],
        "best_macro_f1": threshold_result["best_macro_f1"],
        "thresholds": threshold_result["thresholds"],
        "macro_f1_scores": threshold_result["macro_f1_scores"],
        "figure_paths": {
            "probability_distribution": prob_path,
            "roc_auc": roc_path,
            "precision_recall_curve": pr_path,
            "threshold_vs_macro_f1": threshold_path,
        },
    }
