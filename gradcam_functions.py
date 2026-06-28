"""
gradcam_functions.py

Reusable Grad-CAM visualization utilities for binary image classification.

Typical usage after training:

    from gradcam_functions import visualize_gradcam

    results = visualize_gradcam(
        model=model,
        val_df=val_df,
        dataset_class=MouseDataset,
        target_layer=model.features[-1][-1].block[-1],
        output_dir="output/gradcam_muzzle_only",
        num_images=50,
        transform=val_tfms,
        class_names=["well-being", "impaired"]
    )

This file assumes your dataset returns:

    img_tensor, label = dataset[index]

where img_tensor is a normalized torch Tensor with shape [C, H, W].
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


DEFAULT_IMAGENET_MEAN = (0.485, 0.456, 0.406)
DEFAULT_IMAGENET_STD = (0.229, 0.224, 0.225)


# =========================================================
# Basic utilities
# =========================================================

def get_device() -> torch.device:
    """
    Return the best available PyTorch device.

    Priority:
        1. Apple Silicon MPS
        2. CUDA
        3. CPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def denormalize(
    img_tensor: torch.Tensor,
    mean: Sequence[float] = DEFAULT_IMAGENET_MEAN,
    std: Sequence[float] = DEFAULT_IMAGENET_STD,
) -> np.ndarray:
    """
    Convert a normalized image tensor back to a displayable numpy image.

    Parameters
    ----------
    img_tensor:
        Tensor with shape [C, H, W].
    mean:
        Normalization mean used in the dataset transform.
    std:
        Normalization std used in the dataset transform.

    Returns
    -------
    np.ndarray
        Image array with shape [H, W, C], clipped to [0, 1].
    """
    img = img_tensor.detach().cpu().clone()

    mean_tensor = torch.tensor(mean).view(-1, 1, 1)
    std_tensor = torch.tensor(std).view(-1, 1, 1)

    img = img * std_tensor + mean_tensor
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0.0, 1.0)

    return img


def safe_class_name(class_names: Sequence[str], class_idx: int) -> str:
    """
    Convert a class index to a readable class name.
    Falls back to class_<idx> if the index is out of range.
    """
    if 0 <= int(class_idx) < len(class_names):
        return str(class_names[int(class_idx)])
    return f"class_{int(class_idx)}"


def sanitize_filename(text: str) -> str:
    """
    Make a string safe for filenames.
    """
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
    )


# =========================================================
# Grad-CAM core class
# =========================================================

class GradCAM:
    """
    Grad-CAM implementation for CNN-like PyTorch models.

    Parameters
    ----------
    model:
        Trained PyTorch model.
    target_layer:
        The convolutional layer whose activations should be used for Grad-CAM.
        Example for EfficientNet-like models:
            target_layer = model.features[-1][-1].block[-1]
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(
        self,
        img_tensor: torch.Tensor,
        device: Union[str, torch.device],
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, int, np.ndarray]:
        """
        Generate a Grad-CAM heatmap for one image.

        Parameters
        ----------
        img_tensor:
            Single image tensor with shape [C, H, W].
        device:
            Device used for inference.
        target_class:
            Class index to explain. If None, explains the predicted class.

        Returns
        -------
        cam:
            Normalized Grad-CAM heatmap with shape [h, w].
        pred_class:
            Predicted class index.
        used_class:
            Class index actually used for Grad-CAM.
        probs:
            Softmax probabilities as numpy array.
        """
        self.model.eval()
        device = torch.device(device)
        self.model.to(device)

        img = img_tensor.unsqueeze(0).to(device)

        output = self.model(img)
        probs = torch.softmax(output, dim=1).detach().cpu().numpy()[0]
        pred_class = int(output.argmax(dim=1).item())

        if target_class is None:
            used_class = pred_class
        else:
            used_class = int(target_class)

        score = output[0, used_class]

        self.model.zero_grad(set_to_none=True)
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "GradCAM did not capture activations/gradients. "
                "Check whether target_layer belongs to the model forward path."
            )

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()

        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam, pred_class, used_class, probs

    def remove_hooks(self) -> None:
        """
        Remove registered forward and backward hooks.
        Call this when GradCAM is no longer needed.
        """
        self.forward_hook.remove()
        self.backward_hook.remove()


# =========================================================
# Single-image Grad-CAM saving
# =========================================================

def save_gradcam_for_index(
    model: torch.nn.Module,
    dataset,
    index: int,
    device: Union[str, torch.device],
    gradcam: GradCAM,
    output_dir: Union[str, Path] = "output/gradcam",
    target_class: Optional[int] = None,
    class_names: Sequence[str] = ("well-being", "impaired"),
    mean: Sequence[float] = DEFAULT_IMAGENET_MEAN,
    std: Sequence[float] = DEFAULT_IMAGENET_STD,
    alpha: float = 0.45,
    cmap: str = "jet",
    dpi: int = 300,
    show: bool = False,
) -> Dict[str, object]:
    """
    Generate and save Grad-CAM visualization for one dataset index.

    The saved figure contains:
        1. Original image
        2. Grad-CAM heatmap
        3. Overlay

    Returns metadata about the saved image.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_tensor, true_label = dataset[index]
    true_label = int(true_label)

    cam, pred_class, used_class, probs = gradcam(
        img_tensor=img_tensor,
        device=device,
        target_class=target_class,
    )

    img = denormalize(img_tensor, mean=mean, std=std)

    cam_resized = F.interpolate(
        torch.tensor(cam).unsqueeze(0).unsqueeze(0),
        size=img.shape[:2],
        mode="bilinear",
        align_corners=False,
    ).squeeze().numpy()

    true_name = safe_class_name(class_names, true_label)
    pred_name = safe_class_name(class_names, pred_class)
    used_name = safe_class_name(class_names, used_class)

    true_name_file = sanitize_filename(true_name)
    pred_name_file = sanitize_filename(pred_name)
    used_name_file = sanitize_filename(used_name)

    correctness = "correct" if true_label == pred_class else "wrong"

    filename = (
        f"gradcam_{index:04d}_"
        f"true_{true_name_file}_"
        f"pred_{pred_name_file}_"
        f"target_{used_name_file}_"
        f"{correctness}.png"
    )
    save_path = output_dir / filename

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f"Original\nTrue: {true_name}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cam_resized, cmap=cmap)
    plt.title(f"Grad-CAM\nClass: {used_name}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(cam_resized, cmap=cmap, alpha=alpha)
    plt.title(f"Overlay\nPred: {pred_name}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return {
        "index": index,
        "true_label": true_label,
        "pred_class": pred_class,
        "target_class": used_class,
        "true_name": true_name,
        "pred_name": pred_name,
        "target_name": used_name,
        "correct": true_label == pred_class,
        "probabilities": probs,
        "save_path": str(save_path),
    }


# =========================================================
# Batch Grad-CAM generation
# =========================================================

def visualize_gradcam(
    model: torch.nn.Module,
    val_df,
    dataset_class,
    target_layer: torch.nn.Module,
    output_dir: Union[str, Path] = "output/gradcam",
    num_images: int = 50,
    transform: Optional[Callable] = None,
    build_transforms_fn: Optional[Callable] = None,
    device: Optional[Union[str, torch.device]] = None,
    target_class: Optional[int] = None,
    class_names: Sequence[str] = ("well-being", "impaired"),
    only_misclassified: bool = False,
    only_class: Optional[int] = None,
    random_sample: bool = False,
    random_seed: int = 42,
    mean: Sequence[float] = DEFAULT_IMAGENET_MEAN,
    std: Sequence[float] = DEFAULT_IMAGENET_STD,
    alpha: float = 0.45,
    cmap: str = "jet",
    dpi: int = 300,
    show: bool = False,
    remove_hooks: bool = True,
) -> Dict[str, object]:
    """
    Generate and save Grad-CAM visualizations for multiple validation images.

    Parameters
    ----------
    model:
        Trained PyTorch model.
    val_df:
        Validation dataframe used to construct the validation dataset.
    dataset_class:
        Dataset class, e.g. MouseDataset.
    target_layer:
        Layer used for Grad-CAM.
    output_dir:
        Directory where Grad-CAM images are saved.
    num_images:
        Maximum number of images to save.
    transform:
        Validation transform. If None, build_transforms_fn must be provided.
    build_transforms_fn:
        Optional function returning train/validation transforms.
        Expected usage:
            _, val_tfms = build_transforms_fn()
    device:
        PyTorch device. If None, automatically selected by get_device().
    target_class:
        Class to explain. If None, explains the model's predicted class.
    class_names:
        Names of classes ordered by class index.
    only_misclassified:
        If True, only save examples where prediction != true label.
    only_class:
        If 0 or 1, only save examples whose true label equals this class.
    random_sample:
        If True, sample candidate indices randomly.
        If False, process dataset in original order.
    random_seed:
        Seed for random sampling.
    mean, std:
        Normalization parameters used for denormalization.
    alpha:
        Overlay transparency.
    cmap:
        Matplotlib colormap for heatmap.
    dpi:
        Saved image DPI.
    show:
        If True, display figures while saving. Usually False for batch export.
    remove_hooks:
        If True, remove Grad-CAM hooks after generation.

    Returns
    -------
    dict
        Summary metadata, including saved paths and prediction counts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = get_device()
    else:
        device = torch.device(device)

    if transform is None:
        if build_transforms_fn is None:
            raise ValueError(
                "Either transform or build_transforms_fn must be provided. "
                "Example: _, val_tfms = build_transforms(); pass transform=val_tfms."
            )
        _, transform = build_transforms_fn()

    val_dataset = dataset_class(val_df, transform=transform)

    indices = list(range(len(val_dataset)))
    if random_sample:
        rng = np.random.default_rng(random_seed)
        rng.shuffle(indices)

    gradcam = GradCAM(model=model, target_layer=target_layer)

    saved_results: List[Dict[str, object]] = []
    checked = 0

    try:
        for idx in indices:
            if len(saved_results) >= num_images:
                break

            img_tensor, true_label = val_dataset[idx]
            true_label = int(true_label)

            if only_class is not None and true_label != int(only_class):
                continue

            cam, pred_class, used_class, probs = gradcam(
                img_tensor=img_tensor,
                device=device,
                target_class=target_class,
            )
            checked += 1

            if only_misclassified and pred_class == true_label:
                continue

            # Reuse already-computed Grad-CAM result by temporarily writing it
            # through a lightweight local plotting block instead of calling
            # gradcam twice.
            img = denormalize(img_tensor, mean=mean, std=std)
            cam_resized = F.interpolate(
                torch.tensor(cam).unsqueeze(0).unsqueeze(0),
                size=img.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze().numpy()

            true_name = safe_class_name(class_names, true_label)
            pred_name = safe_class_name(class_names, pred_class)
            used_name = safe_class_name(class_names, used_class)
            correctness = "correct" if true_label == pred_class else "wrong"

            filename = (
                f"gradcam_{idx:04d}_"
                f"true_{sanitize_filename(true_name)}_"
                f"pred_{sanitize_filename(pred_name)}_"
                f"target_{sanitize_filename(used_name)}_"
                f"{correctness}.png"
            )
            save_path = output_dir / filename

            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title(f"Original\nTrue: {true_name}")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(cam_resized, cmap=cmap)
            plt.title(f"Grad-CAM\nClass: {used_name}")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(img)
            plt.imshow(cam_resized, cmap=cmap, alpha=alpha)
            plt.title(f"Overlay\nPred: {pred_name}")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

            if show:
                plt.show()
            else:
                plt.close()

            saved_results.append(
                {
                    "index": idx,
                    "true_label": true_label,
                    "pred_class": pred_class,
                    "target_class": used_class,
                    "true_name": true_name,
                    "pred_name": pred_name,
                    "target_name": used_name,
                    "correct": true_label == pred_class,
                    "probabilities": probs,
                    "save_path": str(save_path),
                }
            )

    finally:
        if remove_hooks:
            gradcam.remove_hooks()

    num_correct = sum(1 for item in saved_results if item["correct"])
    num_wrong = len(saved_results) - num_correct

    return {
        "saved_images": len(saved_results),
        "checked_images": checked,
        "correct_saved": num_correct,
        "wrong_saved": num_wrong,
        "output_dir": str(output_dir),
        "results": saved_results,
    }


# =========================================================
# Convenience function if you already built val_dataset
# =========================================================

def visualize_gradcam_from_dataset(
    model: torch.nn.Module,
    dataset,
    target_layer: torch.nn.Module,
    output_dir: Union[str, Path] = "output/gradcam",
    num_images: int = 50,
    device: Optional[Union[str, torch.device]] = None,
    target_class: Optional[int] = None,
    class_names: Sequence[str] = ("well-being", "impaired"),
    only_misclassified: bool = False,
    only_class: Optional[int] = None,
    random_sample: bool = False,
    random_seed: int = 42,
    mean: Sequence[float] = DEFAULT_IMAGENET_MEAN,
    std: Sequence[float] = DEFAULT_IMAGENET_STD,
    alpha: float = 0.45,
    cmap: str = "jet",
    dpi: int = 300,
    show: bool = False,
    remove_hooks: bool = True,
) -> Dict[str, object]:
    """
    Same as visualize_gradcam(), but uses an already constructed dataset.

    Useful if you already have:

        val_dataset = MouseDataset(val_df, transform=val_tfms)

    Then call:

        visualize_gradcam_from_dataset(
            model=model,
            dataset=val_dataset,
            target_layer=model.features[-1][-1].block[-1]
        )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = get_device()
    else:
        device = torch.device(device)

    indices = list(range(len(dataset)))
    if random_sample:
        rng = np.random.default_rng(random_seed)
        rng.shuffle(indices)

    gradcam = GradCAM(model=model, target_layer=target_layer)
    saved_results: List[Dict[str, object]] = []
    checked = 0

    try:
        for idx in indices:
            if len(saved_results) >= num_images:
                break

            img_tensor, true_label = dataset[idx]
            true_label = int(true_label)

            if only_class is not None and true_label != int(only_class):
                continue

            cam, pred_class, used_class, probs = gradcam(
                img_tensor=img_tensor,
                device=device,
                target_class=target_class,
            )
            checked += 1

            if only_misclassified and pred_class == true_label:
                continue

            img = denormalize(img_tensor, mean=mean, std=std)
            cam_resized = F.interpolate(
                torch.tensor(cam).unsqueeze(0).unsqueeze(0),
                size=img.shape[:2],
                mode="bilinear",
                align_corners=False,
            ).squeeze().numpy()

            true_name = safe_class_name(class_names, true_label)
            pred_name = safe_class_name(class_names, pred_class)
            used_name = safe_class_name(class_names, used_class)
            correctness = "correct" if true_label == pred_class else "wrong"

            filename = (
                f"gradcam_{idx:04d}_"
                f"true_{sanitize_filename(true_name)}_"
                f"pred_{sanitize_filename(pred_name)}_"
                f"target_{sanitize_filename(used_name)}_"
                f"{correctness}.png"
            )
            save_path = output_dir / filename

            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title(f"Original\nTrue: {true_name}")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(cam_resized, cmap=cmap)
            plt.title(f"Grad-CAM\nClass: {used_name}")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(img)
            plt.imshow(cam_resized, cmap=cmap, alpha=alpha)
            plt.title(f"Overlay\nPred: {pred_name}")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

            if show:
                plt.show()
            else:
                plt.close()

            saved_results.append(
                {
                    "index": idx,
                    "true_label": true_label,
                    "pred_class": pred_class,
                    "target_class": used_class,
                    "true_name": true_name,
                    "pred_name": pred_name,
                    "target_name": used_name,
                    "correct": true_label == pred_class,
                    "probabilities": probs,
                    "save_path": str(save_path),
                }
            )

    finally:
        if remove_hooks:
            gradcam.remove_hooks()

    num_correct = sum(1 for item in saved_results if item["correct"])
    num_wrong = len(saved_results) - num_correct

    return {
        "saved_images": len(saved_results),
        "checked_images": checked,
        "correct_saved": num_correct,
        "wrong_saved": num_wrong,
        "output_dir": str(output_dir),
        "results": saved_results,
    }
