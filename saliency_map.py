import torch
import numpy as np
import matplotlib.pyplot as plt


def denormalize(img_tensor):
    """
    Undo ImageNet normalization for visualization.
    img_tensor: shape [3, H, W]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img = img_tensor.cpu() * std + mean
    img = torch.clamp(img, 0, 1)

    return img.permute(1, 2, 0).numpy()


def compute_saliency_map(model, img_tensor, device, target_class=None):
    """
    Compute vanilla gradient saliency map.
    img_tensor: one image tensor, shape [3, H, W]
    """

    model.eval()

    # add batch dimension: [3,H,W] -> [1,3,H,W]
    img = img_tensor.unsqueeze(0).to(device)

    # allow gradient calculation on input image
    img.requires_grad_()

    output = model(img)

    pred_class = output.argmax(dim=1).item()

    if target_class is None:
        target_class = pred_class

    score = output[0, target_class]

    model.zero_grad()
    score.backward()

    # gradient w.r.t. input image
    saliency = img.grad.abs()

    # take max over RGB channels
    saliency, _ = torch.max(saliency, dim=1)

    saliency = saliency.squeeze().detach().cpu().numpy()

    # normalize to [0, 1]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    return saliency, pred_class, target_class


def show_saliency(model, dataset, index, device, target_class=None, class_names=None):
    if class_names is None:
        class_names = ["well-being", "impaired"]

    img_tensor, true_label = dataset[index]

    saliency, pred_class, used_class = compute_saliency_map(
        model=model,
        img_tensor=img_tensor,
        device=device,
        target_class=target_class
    )

    img = denormalize(img_tensor)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f"Original\nTrue: {class_names[true_label]}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(saliency, cmap="hot")
    plt.title(f"Saliency\nClass: {class_names[used_class]}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(saliency, cmap="hot", alpha=0.45)
    plt.title(f"Overlay\nPred: {class_names[pred_class]}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()



def show_saliency_batch(
    model,
    dataset,
    device,
    start_idx=0,
    n_images=10
):
    
    fig, axes = plt.subplots(
        n_images,
        3,
        figsize=(12, 4*n_images)
    )

    for i in range(n_images):

        idx = start_idx + i

        img_tensor, true_label = dataset[idx]

        saliency, pred_class, used_class = compute_saliency_map(
            model=model,
            img_tensor=img_tensor,
            device=device
        )

        img = denormalize(img_tensor)

        axes[i,0].imshow(img)
        axes[i,0].set_title(
            f"Original\nTrue={true_label}"
        )
        axes[i,0].axis("off")

        axes[i,1].imshow(
            saliency,
            cmap="hot"
        )
        axes[i,1].set_title(
            f"Saliency\nPred={pred_class}"
        )
        axes[i,1].axis("off")

        axes[i,2].imshow(img)
        axes[i,2].imshow(
            saliency,
            cmap="hot",
            alpha=0.45
        )
        axes[i,2].set_title("Overlay")
        axes[i,2].axis("off")

    plt.tight_layout()
    plt.show()