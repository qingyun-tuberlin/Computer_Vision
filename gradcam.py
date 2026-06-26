import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(
            self.save_activation
        )
        self.backward_hook = target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, img_tensor, device, target_class=None):
        self.model.eval()

        img = img_tensor.unsqueeze(0).to(device)
        output = self.model(img)

        pred_class = output.argmax(dim=1).item()

        if target_class is None:
            target_class = pred_class

        score = output[0, target_class]

        self.model.zero_grad()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, pred_class, target_class

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()



def show_gradcam(
    model,
    dataset,
    index,
    device,
    gradcam,
    target_class=None,
    class_names=None
):
    if class_names is None:
        class_names = ["well-being", "impaired"]

    img_tensor, true_label = dataset[index]

    cam, pred_class, used_class = gradcam(
        img_tensor=img_tensor,
        device=device,
        target_class=target_class
    )

    img = denormalize(img_tensor)

    cam_resized = F.interpolate(
        torch.tensor(cam).unsqueeze(0).unsqueeze(0),
        size=img.shape[:2],
        mode="bilinear",
        align_corners=False
    ).squeeze().numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f"Original\nTrue: {class_names[int(true_label)]}")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cam_resized, cmap="jet")
    plt.title(f"Grad-CAM\nClass: {class_names[used_class]}")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(cam_resized, cmap="jet", alpha=0.45)
    plt.title(f"Overlay\nPred: {class_names[pred_class]}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()