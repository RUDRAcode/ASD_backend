import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.cm as cm

def find_last_conv_layer(model: nn.Module) -> nn.Module:
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise ValueError("No Conv2d layer found in the model.")
    return last_conv

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def denormalize(tensor):
    img = tensor.clone().cpu()
    for t, m, s in zip(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
        t.mul_(s).add_(m)
    img = (img.clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)
    return img

def overlay_cam_on_image(image_uint8, cam_01, alpha=0.35):

    h, w, _ = image_uint8.shape
    cam_resized = np.array(Image.fromarray((cam_01*255).astype(np.uint8)).resize((w, h)))
    cam_resized = cam_resized.astype(np.float32)/255.0


    colormap = cm.get_cmap('jet')
    heatmap = colormap(cam_resized)[:, :, :3]


    overlay = (1 - alpha) * (image_uint8.astype(np.float32)/255.0) + alpha * heatmap
    overlay = np.clip(overlay*255.0, 0, 255).astype(np.uint8)
    return overlay



class GradCAMPlusPlus:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.fwd_handle = target_layer.register_forward_hook(fwd_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

    def __del__(self):
        try:
            self.fwd_handle.remove()
            self.bwd_handle.remove()
        except:
            pass

    def _normalize_cam(self, cam):
        cam = torch.relu(cam)
        cam -= cam.min()
        if cam.max() != 0:
            cam /= cam.max()
        return cam

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)


        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())


        self.model.zero_grad(set_to_none=True)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        activations = self.activations
        grads = self.gradients
        B, C, H, W = grads.shape


        numerator = grads ** 2
        denominator = 2 * grads ** 2 + (activations * grads ** 3).sum(dim=(2, 3), keepdim=True)
        denominator = torch.where(denominator != 0, denominator, torch.ones_like(denominator))
        alpha = numerator / denominator

        positive_grads = F.relu(score.exp() * grads)
        weights = (alpha * positive_grads).sum(dim=(2, 3), keepdim=True)

        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = self._normalize_cam(cam).detach().cpu().numpy()

        return cam, class_idx