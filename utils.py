from diffusers.loaders import StableDiffusionLoraLoaderMixin
from PIL import Image
import numpy as np
import cv2


def load_emolora(pipe, lora_path):
    """
    Load a LoRA model into specific UNet blocks of the pipeline.
    """
    target_blocks = [
        "unet.down_blocks.1.attentions.1",
        "unet.down_blocks.2.attentions.0",
        "unet.up_blocks.0.attentions.1",
        "unet.up_blocks.0.attentions.2"
    ]

    pipe.unload_lora_weights()
    state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(lora_path)
    filtered_state_dict = {k: v for k, v in state_dict.items() if any(block in k for block in target_blocks)}
    StableDiffusionLoraLoaderMixin.load_lora_into_unet(filtered_state_dict, None, pipe.unet)

    return pipe


def resize_img(img_pil):
    """Image Resize by Longer Side"""
    size = 1024
    w, h = img_pil.size
    if h > w:
        width = size * w // h
        height = size
    else:
        width = size
        height = size * h // w

    new_width = (width//8)*8
    new_height = (height//8)*8

    resized_img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
    return resized_img_pil


def color_fix(img1, img2):
    """Combine the brightness channel of img1 with the color channels of img2"""
    img2 = img2.resize(img1.size, Image.LANCZOS)
    
    lab1 = img1.convert('LAB')
    lab2 = img2.convert('LAB')

    L1, _, _ = lab1.split()
    _, A2, B2 = lab2.split()

    combined_lab = Image.merge('LAB', (L1, A2, B2))

    combined_rgb = combined_lab.convert('RGB')

    return combined_rgb


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def canny_detector(image:Image):
    """preprocess for controlnet-canny"""
    image = np.array(image)
    canny_img = cv2.Canny(image, 100, 200)
    canny_img = HWC3(canny_img)
    return Image.fromarray(canny_img)


def tile_detector(image:Image, ksize):
    """preprocess for controlnet-tile"""
    image_np = np.array(image)

    if ksize % 2 == 0:
        ksize += 1
    blurred_image = cv2.GaussianBlur(image_np, (ksize, ksize), sigmaX=(ksize / 2))

    return Image.fromarray(blurred_image)