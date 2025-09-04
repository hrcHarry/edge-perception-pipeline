from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import random

def _to_uint8(arr):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def gaussian(img, std=0.10):
    arr = np.array(img.convert("L")).astype(np.float32)
    noise = np.random.normal(0, std * 255.0, size=arr.shape)
    return _to_uint8(arr + noise)

def motion_blur(img, k=5):
    return img.convert("L").filter(ImageFilter.GaussianBlur(radius=max(1, k // 3)))

def occlusion(img, frac=0.25):
    im = img.convert("L").copy()
    w, h = im.size
    bw = int(w * (frac ** 0.5))
    bh = int(h * (frac ** 0.5))
    x = random.randint(0, max(0, w - bw))
    y = random.randint(0, max(0, h - bh))
    black = Image.new("L", (bw, bh), color=0)
    im.paste(black, (x, y))
    return im

def brightness(img, factor=1.4):
    return ImageEnhance.Brightness(img.convert("L")).enhance(factor)
