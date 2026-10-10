import numpy as onp  # real NumPy for the image work (your `np` is MLX)
from PIL import Image
from pathlib import Path

def image_to_vector(path, invert=None, threshold=0.2, preview=False):
    '''
    Converts an image file into a flat 784-value float32 vector in MNIST format.

    invert: True/False to force it, or None to guess from the average brightness.
    threshold: after contrast stretching, pixels below this count as background.
    '''
    img = Image.open(path).convert('L')
    arr = onp.asarray(img, dtype=onp.float32) / 255.0

    # Digit should be bright on a dark background
    if invert is None:
        invert = arr.mean() > 0.5
    if invert:
        arr = 1.0 - arr

    # Stretch contrast, then zero out faint noise (e.g. JPEG artefacts)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    arr[arr < threshold] = 0.0

    # Crop to the digit's bounding box
    ys, xs = onp.nonzero(arr)
    if len(ys) == 0:
        raise ValueError("No digit found in the image. Try a lower threshold or set invert manually.")
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # Shrink so the longest side is 20 px, keeping the aspect ratio
    h, w = arr.shape
    scale = 20.0 / max(h, w)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    digit = Image.fromarray((arr * 255).astype(onp.uint8)).resize((new_w, new_h), Image.LANCZOS)
    digit = onp.asarray(digit, dtype=onp.float32)

    # Paste onto a black 28x28 canvas
    canvas = onp.zeros((28, 28), dtype=onp.float32)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = digit

    # Shift so the centre of mass sits in the middle, as in MNIST
    total = canvas.sum()
    yy, xx = onp.indices(canvas.shape)
    cy = (yy * canvas).sum() / total
    cx = (xx * canvas).sum() / total
    canvas = onp.roll(canvas, (int(round(13.5 - cy)), int(round(13.5 - cx))), axis=(0, 1))

    if preview:
        import matplotlib.pyplot as plt
        plt.imshow(canvas, cmap='gray')
        plt.axis('off')
        plt.show()

    return canvas.reshape(784)

