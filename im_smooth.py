import numpy as np
import cv2
from scipy.ndimage import convolve, gaussian_filter, median_filter


def imsmooth(I, name="gaussian", *args):
    if I is None:
        raise ValueError("imsmooth: First argument must be an image")

    I = np.array(I, dtype=np.float64)

    name = name.lower()
    J = None

    if name == "gaussian":
        s = 0.5 if len(args) == 0 else args[0]
        if not isinstance(s, (int, float)) or s <= 0:
            raise ValueError("imsmooth: Third argument must be a positive scalar for Gaussian smoothing")
        J = gaussian_filter(I, sigma=s, mode='nearest')

    elif name == "average":
        s = (3, 3) if len(args) == 0 else args[0]
        if isinstance(s, int):
            s = (s, s)
        elif isinstance(s, (list, tuple)) and len(s) == 2 and all(isinstance(v, int) and v > 0 for v in s):
            pass
        else:
            raise ValueError("imsmooth: Third argument must be a positive scalar or two-vector for averaging")
        kernel = np.ones(s) / np.prod(s)
        J = convolve(I, kernel, mode='nearest')

    elif name == "disk":
        r = 5 if len(args) == 0 else args[0]
        if not isinstance(r, (int, float)) or r <= 0:
            raise ValueError("imsmooth: Third argument must be a positive scalar for disk averaging")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
        kernel = kernel / kernel.sum()
        J = convolve(I, kernel, mode='nearest')

    elif name == "median":
        s = (3, 3) if len(args) == 0 else args[0]
        if isinstance(s, int):
            s = (s, s)
        elif isinstance(s, (list, tuple)) and len(s) == 2 and all(isinstance(v, int) and v > 0 for v in s):
            pass
        else:
            raise ValueError("imsmooth: Third argument must be a positive scalar or two-vector for median filtering")
        J = median_filter(I, size=s, mode='nearest')

    elif name == "bilateral":
        sigma_d = 2 if len(args) < 1 else args[0]
        sigma_r = 10 / 255 if len(args) < 2 else args[1]
        if not isinstance(sigma_d, (int, float)) or sigma_d <= 0:
            raise ValueError("imsmooth: Spread of closeness function must be a positive scalar")
        if not isinstance(sigma_r, (int, float)) or sigma_r <= 0:
            raise ValueError("imsmooth: Spread of similarity function must be a positive scalar")

        J = cv2.bilateralFilter(I.astype(np.float32), d=-1, sigmaColor=sigma_r * 255, sigmaSpace=sigma_d)

    else:
        raise ValueError(f"imsmooth: Unsupported smoothing type '{name}'")

    return J
