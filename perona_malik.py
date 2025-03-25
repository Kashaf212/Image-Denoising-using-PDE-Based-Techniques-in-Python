import numpy as np


def perona_malik(I, iterations=20, lambda_=0.05, K=15, option=1):
    """
    Applies Perona-Malik anisotropic diffusion for noise removal.

    Parameters:
    - I: Input noisy image (grayscale).
    - iterations: Number of iterations.
    - lambda_: Controls speed of diffusion (0 < lambda_ < 0.25 for stability).
    - K: Gradient threshold (higher keeps more edges).
    - option: 1 (exponential) or 2 (quadratic) diffusion function.

    Returns:
    - Denoised image.
    """
    I = I.astype(np.float64)
    if len(I.shape) == 3:  # If the image has three dimensions (RGB)
        X, Y, Z = I.shape
    else:
        X, Y = I.shape
    I_new = I.copy()

    for _ in range(iterations):
        # Compute image gradients
        north = np.roll(I_new, -1, axis=0) - I_new
        south = np.roll(I_new, 1, axis=0) - I_new
        east = np.roll(I_new, -1, axis=1) - I_new
        west = np.roll(I_new, 1, axis=1) - I_new

        # Perona-Malik edge-stopping function
        if option == 1:  # Exponential function
            cN = np.exp(-(north / K) ** 2)
            cS = np.exp(-(south / K) ** 2)
            cE = np.exp(-(east / K) ** 2)
            cW = np.exp(-(west / K) ** 2)
        else:  # Quadratic function
            cN = 1 / (1 + (north / K) ** 2)
            cS = 1 / (1 + (south / K) ** 2)
            cE = 1 / (1 + (east / K) ** 2)
            cW = 1 / (1 + (west / K) ** 2)

        # Apply diffusion process
        I_new += lambda_ * (cN * north + cS * south + cE * east + cW * west)

    return np.uint8(I_new)
