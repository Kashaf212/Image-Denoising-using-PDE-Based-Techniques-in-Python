import numpy as np
import cv2

def fspecial(filter_type, arg1=None, arg2=None):
    """
    Creates predefined 2D filters similar to MATLAB's fspecial.

    Parameters:
    - filter_type (str): Type of filter ('average', 'disk', 'gaussian', 'laplacian', 'log', 'motion', 'prewitt', 'sobel', 'kirsch', 'unsharp').
    - arg1 (various): Size, radius, sigma, or other parameter depending on the filter type.
    - arg2 (various, optional): Additional parameter depending on the filter type.

    Returns:
    - numpy.ndarray: The generated filter.
    """

    filter_type = filter_type.lower()

    # Average Filter
    if filter_type == "average":
        if arg1 is None:
            size = (3, 3)  # Default size
        else:
            size = tuple(arg1) if isinstance(arg1, (list, tuple)) else (arg1, arg1)
        return np.ones(size) / np.prod(size)

    # Disk Filter
    elif filter_type == "disk":
        if arg1 is None:
            r = 5  # Default radius
        else:
            r = int(arg1)
        y, x = np.ogrid[-r:r+1, -r:r+1]
        mask = x**2 + y**2 <= r**2
        kernel = np.zeros((2*r+1, 2*r+1))
        kernel[mask] = 1
        return kernel / np.sum(kernel)

    # Gaussian Filter
    elif filter_type == "gaussian":
        if arg1 is None:
            size = (3, 3)
        else:
            size = tuple(arg1) if isinstance(arg1, (list, tuple)) else (arg1, arg1)
        sigma = arg2 if arg2 else 0.5
        return cv2.getGaussianKernel(size[0], sigma) @ cv2.getGaussianKernel(size[1], sigma).T

    # Laplacian Filter
    elif filter_type == "laplacian":
        alpha = arg1 if arg1 else 0.2
        f = np.array([[alpha/4, (1-alpha)/4, alpha/4],
                      [(1-alpha)/4, -1, (1-alpha)/4],
                      [alpha/4, (1-alpha)/4, alpha/4]])
        return f

    # Laplacian of Gaussian (LoG) Filter
    elif filter_type == "log":
        size = (5, 5) if arg1 is None else (arg1, arg1)
        sigma = arg2 if arg2 else 0.5
        hsize = [s-1 for s in size]
        x, y = np.meshgrid(np.arange(-hsize[0]//2, hsize[0]//2+1),
                           np.arange(-hsize[1]//2, hsize[1]//2+1))
        gauss = np.exp(-(x**2 + y**2) / (2*sigma**2))
        f = ((x**2 + y**2 - 2*sigma**2) * gauss) / (2*np.pi*sigma**6 * np.sum(gauss))
        return f

    # Motion Blur Filter
    elif filter_type == "motion":
        length = arg1 if arg1 else 9
        angle = arg2 if arg2 else 0
        f = np.zeros((length, length))
        f[length//2, :] = 1
        f = f / np.sum(f)
        return f

    # Prewitt Filter
    elif filter_type == "prewitt":
        return np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    # Sobel Filter
    elif filter_type == "sobel":
        return np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Kirsch Filter
    elif filter_type == "kirsch":
        return np.array([[3, 3, 3], [3, 0, -3], [-5, -5, -5]])

    # Unsharp Masking Filter
    elif filter_type == "unsharp":
        alpha = arg1 if arg1 else 0.2
        f = (1/(alpha+1)) * np.array([[-alpha, -1, -alpha],
                                      [-1, alpha+5, -1],
                                      [-alpha, -1, -alpha]])
        return f

    else:
        raise ValueError(f"fspecial: Filter type '{filter_type}' is not supported.")


