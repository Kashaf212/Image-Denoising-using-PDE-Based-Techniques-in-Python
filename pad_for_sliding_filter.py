import numpy as np

def pad_for_sliding_filter(im, window_size, padval=0):
    """
    Pads an image for a sliding filter operation, ensuring odd-sized dimensions.

    Parameters:
    - im (numpy.ndarray): Input image.
    - window_size (tuple or list): Size of the filtering window.
    - padval (int, optional): Padding value (default: 0).

    Returns:
    - numpy.ndarray: Padded image with correct dimensions.
    """
    if im.ndim != len(window_size):
        raise ValueError("Image and window size must have the same number of dimensions")

    # Compute the required padding (floor of half window size)
    pad = tuple(np.floor(np.array(window_size) / 2).astype(int))

    # Apply padding using the given pad value
    im = np.pad(im, [(p, p) for p in pad], mode="constant", constant_values=padval)

    # Check for even-sized dimensions
    even = np.mod(window_size, 2) == 0  # True if dimension size is even

    # If any dimension is even, remove the extra padding from the first row/column
    if np.any(even):
        idx = [slice(e, None) for e in even]  # Skip the first row/column if needed
        im = im[tuple(idx)]

    return im
