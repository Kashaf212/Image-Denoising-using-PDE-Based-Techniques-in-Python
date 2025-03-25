import numpy as np

def padarray(A, padsize, padval=0, direction="both", mode="constant"):
    """
    Pads an array with specified padding size and value.

    Parameters:
    - A (numpy.ndarray): Input array.
    - padsize (tuple or list): Number of rows/columns to pad in each dimension.
    - padval (int, float, or str): Padding value or method ('circular', 'replicate', 'reflect', 'symmetric').
    - direction (str): 'pre', 'post', or 'both' (default: 'both').
    - mode (str): Padding mode for numpy.pad (default: 'constant').

    Returns:
    - numpy.ndarray: Padded array.
    """

    # Convert padsize to a tuple for NumPy
    if isinstance(padsize, int):
        padsize = (padsize, padsize)
    elif isinstance(padsize, list):
        padsize = tuple(padsize)

    # Convert direction to NumPy pad width format
    if direction == "both":
        pad_width = [(p, p) for p in padsize]
    elif direction == "pre":
        pad_width = [(p, 0) for p in padsize]
    elif direction == "post":
        pad_width = [(0, p) for p in padsize]
    else:
        raise ValueError("Invalid direction. Choose from 'pre', 'post', 'both'.")

    # Handle different padding modes
    if isinstance(padval, str):  # If padval is a mode (e.g., 'replicate', 'symmetric')
        if padval == "circular":
            mode = "wrap"
        elif padval == "replicate":
            mode = "edge"
        elif padval == "reflect":
            mode = "reflect"
        elif padval == "symmetric":
            mode = "symmetric"
        else:
            raise ValueError(f"Unknown padding mode: {padval}")

        return np.pad(A, pad_width, mode=mode)

    # Default: Constant padding with specified value
    return np.pad(A, pad_width, mode="constant", constant_values=padval)
