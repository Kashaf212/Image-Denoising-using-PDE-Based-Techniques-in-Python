import numpy as np
import cv2
import matplotlib.pyplot as plt
def imcast(img, outcls, indexed=False):
    """
    Convert an image to a specified data type with appropriate scaling.

    Parameters:
    - img: numpy.ndarray
        Input image.
    - outcls: str
        Desired output data type (e.g., 'uint8', 'uint16', 'double', 'logical').
    - indexed: bool, optional
        If True, treats the image as an indexed image.

    Returns:
    - numpy.ndarray
        Converted image.

    Raises:
    - TypeError: If an unsupported data type is requested.
    """

    dtype_map = {
        "uint8": np.uint8,
        "uint16": np.uint16,
        "int16": np.int16,
        "double": np.float64,
        "single": np.float32,
        "logical": np.bool_
    }

    if outcls not in dtype_map:
        raise TypeError(f"imcast: Unsupported TYPE '{outcls}'")

    # Get input class
    incls = str(img.dtype)

    # If already in the correct type, return as is
    if incls == outcls:
        return img

    # Indexed image conversion
    if indexed:
        if not np.issubdtype(img.dtype, np.integer):
            raise TypeError("imcast: Input should be an indexed image but it is not.")

        # Check if max index fits in the target integer type
        max_val = np.max(img)
        if outcls in ["uint8", "uint16", "int16"] and max_val > np.iinfo(dtype_map[outcls]).max:
            raise ValueError(f"imcast: IMG has too many colours '{max_val}' for the range of values in '{outcls}'")

        return img.astype(dtype_map[outcls])

    # Normal image conversions
    if incls in ["float64", "float32"]:  # Floating point to integer
        if outcls in ["uint8", "uint16", "int16"]:
            scale_factor = {
                "uint8": 255,
                "uint16": 65535,
                "int16": 32767
            }[outcls]
            img = np.clip(img * scale_factor, 0, scale_factor)  # Scale and clip
            return img.astype(dtype_map[outcls])

    elif incls == "uint8":  # Integer to floating point or logical
        if outcls in ["double", "single"]:
            return img.astype(dtype_map[outcls]) / 255.0
        elif outcls == "logical":
            return img > 0

    elif incls == "uint16":
        if outcls in ["double", "single"]:
            return img.astype(dtype_map[outcls]) / 65535.0
        elif outcls == "uint8":
            return (img / 257).astype(np.uint8)  # 65535/255 = 257
        elif outcls == "logical":
            return img > 0

    elif incls == "int16":
        if outcls in ["double", "single"]:
            return (img + 32768) / 65535.0  # Scale from [-32768, 32767] to [0, 1]
        elif outcls == "uint8":
            return ((img + 32768) / 257).astype(np.uint8)
        elif outcls == "uint16":
            return (img + 32768).astype(np.uint16)
        elif outcls == "logical":
            return img > 0

    # Unknown image type
    raise TypeError(f"imcast: Unknown image of class '{incls}'")




# print(imcast(cv2.imread("Cat.jpg"),  "double"))
# img = cv2.imread("cat.jpg")  # Reads the image as a NumPy array
# print(img.dtype)
#
# (plt.imshow(img))
# plt.show()
