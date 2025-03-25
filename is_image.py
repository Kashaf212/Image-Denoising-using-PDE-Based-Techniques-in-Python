import numpy as np
import matplotlib.pyplot as plt

def is_image(img):
    """
    Checks if the input is a valid image.

    Conditions:
    - Must be a NumPy array
    - Must be numeric (integer or float) or boolean (logical)
    - Must not be sparse
    - Must not be empty
    - Must not contain complex numbers
    """
    if isinstance(img, np.ndarray):  # Check if input is a NumPy array
        if img.dtype.kind in {'i', 'u', 'f', 'b'}:  # Integer, Unsigned int, Float, or Boolean
            if img.size > 0 and np.isrealobj(img):  # Not empty & contains real numbers
                return True
    return False


# import cv2
#
# # Load an image
# img = cv2.imread('')
# print(is_image(img))  # Should return True
# print(img)
# (plt.imshow(img))
# plt.show()