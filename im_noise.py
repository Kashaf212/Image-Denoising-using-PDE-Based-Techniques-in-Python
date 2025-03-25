import numpy as np

def im_noise(A, stype, a=None, b=None):
    """
    Adds different types of noise to an image.

    Parameters:
    - A (numpy.ndarray): Input image.
    - stype (str): Type of noise ('poisson', 'gaussian', 'salt & pepper', 'speckle').
    - a (float, optional): First parameter (varies by noise type).
    - b (float, optional): Second parameter (varies by noise type).

    Returns:
    - numpy.ndarray: Noisy image.
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("im_noise: First argument must be an image (numpy array).")
    if not isinstance(stype, str):
        raise TypeError("im_noise: Second argument must be a string representing noise type.")

    # Store original class
    in_class = A.dtype

    # Convert to float for processing
    A = A.astype(np.float64)

    if stype.lower() == "poisson":
        A = np.random.poisson(A).astype(np.float64)

    elif stype.lower() == "gaussian":
        A = A / 255.0  # Normalize if uint8
        if a is None: a = 0.0  # Mean
        if b is None: b = 0.01  # Variance
        A = A + np.random.normal(a, np.sqrt(b), A.shape)
        A = np.clip(A, 0, 1) * 255  # Scale back if uint8

    elif stype.lower() in ["salt & pepper", "salt and pepper"]:
        if a is None: a = 0.05  # Default noise density
        noise = np.random.rand(*A.shape)
        A[noise < (a / 2)] = 0   # Salt
        A[noise > 1 - (a / 2)] = 255  # Pepper

    elif stype.lower() == "speckle":
        A = A / 255.0  # Normalize
        if a is None: a = 0.04  # Default variance
        A = A * (1 + np.random.normal(0, np.sqrt(a), A.shape))
        A = np.clip(A, 0, 1) * 255  # Scale back

    else:
        raise ValueError(f"im_noise: Unknown noise type '{stype}'.")

    return A.astype(in_class)  # Convert back to original type


                        ###### TO UNDERSTAND HOW THIS CODE WILL WORK ########


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import is_image
#
# # Load original image
# img = cv2.imread("cat_original.png", cv2.IMREAD_GRAYSCALE)
#
# # Check if it's a valid image
# #print(is_image(img))  # Should return True
#
# # Add different types of noise
# noisy_gaussian = im_noise(img, "gaussian", 0, 0.01)
# print(noisy_gaussian)
# noisy_salt_pepper = im_noise(img, "salt & pepper", 0.05)

# Display Original and Noisy Images
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#
# axes[0].imshow(img, cmap="gray")
# axes[0].set_title("Original Image")
# axes[0].axis("off")
#
# axes[1].imshow(noisy_gaussian, cmap="gray")
# axes[1].set_title("Gaussian Noise Added")
# axes[1].axis("off")
#
# axes[2].imshow(noisy_salt_pepper, cmap="gray")
# axes[2].set_title("Salt & Pepper Noise Added")
# axes[2].axis("off")
#
# plt.show()
