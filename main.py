import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from is_image import is_image
from im_cast import imcast
from apply_padding import padarray
from pad_for_sliding_filter import pad_for_sliding_filter
from kernel import fspecial
from im_smooth import imsmooth
from im_noise import im_noise
from variational_denoiser import variational_denoiser
from perona_malik import perona_malik

#  Step 1: Read and convert image to grayscale
I = cv2.imread("input_images/img6.png")
is_image(I)
#I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

#  Step 2: Add Gaussian noise
A = im_noise(I, "gaussian", 0.2, 0.1)  # Adds noise with mean=0.2, variance=0.1

#  Step 3: Apply Padding before Filtering
if len(A.shape) == 3:  # If the image is in RGB/BGR format
    A = np.mean(A, axis=2).astype(np.uint8) #converted to greyscale
window_size = (5, 5) if A.ndim == 2 else (3, 3, A.shape[2])
A_padded = pad_for_sliding_filter(A, window_size, padval=0)  # Pad image before filtering

#  Step 4: Apply Pre-Smoothing Filters (before PDE)
A_smooth = imsmooth(A_padded, "gaussian", 1.5)  # Prepares image for PDE denoising

#  Step 5: Apply PDE-Based Denoising Methods
# Variational Denoiser
B = variational_denoiser(A_smooth, 150, 3, 0.05, 0.999, 0.99, 0.01, 0.00005, 50)
C = perona_malik(A_smooth, iterations=20, K=10, lambda_=0.05) ## Perona-Malik Denoiser

#  Step 6: Apply Other Denoising Techniques for Comparison
D = imsmooth(A_smooth, "average", (3, 3))  # Average Filter
E = imsmooth(A_smooth, "median", (3, 3)) # Median Filter
F = imsmooth(A_smooth, "bilateral", 2, 10/255)  # Bilateral Filter


results_folder = "results"
os.makedirs(results_folder, exist_ok=True)

#  Step 7: Display Results
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(I, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 3, 2)
plt.imshow(A, cmap='gray')
plt.title('Noised Image')

plt.subplot(2, 3, 3)
plt.imshow(B, cmap='gray')
plt.title('Variational Denoiser')

plt.subplot(2, 3, 4)
plt.imshow(C, cmap='gray')
plt.title('Perona-Malik Denoiser')

plt.subplot(2, 3, 5)
plt.imshow(D, cmap='gray')
plt.title('Average Filter Denoiser')

plt.subplot(2, 3, 6)
plt.imshow(E, cmap='gray')
plt.title('Median Filter Denoiser')

# Save the figure in the "results" folder
save_path = os.path.join(results_folder, "denoising_results_img6.jpg")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

# Step 8: Compute Noise Estimation Error (NE)
# Ensure I is grayscale before computing noise errors
if len(I.shape) == 3:
    I = np.mean(I, axis=2).astype(np.uint8)  # Convert RGB to grayscale

I = I.astype(np.float64)
A = A.astype(np.float64)
B = B.astype(np.float64)
C = C.astype(np.float64)
D = D.astype(np.float64)
E = E.astype(np.float64)
F = F.astype(np.float64)


I /= 255.0
A /= 255.0
B /= 255.0
C /= 255.0
D /= 255.0
E /= 255.0
F /= 255.0
# B = np.squeeze(B)  # Removes extra singleton dimensions
B = cv2.resize(B, (I.shape[1], I.shape[0]))
C = cv2.resize(C, (I.shape[1], I.shape[0]))
D = cv2.resize(D, (I.shape[1], I.shape[0]))
E = cv2.resize(E, (I.shape[1], I.shape[0]))
F = cv2.resize(F, (I.shape[1], I.shape[0]))

I = I.astype(np.float64) / 255.0
A = A.astype(np.float64) / 255.0
B = B.astype(np.float64) / 255.0
C = C.astype(np.float64) / 255.0
D = D.astype(np.float64) / 255.0
E = E.astype(np.float64) / 255.0
F = F.astype(np.float64) / 255.0

# Compute noise estimation error (NE)
NE1 = np.sqrt(np.sum((A - I) ** 2))
NE2 = np.sqrt(np.sum((B - I) ** 2))
NE3 = np.sqrt(np.sum((C - I) ** 2))
NE4 = np.sqrt(np.sum((D - I) ** 2))
NE5 = np.sqrt(np.sum((E - I) ** 2))
NE6 = np.sqrt(np.sum((F - I) ** 2))


#  Step 9: Print Errors
print(f"NE1 (Noised): {NE1}")
print(f"NE2 (Variational Denoiser): {NE2}")
print(f"NE3 (Perona & Malik Denoiser): {NE3}")
print(f"NE4 (Average Filter): {NE4}")
print(f"NE5 (Median Filter): {NE5}")
print(f"NE6 (Bilateral Filter): {NE6}")
