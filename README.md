This project focuses on denoising grayscale images using Partial Differential Equation (PDE) based techniques, particularly the Variational Model and Perona-Malik Anisotropic Diffusion Model. The project also compares these methods with traditional denoising filters such as Average, Median, and Bilateral Filters.

📌 Objective

To remove noise from images using PDE-based methods while preserving important image details like edges and textures.

📁 Project Structure

<pre> Denoising-PDEs/
├── main.py                     # Main driver script
├── imnoise.py                 # Adds synthetic noise (Gaussian, Salt & Pepper, etc.)
├── imcast.py                  # Handles image data type conversions
├── imsmooth.py                # Applies pre-smoothing filters
├── pad_for_sliding_filter.py # Pads images before filtering
├── variational_denoiser.py   # Variational PDE denoising
├── perona_malik.py           # Perona-Malik anisotropic diffusion denoising
├── fspecial.py               # Custom kernels (Gaussian, Laplacian, etc.)
├── result/                   # Folder to save results </pre>

🧠 Key Concepts

Variational Model: Solves an energy minimization problem to reduce noise.

Perona-Malik Model: Performs edge-preserving smoothing using anisotropic diffusion.

Pre-Smoothing: Gaussian filter applied before PDEs to make them more stable.

✅ Features

Support for different noise types: Gaussian, Salt & Pepper, Speckle, Poisson

Flexible pre-smoothing and padding strategies

Visual and numerical comparison of denoising results

Calculates Noise Estimation Error (NE) for each method

🔧 How to Run

1. Install dependencies:

<pre> pip install numpy opencv-python matplotlib scipy scikit-image </pre>

2. Add your image (e.g., image.jpg or image.png) to the project folder.
  
3. Run the main script:

<pre> python main.py </pre>

4. Denoised results and comparison plots will be saved in the result/ folder.

📊 Output

The following denoising techniques are compared:

Variational PDE

Perona-Malik PDE

Average Filter

Median Filter

Bilateral Filter

Each method's performance is compared using:

Visual Results

Noise Estimation Error (NE)

📈 Sample Results:

NE1 (Noised): 0.2137

NE2 (Variational Denoiser): 0.1925

NE3 (Perona-Malik Denoiser): 0.1915

NE4 (Average Filter): 0.1924

NE5 (Median Filter): 0.1927

NE6 (Bilateral Filter): 0.1926

👤 Author

Kashaf

BS Mathematics Student

Passionate about Image Processing, Computer Vision and AI Research

⭐ Acknowledgements

Inspired by research from:

Springer VCIBA Paper on Variational and Perona-Malik Denoising

🧠 Future Work

Extend to color images

Implement PSNR and SSIM metrics

Add more PDE models (Total Variation, ROF, etc.)

Deploy as a web app using Streamlit

Feel free to fork or star the project! Contributions are welcome.
