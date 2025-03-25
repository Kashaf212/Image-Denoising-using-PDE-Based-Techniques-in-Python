### This project focuses on denoising grayscale images using Partial Differential Equation (PDE) based techniques, particularly the Variational Model and Perona-Malik Anisotropic Diffusion Model. The project also compares these methods with traditional denoising filters such as Average, Median, and Bilateral Filters.

## Objective

To remove noise from images using PDE-based methods while preserving important image details like edges and textures.

## Project Structure

<pre> Denoising-PDEs/
├── main.py                      # Main driver script
├── is_image.py                  # checks if the image is valid
├── im_noise.py                  # Adds synthetic noise (Gaussian, Salt & Pepper, etc.)
├── im_cast.py                   # Handles image data type conversions
├── im_smooth.py                 # Applies pre-smoothing filters
├── apply_padding.py             # Handles pre-filter padding
├── pad_for_sliding_filter.py    # Pads images before filtering
├── kernel.py                    # Defines custom filter kernels (Gaussian, Laplacian, etc.)     
├── variational_denoiser.py      # Variational PDE denoising
├── perona_malik.py              # Perona-Malik anisotropic diffusion denoising
├── result/                      # Folder to save results </pre>

## Key Concepts

**Variational Model**: Solves an energy minimization problem to reduce noise.  
**Perona-Malik Model**: Performs edge-preserving smoothing using anisotropic diffusion.  
**Pre-Smoothing**: Gaussian filter applied before PDEs to make them more stable.

## Features

- Support for different noise types: Gaussian, Salt & Pepper, Speckle, Poisson  
- Flexible pre-smoothing and padding strategies  
- Visual and numerical comparison of denoising results  
- Calculates Noise Estimation Error (NE) for each method

## How to Run

1. **Install dependencies**:  
   ```bash
   pip install numpy opencv-python matplotlib scipy scikit-image
2. Add your image (e.g., image.jpg or image.png) to the project folder.
3. Run the main script:
   ```bash
   python main.py
4. Denoised results and comparison plots will be saved in the result/ folder.

## Output

The following denoising techniques are compared:

- Variational PDE  
- Perona-Malik PDE  
- Average Filter  
- Median Filter  
- Bilateral Filter  

Each method's performance is compared using:

- Visual Results  
- Noise Estimation Error (NE)


## Sample Results:

NE1 (Noised): 0.2137  
NE2 (Variational Denoiser): 0.1925  
NE3 (Perona-Malik Denoiser): 0.1915  
NE4 (Average Filter): 0.1924  
NE5 (Median Filter): 0.1927  
NE6 (Bilateral Filter): 0.1926

## About Me

**Kashaf Jamil**  
Department of Mathematics  
University of Gujrat  
Roll No: 18541509-085  
**Research Interests:** Image Processing, Computer Vision, Artificial Intelligence

## Acknowledgements

Inspired by research from:

 [Springer VCIBA Paper on Variational and Perona-Malik Denoising](https://vciba.springeropen.com/articles/10.1186/s42492-019-0016-7)

## Future Work

- Extend to color images  
- Implement PSNR and SSIM metrics  
- Add more PDE models (Total Variation, ROF, etc.)  
- Deploy as a web app using Streamlit

### Feel free to fork or star the project! Contributions are welcome.
