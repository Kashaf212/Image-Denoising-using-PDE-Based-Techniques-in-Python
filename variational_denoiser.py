import numpy as np


def variational_denoiser(I, alpha, k, neta, beta, gamma, nu, lambda_, N):
    """
    Variational denoising function
    :param I: Input image (assumed 8-bit grayscale)
    :param alpha, k, neta, beta, gamma, nu, lambda_: Algorithm parameters
    :param N: Number of iterations
    :return: Denoised image
    """
    I = I.astype(np.uint8)  # Converting into 8-bit image
    u = I.astype(float)

    if len(u.shape) == 2:  # If grayscale, add a third dimension
        u = u[:, :, np.newaxis]  # Convert 2D to 3D (height, width, 1)

    X, Y, Z = u.shape  # Now safely get dimensions

    for _ in range(N):
        o = np.zeros((X, Y, Z))

        c1 = np.copy(u)
        c1[:-1, :, :] = u[1:, :, :]

        c2 = np.copy(u)
        c2[1:, :, :] = u[:-1, :, :]

        c3 = np.copy(u)
        c3[:, :-1, :] = u[:, 1:, :]

        c4 = np.copy(u)
        c4[:, 1:, :] = u[:, :-1, :]

        del1 = (np.sqrt(beta * (np.linalg.norm(c1.astype(float)) ** 2) + gamma) + neta * np.sqrt(k)) / \
               (np.sqrt(beta * (np.linalg.norm(c1.astype(float)) ** 2) + gamma))
        del2 = (np.sqrt(beta * (np.linalg.norm(c2.astype(float)) ** 2) + gamma) + neta * np.sqrt(k)) / \
               (np.sqrt(beta * (np.linalg.norm(c2.astype(float)) ** 2) + gamma))
        del3 = (np.sqrt(beta * (np.linalg.norm(c3.astype(float)) ** 2) + gamma) + neta * np.sqrt(k)) / \
               (np.sqrt(beta * (np.linalg.norm(c3.astype(float)) ** 2) + gamma))
        del4 = (np.sqrt(beta * (np.linalg.norm(c4.astype(float)) ** 2) + gamma) + neta * np.sqrt(k)) / \
               (np.sqrt(beta * (np.linalg.norm(c4.astype(float)) ** 2) + gamma))

        u = u + lambda_ * (del1 * c1 + del2 * c2 + del3 * c3 + del4 * c4) - (u - I[:, :, np.newaxis]) / alpha


    denoised_image = np.uint8(u)  # Converting back into 8-bit image
    return denoised_image
