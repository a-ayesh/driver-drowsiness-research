import numpy as np

def corr(a):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            u = np.sum((a[:,i,j] - np.mean(a[:,i,j])) ** 2) / a[:,i,j].shape[0]
            z = a[:,i,j] - u
            outer = np.outer(z, z.T)
            result += outer
    return result

def region_covariance_descriptor(F, l=1):
    """Computes the region covariance descriptor of an image.

    Args:
        F: A 3D numpy array of shape (d, n, m), where:
        - d is the number of channels in the image.
        - n is the height of the image.
        - m is the width of the image.
        l: The half-side length of the square region used for computing the
        covariance descriptor.

    Returns:
        A 3D numpy array of shape (d, d, n - 2 * l, m - 2 * l) containing the
        region covariance descriptors.
    """

    d, n, m = F.shape

    # Integral image of each channel (with padding)
    padding = ((0, 0),(0, 1), (0, 1))  # Pad with one zero on each side
    F_padded = np.pad(F, padding, mode='constant')
    assert F_padded.shape[1] == 225 and F_padded.shape[2] == 225
    P = np.zeros((d, n + 1, m + 1), dtype=np.int64)
    for i in range(d):
        P[i, :, :] = np.cumsum(np.cumsum(F_padded[i, :, :].astype(np.int64), axis=0), axis=1)

    P = P[:, 1:, 1:]

    F2 = np.zeros((d, d, n, m), dtype=np.int64)
    for i in range(d):
        for j in range(d):
            F2[i, j, :, :] = (F[i, :, :] * F[j, :, :]).astype(np.int64)

    padding = ((0,0), (0,0), (0, 1), (0, 1))
    
    F2_padded = np.pad(F2, padding, mode='constant', constant_values=0)

    Q = np.zeros((d, d, n+1, m+1), dtype=np.int64)
    for i in range(d):
        for j in range(d):
            Q[i, j, :, :] = integral_image(F2_padded[i, j, :, :])
    Q = Q[:, :, 1:, 1:]

    Qshift = (np.roll(Q, (l, l), axis=(2, 3)) + np.roll(Q, (-l, -l), axis=(2, 3)) -
          np.roll(Q, (l, -l), axis=(2, 3)) - np.roll(Q, (-l, l), axis=(2, 3)))

    Pshift = (np.roll(P, l, axis=1) + np.roll(P, l, axis=2) -
            np.roll(P, -l, axis=1) - np.roll(P, -l, axis=2))

    PshiftRep = np.tile(Pshift[:, np.newaxis], (1, d, 1, 1))
    PshiftRepTrans = np.transpose(PshiftRep, (1, 0, 2, 3))
    Pmult = PshiftRep * PshiftRepTrans

    C = (1 / ((2 * l + 1) ** 2 - 1)) * (Qshift.astype(float) -
                                        (1 / ((2 * l + 1) ** 2)) * Pmult.astype(float))

    
    return C[:,:,l:-l,l:-l]

def integral_image(F):
    return np.cumsum(np.cumsum(F, axis=1), axis=0)



img = np.random.rand(3, 224, 224)

descriptor = region_covariance_descriptor(img, l=1)

print(descriptor)
