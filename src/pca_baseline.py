import os
import glob
import math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

# Define PCA function
def pca(X):
    """
    Principal Component Analysis
    input: X, matrix with trainnig data stored as flattened arrays in rows
    return: projection matrix (with important dimensions first), variance and mean.

    SVD factorization:  A = U * Sigma * V.T
                        A.T * A = V * Sigma^2 * V.T  (V is eigenvectors of A.T*A)
                        A * A.T = U * Sigma^2 * U.T  (U is eigenvectors of A * A.T)
                        A.T * U = V * Sigma

    """

    # get matrix dimensions
    print("X shape", X.shape)
    num_data, dim = X.shape

    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim > num_data:
        # PCA compact trick
        M = np.dot(X, X.T)  # covariance matrix
        e, U = np.linalg.eigh(M)  # calculate eigenvalues an deigenvectors
        tmp = np.dot(X.T, U).T
        V = tmp[::-1]  # reverse since the last eigenvectors are the ones we want
        S = np.sqrt(e)[::-1]  # reverse since the last eigenvalues are in increasing order
        for i in range(V.shape[1]):
            V[:, i] /= S
    else: 
        # normal PCA, SVD method
        U, S, V = np.linalg.svd(X)
        V = V[:num_data]  # only makes sense to return the first num_data

    return V, S, mean_X

def run_grayscale(input_files):
    ## Display first 7 images
    fig_org = plt.figure()

    tmp_img = np.array(Image.open(input_files[0]).convert('L'))
    #tmp_img = np.array(Image.open(input_files[0]))
    m, n = tmp_img.shape

    # load images into matrix
    #lwf_immatrix = np.array([np.array(Image.open(im)).flatten()
    #                         for im in input_files])
    lwf_immatrix = np.array([np.array(Image.open(im).convert('L')).flatten()
                             for im in input_files])

    #print(lwf_immatrix[0])
    print(lwf_immatrix[0].shape)

    for i in range(len(lwf_immatrix)):
        plt.subplot(10, 10, i+1)
        #plt.imshow(lwf_immatrix[i].reshape(m, n, k).astype(np.uint8))
        plt.imshow(lwf_immatrix[i].reshape(m, n))
        plt.axis('off')

    # PCA on face images
    bV, bS, bimmean = pca(lwf_immatrix)

    # Show Results
    # First one is the mean image
    # Rest 11 are the top 11 features extracted for pictures of Bill Clinton

    fig = plt.figure()
    plt.gray()

    plt.subplot(3, 4, 1)
    plt.imshow(bimmean.reshape(m, n))
    plt.axis('off')

    for i in range(8):
        plt.subplot(3, 3, i+2)
        #plt.imshow(bV[i].reshape(m, n))
        plt.imshow(bV[i].reshape(m, n))
        #result = Image.fromarray(bV[i].reshape(m, n)).convert("L")
        #result.save("result_%d.jpg" % i)
        scipy.misc.imsave("stella_result_%d.jpg" % i, bV[i].reshape(m, n))
        plt.axis('off')

    plt.show()


def run_rgb(input_files):
    ## Display first 7 images
    fig_org = plt.figure()

    tmp_img = np.array(Image.open(input_files[0]))
    #tmp_img = np.array(Image.open(input_files[0]))
    m, n, k = tmp_img.shape

    # load images into matrix
    #lwf_immatrix = np.array([np.array(Image.open(im)).flatten()
    #                         for im in input_files])
    lwf_immatrix = np.array([np.array(Image.open(im)) for im in input_files])

    print(lwf_immatrix[0].shape)
    #print(lwf_immatrix[1].shape)

    for i in range(len(lwf_immatrix)):
        plt.subplot(10, 10, i+1)
        #plt.imshow(lwf_immatrix[i].reshape(m, n, k).astype(np.uint8))
        plt.imshow(lwf_immatrix[i])
        plt.axis('off')

    print(lwf_immatrix[0][:, :, 0].shape)

    r_matrix = np.array([i[:, :, 0].flatten() for i in lwf_immatrix])
    g_matrix = np.array([i[:, :, 1].flatten() for i in lwf_immatrix])
    b_matrix = np.array([i[:, :, 2].flatten() for i in lwf_immatrix])


    # PCA on face images
    bV_r, bS_r, bimmean_r = pca(r_matrix)
    bV_g, bS_g, bimmean_g = pca(g_matrix)
    bV_b, bS_b, bimmean_b = pca(b_matrix)


    # Show Results
    # First one is the mean image
    # Rest 11 are the top 11 features extracted for pictures of Bill Clinton

    fig = plt.figure()

    """
    pca_results = [(bV_r, bS_r, bimmean_r), (bV_g, bS_g, bimmean_g), (bV_b, bS_b, bimmean_b)]

    for c, result in enumerate(pca_results):
        for i in range(8):
            bV = result[0]
            scipy.misc.imsave("stella-jean_%d_result_%d.jpg" % (c, i), bV[i].reshape(m, n))
    """

    plt.subplot(3, 4, 1)
    rgb = np.dstack((bimmean_r.reshape(m, n), bimmean_g.reshape(m, n), bimmean_b.reshape(m, n)))
    plt.imshow(rgb)
    plt.axis('off')

    for i in range(8):
        plt.subplot(3, 3, i+2)
        #plt.imshow(bV[i].reshape(m, n))
        rgb = np.dstack((bV_r[i].reshape(m, n), bV_g[i].reshape(m, n), bV_b[i].reshape(m, n)))
        plt.imshow(rgb)
        scipy.misc.imsave("chanel_rgb_result_%d.jpg" % i, rgb)
        plt.axis('off')


def run_rgb_2(input_files):
    ## Display first 7 images
    fig_org = plt.figure()

    tmp_img = np.array(Image.open(input_files[0]))
    #tmp_img = np.array(Image.open(input_files[0]))
    m, n, k = tmp_img.shape

    # load images into matrix
    #lwf_immatrix = np.array([np.array(Image.open(im)).flatten()
    #                         for im in input_files])
    lwf_immatrix = np.array([np.array(Image.open(im)).flatten() for im in input_files])

    print(lwf_immatrix[0].shape)

    for i in range(len(lwf_immatrix)):
        plt.subplot(10, 10, i+1)
        plt.imshow(lwf_immatrix[i].reshape(m, n, k))
        plt.axis('off')

    bV, bS, bimmean = pca(lwf_immatrix)

    for i in range(8):
        plt.subplot(3, 3, i+2)
        #plt.imshow(bV[i].reshape(m, n))
        plt.imshow(bV[i].reshape(m, n, k))
        #result = Image.fromarray(bV[i].reshape(m, n)).convert("L")
        #result.save("result_%d.jpg" % i)
        scipy.misc.imsave("stella-jean_rgb_result_%d.jpg" % i, bV[i].reshape(m, n, k))
        plt.axis('off')




if __name__ == "__main__":
    ## List of Bill Clinton Face Images
    src_dir = "../img/stella-jean"
    #src_dir = "../img/chanel"
    input_files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith(".jpg")]
    run_rgb_2(input_files)
