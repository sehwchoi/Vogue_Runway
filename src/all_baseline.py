import os
import glob
import math
from PIL import Image
from sklearn.cluster import KMeans

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

    return U, V, S, mean_X


def run_clustering(input_files):
    ## Display first 7 images
    fig_org = plt.figure()

    tmp_img = np.array(Image.open(input_files[0]))
    #tmp_img = np.array(Image.open(input_files[0]))
    m, n, k = tmp_img.shape

    # load images into matrix
    #lwf_immatrix = np.array([np.array(Image.open(im)).flatten()
    #                         for im in input_files])
    lwf_immatrix = []
    for im in input_files:
        im_mtx = np.array(Image.open(im))
        r, g, b = im_mtx[:, :, 0], im_mtx[:, :, 1], im_mtx[:, :, 2]
        r = [str(s) + 'r' for s in r.flatten()]
        g = [str(s) + 'g' for s in g.flatten()]
        b = [str(s) + 'b' for s in b.flatten()]
        im_mtx = []
        im_mtx.extend(r)
        im_mtx.extend(g)
        im_mtx.extend(b)

        lwf_immatrix.append(lwf_immatrix)

    lwf_immatrix = np.array(lwf_immatrix)
    print(lwf_immatrix[0].shape)

    for i in range(len(lwf_immatrix)):
        plt.subplot(10, 10, i+1)
        plt.imshow(lwf_immatrix[i].reshape(m, n, k))
        plt.axis('off')

    bU, bV, bS, bimmean = pca(lwf_immatrix)

    print(bU.shape)
    print(bU)

    kmeans = KMeans(n_clusters=8, random_state=0).fit(bU)
    print(kmeans)
    labels = kmeans.labels_
    print("labels", labels)
    clusters = {}
    for idx, item in enumerate(labels):
        if item in clusters:
            clusters[item].append(lwf_immatrix[idx])
        else:
            clusters[item] = [lwf_immatrix[idx]]

    print("cluster", clusters)

    for cluster in clusters.keys():
        for i, data in enumerate(clusters[cluster]):
            plt.imshow(data.reshape(m, n, k))
            #result = Image.fromarray(bV[i].reshape(m, n)).convert("L")
            #result.save("result_%d.jpg" % i)
            scipy.misc.imsave("results/stella_kmean_%d_%d.jpg" % (cluster+1, i), data.reshape(m, n, k))
            plt.axis('off')


def run_pca(input_files, show_name):
    ## Display first 7 images
    fig_org = plt.figure()

    tmp_img = np.array(Image.open(input_files[0]))
    m, n, k = tmp_img.shape
    print("shape", tmp_img.shape)

    # load images into matrix
    #lwf_immatrix = np.array([np.array(Image.open(im)).flatten()
    #                         for im in input_files])
    lwf_immatrix_r = np.array([np.array(Image.open(im))[:, :, 0].flatten() for im in input_files])
    lwf_immatrix_g = np.array([np.array(Image.open(im))[:, :, 1].flatten() for im in input_files])
    lwf_immatrix_b = np.array([np.array(Image.open(im))[:, :, 2].flatten() for im in input_files])

    matrices = [('r', lwf_immatrix_r), ('g', lwf_immatrix_g), ('b', lwf_immatrix_b)]
    results = {}
    for item in matrices:
        rgb_type, lwf_immatrix = item[0], item[1]

        print("Shape of data", lwf_immatrix[0].shape)
        print(lwf_immatrix)

        for i in range(len(lwf_immatrix)):
            plt.subplot(10, 10, i+1)
            plt.imshow(lwf_immatrix[i].reshape(m, n))
            plt.axis('off')

        bU, bV, bS, bimmean = pca(lwf_immatrix)
        results[rgb_type] = (bU, bV, bS, bimmean)

        fig = plt.figure()
        plt.gray()

        plt.subplot(3, 4, 1)
        plt.imshow(bimmean.reshape(m, n))
        #scipy.misc.imsave("results/%s_pca_mean_%s_%d.jpg" % (show_name, rgb_type, i), bimmean.reshape(m, n))
        plt.axis('off')

        for i in range(8):
            plt.subplot(3, 3, i+2)
            plt.imshow(bV[i].reshape(m, n))
            #scipy.misc.imsave("results/%s_pca_%s_%d.jpg" % (show_name, rgb_type, i), bV[i].reshape(m, n))
            plt.axis('off')

        plt.show()


    for i in range(8):
        bV_r = results['r'][1][i].reshape(m, n)
        bV_g = results['g'][1][i].reshape(m, n)
        bV_b = results['b'][1][i].reshape(m, n)
        merged = np.dstack((bV_r, bV_g, bV_b))
        plt.subplot(3, 3, i + 2)
        plt.imshow(merged)
        scipy.misc.imsave("results/%s_pca_merge_%d.jpg" % (show_name, i), merged)
        plt.axis('off')

    plt.show()

def run_pca_all(input_file, show_name):
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

    fig = plt.figure()
    plt.gray()

    plt.subplot(3, 4, 1)
    plt.imshow(bimmean.reshape(m, n, k))
    scipy.misc.imsave("results/%s_pca_all_mean.jpg" % show_name, bimmean.reshape(m, n, l))
    plt.axis('off')

    for i in range(8):
        plt.subplot(3, 3, i+2)
        #plt.imshow(bV[i].reshape(m, n))
        plt.imshow(bV[i].reshape(m, n, k))
        #result = Image.fromarray(bV[i].reshape(m, n)).convert("L")
        #result.save("result_%d.jpg" % i)
        scipy.misc.imsave("results/%s_pca_all_%d.jpg" % (show_name, i), bV[i].reshape(m, n, k))
        plt.axis('off')


if __name__ == "__main__":
    ## List of Bill Clinton Face Images
    show_name = "stella-jean"
    src_dir = "../img/%s" % show_name
    #src_dir = "../img/chanel"
    input_files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith(".jpg")]
    run_pca(input_files, show_name)
