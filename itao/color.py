import numpy as np
import cv2
from sklearn.cluster import KMeans


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


def image_color_cluster(image, k=3):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    clt = KMeans(n_clusters=k)
    clt.fit(image)
    hist = centroid_histogram(clt)

    res = []
    for h, c in zip(hist, clt.cluster_centers_):
        res.append([h, c])

    res.sort()

    bgColor = ''.join([hex(int(x))[2:] for x in res[-1][1]])
    fColor = ''.join([hex(int(x))[2:] for x in res[-2][1]])

    rq = {"backgroundColor": bgColor, "fontColor": fColor}
    return rq
