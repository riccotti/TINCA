import json
import pickle
import datetime
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

path = './'


def get_knee_point_value(values):
    y = values
    x = np.arange(0, len(y))

    index = 0
    max_d = -float('infinity')

    for i in range(0, len(x)):
        c = closest_point_on_segment(a=[x[0], y[0]], b=[x[-1], y[-1]], p=[x[i], y[i]])
        d = np.sqrt((c[0] - x[i]) ** 2 + (c[1] - y[i]) ** 2)
        if d > max_d:
            max_d = d
            index = i

    return index


def closest_point_on_segment(a, b, p):
    sx1 = a[0]
    sx2 = b[0]
    sy1 = a[1]
    sy2 = b[1]
    px = p[0]
    py = p[1]

    x_delta = sx2 - sx1
    y_delta = sy2 - sy1

    if x_delta == 0 and y_delta == 0:
        return p

    u = ((px - sx1) * x_delta + (py - sy1) * y_delta) / (x_delta * x_delta + y_delta * y_delta)
    cp_x = sx1 + u * x_delta
    cp_y = sy1 + u * y_delta
    closest_point = [cp_x, cp_y]

    return closest_point


def main():

    min_k = 2
    max_k = 150 + 1

    df = pd.read_csv(path + 'tinca_immigrants_clustering_dataset.csv.gz')
    X = df[[c for c in df.columns if c not in ['customer_id', 'country']]].values

    sse_list = list()
    for k in range(min_k, max_k):
        print(datetime.datetime.now(), 'clustering %s' % k)
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
        kmeans.fit(X)
        sse_list.append(kmeans.inertia_)

    sse_file = open(path + 'sse_list.csv', 'w')
    for sse in sse_list:
        sse_file.write('%s\n' % sse)
    sse_file.close()

    n_clusters = get_knee_point_value(sse_list)
    print('n clusters', n_clusters)

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10)
    kmeans.fit(X)

    df['cluster_label'] = kmeans.labels_
    df[['customer_id', 'country', 'cluster_label']].to_csv(path + 'tinca_immigrants_clustering.csv.gz', index=False,
                                                           header=True, compression='gzip')


if __name__ == "__main__":
    main()
