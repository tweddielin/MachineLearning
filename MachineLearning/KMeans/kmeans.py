import numpy as np

np.random.seed(42)

def kmeans_plus2(x_train, n_clusters):
    c = x_train[np.random.choice(x_train.shape[0], 1, replace=False)]
    for i in range(1, n_clusters):
        dd = np.sqrt(((x_train - c[:, np.newaxis])**2).sum(axis=2)).T.min(axis=1)
        dd = dd**2
        p_dist = dd / dd.sum()
        new_c = x_train[np.random.choice(x_train.shape[0], 1, p=p_dist)]
        c = np.append(c, new_c, axis=0)
    return c

def random_centroids(x_train, n_clusters):
    sample = np.random.choice(x_train.shape[0], n_clusters, replace=False)
    return x_train[sample]

def assign_points(points, centroids):
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    distortion = np.amin(np.transpose(distances)**2, axis=1).sum()
    return np.argmin(distances, axis=0), distortion

def update_centroids(points, assigned_points, n_clusters):
    return np.array([points[assigned_points==k].mean(axis=0) for k in range(n_clusters)])

class KMeans(object):
    def __init__(self, n_clusters=5, init_method='random'):
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.centroids = None
        self.distortion = []

    def fit(self,x_train):
        if isinstance(self.init_method, basestring) and self.init_method == 'random':
            self.centroids = random_centroids(x_train, self.n_clusters)

        if isinstance(self.init_method, basestring) and self.init_method == 'kmeans++':
            self.centroids = kmeans_plus2(x_train, self.n_clusters)

        while True:
            update  = assign_points(x_train, self.centroids)
            new_centroids = update_centroids(x_train, update[0], self.n_clusters)
            self.distortion.append(update[1])
            if np.array_equal(self.centroids, new_centroids):
                self.centroids = new_centroids
                break
            self.centroids = new_centroids

        return update[0]


