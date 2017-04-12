import numpy as np
from scipy.stats import multivariate_normal

class GMM(object):
    def __init__(self, n_clusters=3, eps=0.001):
        self.n_clusters = n_clusters
        self.eps = eps
        self.mu = None
        self.mu_init = None
        self.cov = None
        self.cov_init = None
        self.pie = None
        self.pie_init = None
        self.p_ij = None
        self.p_ij_init = None
        #self.log_likelihoods = []
        self.e = []

    def fit(self, x_train, max_iterations=1000, eps=1e-5):
        n, d = x_train.shape

        if self.mu is None:
            self.mu_init = x_train[np.random.choice(n, self.n_clusters, replace=False)]
            self.mu = self.mu_init

        if self.cov is None:
            self.cov_init = [np.eye(d)] * self.n_clusters
            self.cov = self.cov_init

        if self.pie is None:
            self.pie_init = [1./self.n_clusters] * self.n_clusters
            self.pie = self.pie_init

        if self.p_ij is None:
            self.p_ij_init = np.zeros((n, self.n_clusters))
            self.p_ij = self.p_ij_init

        #log_likelihoods = []

        while True:
            g_matrix = map(lambda mu, cov: list(map(lambda x: multivariate_normal.pdf(x, mu, cov), x_train)), self.mu, self.cov)
            new_p_ij = np.array(list(g_matrix)).T / np.array(list(g_matrix)).T.sum(axis=1)[:, np.newaxis]
            self.e.append(np.linalg.norm(new_p_ij - self.p_ij))

            if np.linalg.norm(new_p_ij - self.p_ij) < eps:
                break

            self.p_ij = new_p_ij

            new_mu = np.dot(self.p_ij.T, x_train) / self.p_ij.T.sum(axis=1)[:, np.newaxis]
            new_pie = self.p_ij.T.sum(axis=1) / self.p_ij.shape[0]
            new_cov = []
            x_t = x_train - self.mu[:, np.newaxis]
            for j in range(self.n_clusters):
                x_u = np.array(map(lambda x: np.dot(x.T, x), x_t[j][:, np.newaxis]))
                new_cov.append((x_u * self.p_ij.T[j][:, np.newaxis][:, np.newaxis]).sum(axis=0) / self.p_ij.T[j].sum())

            #log_likelihood = np.sum(np.log(np.sum(self.p_ij, axis=1)))
            #self.log_likelihoods.append(log_likelihood)
            self.mu = new_mu
            self.pie = new_pie
            self.cov = new_cov

        return self.pie, self.mu, self.cov, self.p_ij




