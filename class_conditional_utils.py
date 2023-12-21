import numpy as np
from sklearn.mixture import GaussianMixture


def ccgmm_codivide(loss: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    To compute the GMM probabilities with a Class-Conditional approach. And to log the means of the resulting GMM.
    This function also computes the original GMM division and logs the comparison between the two.

    @params:
    - targets - np.array with the class of every element.
    """
    num_classes = max(targets) + 1  # Find total number of classes
    prob = np.zeros(loss.size()[0])
    for c in range(num_classes):
        mask = targets == c

        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(loss[:, 0][mask].reshape(-1, 1))

        clean_idx, _ = gmm.means_.argmin(), gmm.means_.argmax()

        p = gmm.predict_proba(loss[:, 0][mask].reshape(-1, 1))
        prob[mask] = p[:, clean_idx]

    return prob


def gmm_codivide(loss: np.ndarray) -> np.ndarray:
    """
    To compute the GMM probabilities with a Class-Conditional approach. And to log the means of the resulting GMM.
    This function also computes the original GMM division and logs the comparison between the two.

    @params:
    - targets - np.array with the class of every element.
    """
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(loss)
    prob = gmm.predict_proba(loss)
    prob = prob[:, gmm.means_.argmin()]
    return prob
