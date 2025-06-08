import numpy as np
from scipy.stats import norm

def estimate_grand_mean(C_Y: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """"Estimates grand mean i.e. the root node values of a phylogenetic tree

    Args:
        C_Y (np.ndarray): ovariance among tips
        Y (np.ndarray): Trait matrix at the tips

    Returns:
        np.ndarray: Estimated traits at root node
    """
    n = C_Y.shape[0]
    ones = np.ones((n, 1))
    inv_C = np.linalg.inv(C_Y)
    GM = (np.linalg.inv(ones.T @ inv_C @ ones) @ ones.T @ inv_C @ Y).T
    return GM

def estimate_inner_nodes(GM: np.ndarray, C_AY: np.ndarray, C_Y: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Estimates ancestral traits of internal nodes of a phylogenetic tree.

    Args:
        GM: Grand mean - the estimated values at the root node of a phylogenetic tree
        C_AY: Covariance between inner nodes and tips
        C_Y: Covariance among tips
        Y: Trait matrix at the tips

    Returns:
        np.ndarray: Estimated traits at internal nodes
    """ 
    GM = GM.reshape(1, -1)
    ancestral_estimates = C_AY @ np.linalg.inv(C_Y) @ (Y - GM) + GM
    return ancestral_estimates

def estimate_conditional_covariance(C_A : np.ndarray, C_AY : np.ndarray, C_Y : np.ndarray) -> np.ndarray:
    """Estimates the covariance matrix for the GLS estimation of ancestral states.

    Args:
        C_A: Covariance among inner nodes (ancestral states)
        C_AY: Covariance between inner nodes and tips
        C_Y: Covariance among tips

    Returns:
        np.ndarray: The covariance matrix for ancestral sate reconstruction
    """ 
    return C_A - C_AY @ np.linalg.inv(C_Y) @ C_AY.T

def get_confidence_interval(cov : np.ndarray, ace : np.ndarray) ->  np.ndarray:
    """Calculates the 0.95 confidence interval for the GLS estimator's ancestral state reconstructions.

    Args:
        cov: The covariance matrix for ancestral state reconstruction
        ace: Ancestral state reconstructions i.e. estimated trait values of inner nodes

    Returns:
        np.ndarray: 0.95 CI interval for each ancestral node's estimated traits
    """ 
    standard_error = np.sqrt(np.diag(cov)).reshape(-1, 1)
    margin = standard_error * norm.ppf(0.975)
    return np.stack((ace + margin, ace - margin), axis=1)