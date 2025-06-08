import numpy as np
import pandas as pd
from .gls_reconstuction import estimate_grand_mean
from .phylo_matrices import PhylogenicCovarianceMatrices

class PCA:
    def __init__(self):
        self.U = None                       # Eigenvectors
        self.S = None                       # Eigenvalues
        self.mean = None                    # Mean
        self.explained_variance = None
        self.cumulative_variance = None
        
    def _explained_variance(self):
        summed_var = np.sum(self.S)
        explained_proportion = self.S / summed_var
        self.explained_variance = explained_proportion
        
    def _cumulative_variance(self):
        if self.explained_variance is None:
            self._explained_variance()
        self.cumulative_variance = np.cumsum(self.explained_variance)

    def fit(self, Y: np.ndarray):
        """Perform regular PCA on tip trait matrix Y."""
        self.mean = np.mean(Y, axis=0)
        Y_centered = (Y - self.mean)
        # Covariance matrix
        cov_matrix = np.cov(Y_centered, rowvar=False)
        # Extract eigenvalues and eigenvectors
        self.U, self.S, Vh = np.linalg.svd(cov_matrix.T)
        return self.U, self.S

    def transform(self, features_leaves: np.ndarray, features_inner: np.ndarray = None, dims: int = 2):
        """Project tip and inner node trait data onto the top `dims` principal components."""
        if self.U is None or self.mean is None:
            raise ValueError("You must run `.fit()` before calling `.transform()`.")

        mean = self.mean.reshape(1, -1)
        centered_leaves = features_leaves - mean
        transformed_leaves = centered_leaves @ self.U[:, :dims]
        
        if features_inner is not None: 
            centered_inner = features_inner - mean
            transformed_inner = centered_inner @ self.U[:, :dims]
            return transformed_leaves, transformed_inner
        else: 
            return transformed_leaves
        
    def fit_transform(self, tree, dims: int = 2):
        """Performs regular PCA on tip trait values of a tree and returns the transformed values"""
        Y = tree.data['value'][tree.is_leaf][:,-1]
        self.fit(Y)
        return self.transform(Y, None, dims)
    
    def get_explained_variance(self, pcs = 10):
        if self.S is None:
            raise ValueError("You must run `.fit()` before calling `.get_explained_variance()`.")
        if self.explained_variance is None:
            self._explained_variance()
        pc_labels = [f"PC{i+1}" for i in range(len(self.explained_variance))]
        df = pd.DataFrame({
            "PC": pc_labels,
            "Explained Variance (%)": np.round(self.explained_variance * 100, 2)
        })
        print(df.head(pcs))
            
    def get_cumulative_variance(self, pcs = 10):
        if self.S is None:
            raise ValueError("You must run `.fit()` before calling `.get_cumulative_variance()`.")
        if self.cumulative_variance is None:
            self._cumulative_variance()
        pc_labels = [f"PC{i+1}" for i in range(len(self.cumulative_variance))]
        df = pd.DataFrame({
            "PC": pc_labels,
            "Cumulative Variance (%)": np.round(self.cumulative_variance * 100, 2)
        })
        print(df.head(pcs))
    
class PhylogeneticPCA:
    def __init__(self):
        self.Up = None                      # Eigenvectors
        self.Sp = None                      # Eigenvalues
        self.a = None                       # Ancestral mean
        self.explained_variance = None
        self.cumulative_variance = None
        
    def _explained_variance(self):
        summed_var = np.sum(self.Sp)
        explained_proportion = self.Sp / summed_var
        self.explained_variance = explained_proportion
        
    def _cumulative_variance(self): 
        if self.explained_variance is None:
            self._explained_variance()
        self.cumulative_variance = np.cumsum(self.explained_variance)

    def fit(self, a: np.ndarray, C: np.ndarray, Y: np.ndarray):
        """Perform phylogenetic PCA on tip trait matrix Y using phylogenetic covariance C and ancestral mean a."""
        n = Y.shape[0]
        inv_C = np.linalg.inv(C)
        self.a = a.reshape(1, -1)
        # Center the data
        Y_centered = Y - self.a
        # Evolutionary covariance matrix
        P = (1 / (n - 1)) * Y_centered.T @ inv_C @ Y_centered
        # PCA via SVD
        self.Up, self.Sp, Vh = np.linalg.svd(P)
        return self.Up, self.Sp

    def transform(self, Y_leaves: np.ndarray, Y_inner: np.ndarray = None, dims: int = 2):
        """Project trait data onto the top `dims` principal components."""
        if self.Up is None or self.a is None:
            raise ValueError("You must run `.fit()` before calling `.transform()`.")
        
        a = self.a
        centered_leaves = Y_leaves - a
        transformed_leaves = centered_leaves @ self.Up[:, :dims]
        
        if Y_inner is not None: 
            centered_inner = Y_inner - a
            transformed_inner = centered_inner @ self.Up[:, :dims]
            return transformed_leaves, transformed_inner
        else: 
            return transformed_leaves
        
    def fit_transform(self, tree, dims: int = 2):
        """Performs phylogenetic PCA on tip trait values of a tree and returns the transformed values"""
        CovMatrices = PhylogenicCovarianceMatrices(tree)
        Y = tree.data['value'][tree.is_leaf][:,-1]
        C = CovMatrices.get_covariance_matrices()
        a = estimate_grand_mean(C,Y)
        self.fit(a, C, Y)
        return self.transform(Y, None, dims)
    
    """Explained variance is only meaningful if branch lengths of phylogeny are in a compatible phenotypic unit"""
    def get_explained_variance(self, pcs = 10):
        if self.Sp is None:
            raise ValueError("You must run `.fit()` before calling `.get_explained_variance()`.")
        if self.explained_variance is None:
            self._explained_variance()
        pc_labels = [f"PC{i+1}" for i in range(len(self.explained_variance))]
        df = pd.DataFrame({
            "PC": pc_labels,
            "Explained Variance (%)": np.round(self.explained_variance * 100, 2)
        })
        print(df.head(pcs))
        
    """Explained variance is only meaningful if branch lengths of phylogeny are in a compatible phenotypic unit"""        
    def get_cumulative_variance(self, pcs = 10):
        if self.Sp is None:
            raise ValueError("You must run `.fit()` before calling `.get_cumulative_variance()`.")
        if self.cumulative_variance is None:
            self._cumulative_variance()
        pc_labels = [f"PC{i+1}" for i in range(len(self.cumulative_variance))]
        df = pd.DataFrame({
            "PC": pc_labels,
            "Cumulative Variance (%)": np.round(self.cumulative_variance * 100, 2)
        })
        print(df.head(pcs))
    
    
        
        



        
