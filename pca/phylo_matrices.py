from ..models.lambdamodels import DownLambda
from ..execution import OrderedExecutor
from ..tree import HypTree
from ..tree.topology import TopologyNode
import jax
import numpy as np

class PhylogenicCovarianceMatrices:
    """Generates phylogenetic covariance matrices C_Y, C_A, C_AY as described in Martins and Hansen (1997). 
    """

    def __init__(self, tree: HypTree):
        self.tree = tree
        self._prepare_tree()
        self.cumulative_lengths
        self.levels

    def _prepare_tree(self):
        tree = self.tree

        tree.add_property('level', shape=(1,))
        tree.add_property('cum_sum_edge_length', shape=(1,))
        
        @jax.jit
        def down(edge_length, parent_cum_sum_edge_length, parent_level, **args):
            return {
                'cum_sum_edge_length': edge_length + parent_cum_sum_edge_length,
                'level': 1 + parent_level
            }

        downmodel = DownLambda(down_fn=down)
        exe = OrderedExecutor(downmodel)
        exe.down(tree)
        
        self.levels = {node.id: tree.data['level'][node.id][0] for node in tree.iter_topology_bfs()}
        self.cumulative_lengths = {node.id: tree.data['cum_sum_edge_length'][node.id][0] for node in tree.iter_topology_bfs()}

    def _find_shared_branch_lengths(self, a: TopologyNode, b: TopologyNode) -> float:
        while self.levels[a.id] > self.levels[b.id]:
            a = a.parent
        while self.levels[a.id] < self.levels[b.id]:
            b = b.parent
        while a.id != b.id:
            a = a.parent
            b = b.parent
        return self.cumulative_lengths[a.id]

    def get_covariance_matrices(self):
        """Returns the phylogenetic covariance matrices in breadth-first order.

        Returns:
            C_Y: Covariance among tips
            C_A: Covariance among inner nodes
            C_AY: Covariance between inner nodes and tips
        """
        tree = self.tree
        leaves = list(tree.iter_topology_leaves_bfs())
        inner_nodes = list(tree.iter_topology_inner_nodes_bfs())
        all_nodes = leaves + inner_nodes
        n_leaves = len(leaves)
        n_inner = len(inner_nodes)
        n = n_leaves + n_inner
        C_full = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n): 
                if i != j:
                    shared = self._find_shared_branch_lengths(all_nodes[i], all_nodes[j])
                    C_full[i, j] = shared
                    C_full[j, i] = shared  
                else:
                    C_full[i, j] = self.cumulative_lengths[all_nodes[i].id]
        
        C_Y = C_full[:n_leaves, :n_leaves]
        C_A = C_full[n_leaves:, n_leaves:]
        C_AY = C_full[n_leaves:, :n_leaves]

        return C_Y, C_A, C_AY
        
    
        
