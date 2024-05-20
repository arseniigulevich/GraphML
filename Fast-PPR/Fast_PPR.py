import numpy as np
import torch as torch
from torch_cluster import random_walk
from torch.distributions.geometric import Geometric

def get_in_neighbors(G, u) -> list:
    mask = G[1] == u
    return G[:, mask][0].tolist()

def get_out_degree(G, u) -> int:
    return (G[0] == u).nonzero().size(dim=0)

class PPR:
    def __init__(self, G, alpha, delta):
        self._G = G  # EdgeIndex
        self._alpha = alpha
        self._delta = delta
        self._epsilon = np.sqrt(delta)
        self._beta = 1 / 6
        self._c = 500

    def fast_ppr(self, s, t):
        t_set, f_set, pi_inv = self._frontier(t)
        if s in t_set:
            return pi_inv[s].item(), len(t_set | f_set)
        else:
            number_of_walks = int(np.ceil(self._c * self._epsilon / self._delta))
            sum_pi_inv = 0
            for i in range(number_of_walks):
                Geom = Geometric(torch.tensor([self._alpha]))
                L = int(np.ceil(Geom.sample().item()))
                L = L if L > 0 else 1
                walk = random_walk(self._G[0], self._G[1], torch.tensor([s]), walk_length=L).flatten()
                for v in walk:
                    if v.item() in f_set:
                        sum_pi_inv += pi_inv[v].item()
            return sum_pi_inv/number_of_walks, len(t_set | f_set)

    def _frontier(self, t):
        error_inv = self._beta * self._epsilon
        estimate_vec = torch.zeros(self._G.num_rows)
        estimate_vec[t] = self._alpha
        residual_vec = torch.clone(estimate_vec)
        target_set = {t}
        frontier_set = set([])
        residual_vec_bigger_ix = (residual_vec > error_inv * self._alpha).nonzero().flatten()

        while residual_vec_bigger_ix.size(dim=0) > 0:
            w = residual_vec_bigger_ix[0].item()
            for u in get_in_neighbors(self._G, w):
                capital_delta = (1.0 - self._alpha) * residual_vec[w].item() / get_out_degree(self._G, u)
                estimate_vec[u] = estimate_vec[u] + capital_delta
                residual_vec[u] = residual_vec[u] + capital_delta
                if estimate_vec[u].item() > error_inv:
                    target_set.add(u)
                    frontier_set = set(get_in_neighbors(self._G, u)).union(frontier_set)
            residual_vec[w] = 0
            residual_vec_bigger_ix = (residual_vec > error_inv * self._alpha).nonzero().flatten()

        frontier_set = frontier_set - target_set
        return target_set, frontier_set, estimate_vec




