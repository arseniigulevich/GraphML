import networkit as nk
import numpy as np
import torch as torch
import networkx as nx
import torch_geometric


class PPR:
    def __init__(self, G, alpha, delta):
        self._G = G
        self._alpha = alpha
        self._delta = delta

    def fast_ppr(self, s, t):
        t_set, f_set, pi_inv = self._frontier(t)
        c = 350
        beta = 1/6


        pass
    def _frontier(self ,t):
        pass