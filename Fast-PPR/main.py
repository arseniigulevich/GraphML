from torch_geometric.utils import from_networkit
from torch_geometric import EdgeIndex
import networkit as nk
import Fast_PPR as fp
import torch
import random as rd


G_nk = nk.readGraph("./foodweb-baydry.konect")
n = G_nk.numberOfNodes()
is_undir = not G_nk.isDirected()
G_pyg = from_networkit(G_nk)[0]
G_pyg_rev = torch.stack((G_pyg[1], G_pyg[0]))

G_pyg = EdgeIndex(G_pyg, sparse_size=(n, n), is_undirected=is_undir)
G_pyg_rev = EdgeIndex(G_pyg_rev, sparse_size=(n, n), is_undirected=is_undir)

my_ppr = fp.PPR(G_pyg, 0.3, 1/n)  # G_pyg.sparse_size(dim=0)

results = []
sizes = []
number_of_runs = 5

for i in range(number_of_runs):
    s = rd.randrange(n)
    t = s
    while t == s:
        t = rd.randrange(n)
    result, combined_size = my_ppr.fast_ppr(s, t)
    results.append([s, t, result])
    sizes.append(combined_size)

print(f'delta: {1/n}')
print(f'fast ppr scores: {results}')
print(f'average size of the union of target and frontier sets: {sum(sizes)/len(sizes)}')
print(f'number of scores higher than delta: {(torch.tensor([x[2] for x in results]) > 1/n).nonzero().flatten().size(dim=0)}')





