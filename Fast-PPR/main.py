from torch_geometric.utils import from_networkit
from torch_geometric import EdgeIndex
import networkit as nk
from


G_nk = nk.readGraph("./foodweb-baydry.konect")
n = G_nk.numberOfNodes()
is_undir = not G_nk.isDirected()
G_pyg = from_networkit(G_nk)[0]
G_pyg = EdgeIndex(G_pyg, sparse_size=(n, n), is_undirected=is_undir)
print(G_pyg)

