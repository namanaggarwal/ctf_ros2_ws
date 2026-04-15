import numpy as np
import networkx as nx
from ctf_rover.customCTF import *

def generate_graph(seed=30597, nx_dim=6, ny_dim=6):
    rng = np.random.default_rng(seed)
    N = nx_dim * ny_dim

    jitter = 0.35
    rho = 0.80
    ell = 3.2

    choke_row_start = 2
    choke_row_end = 3
    passage_cols = {0, 1, 2, ny_dim-3, ny_dim-2, ny_dim-1}
    funnel_rows = 2

    # ── Positions ─────────────────────────────────────────────────────────────────
    pos = {}
    for i in range(nx_dim):
        for j in range(ny_dim):
            u = i * ny_dim + j
            pos[u] = np.array([j + jitter * rng.normal(),
                                -i + jitter * rng.normal()], dtype=float)

    # ── 4-connected grid edges ────────────────────────────────────────────────────
    edges = []
    for i in range(nx_dim):
        for j in range(ny_dim):
            u = i * ny_dim + j
            if i + 1 < nx_dim:
                edges.append((u, (i + 1) * ny_dim + j))
            if j + 1 < ny_dim:
                edges.append((u, i * ny_dim + (j + 1)))

    # ── Correlated random field ───────────────────────────────────────────────────
    coords = np.array([[i, j] for i in range(nx_dim) for j in range(ny_dim)], dtype=float)
    D      = np.linalg.norm(coords[:, None] - coords[None], axis=2)
    Sigma  = np.exp(-D / ell) + 1e-6 * np.eye(N)
    L      = np.linalg.cholesky(Sigma)
    z      = L @ rng.standard_normal(N)
    tau    = np.quantile(z, 1 - rho)
    open_nodes = {u for u in range(N) if z[u] >= tau}

    # ── Choke mask ────────────────────────────────────────────────────────────────
    # Force close the wall; force open passage cols inside choke
    for i in range(choke_row_start, choke_row_end + 1):
        for j in range(ny_dim):
            u = i * ny_dim + j
            if j not in passage_cols:
                open_nodes.discard(u)
            else:
                open_nodes.add(u)

    # Funnel: force passage cols open for `funnel_rows` rows above and below the choke
    # so passages are guaranteed to reach into both regions regardless of CRF
    for i in range(max(0, choke_row_start - funnel_rows), choke_row_start):
        for j in passage_cols:
            open_nodes.add(i * ny_dim + j)
    for i in range(choke_row_end + 1, min(nx_dim, choke_row_end + 1 + funnel_rows)):
        for j in passage_cols:
            open_nodes.add(i * ny_dim + j)

    # ── Build graph + largest CC ──────────────────────────────────────────────────
    G = nx.Graph()
    for u in open_nodes:
        G.add_node(u, pos=pos[u])
    for u, v in edges:
        if u in open_nodes and v in open_nodes:
            G.add_edge(u, v)

    largest_cc = max(nx.connected_components(G), key=len)
    G_map = G.subgraph(largest_cc).copy()

    node_pose_dict = nx.get_node_attributes(G_map, 'pos')

    # get bounds
    x_vals = [p[0] for p in node_pose_dict.values()]
    y_vals = [p[1] for p in node_pose_dict.values()]

    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    def update_pos(pos):
        x, y = pos

        x_new = 0 + 10.*(x-x_min)/(x_max - x_min)
        y_new = -10 + 10.*(y-y_min)/(y_max - y_min)

        return np.array([x_new, y_new])

    # apply normalization
    updated_pos = {
        node: {"pos": update_pos(node_pose_dict[node])}
        for node in G_map.nodes()
    }
    updated_pos = {node: {"pos": update_pos(node_pose_dict[node])} for node in G_map.nodes()}

    nx.set_node_attributes(G_map, updated_pos)

    return G_map

def nx_to_marker_data(G):
    node_pose_dict = nx.get_node_attributes(G, 'pos')

    nodes = []
    node_index = {}

    # assign indices
    for idx, node in enumerate(G.nodes()):
        x, y = node_pose_dict[node]
        nodes.append((float(x), float(y)))
        node_index[node] = idx

    edges = []
    for u, v in G.edges():
        edges.append((node_index[u], node_index[v]))

    return nodes, edges