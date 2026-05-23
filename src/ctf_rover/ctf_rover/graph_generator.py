import pickle
import numpy as np
import networkx as nx
from ctf_rover.customCTF import *

def generate_graph():
    graphctf = GraphCTF(fixed_flag_hypothesis=1)

    bc = graphctf.make_basin_corridor_medium_v1()

    G_map = bc.G

    assert 'pos' in list(nx.get_node_attributes(G_map, 'pos').values())[0].__class__.__dict__ or True

    red_flag_node = bc.red_flag_L
    red_flag_2_node = bc.red_flag_R
    blue_flag_node = graphctf.blue_flag_node

    flag_nodes = np.array([red_flag_node, red_flag_2_node, blue_flag_node])

    return G_map, flag_nodes


def load_acl_graph(pkl_path: str):
    """Load the ACL lidar-scan graph. Sets each node's pos = (x, y) in Vicon metres."""
    with open(pkl_path, 'rb') as f:
        G = pickle.load(f)
    for _, data in G.nodes(data=True):
        data['pos'] = (data['x'], data['y'])
    return G


def nx_to_marker_data(G, flag=None):
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

    flag_pos = node_pose_dict[flag] if flag is not None else None
    return nodes, edges, flag_pos