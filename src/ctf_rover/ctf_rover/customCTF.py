import functools
import random
import copy

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from gymnasium.spaces import Discrete, MultiDiscrete, Tuple
from gymnasium.spaces import Dict, Box

from pettingzoo import ParallelEnv
from pettingzoo.test import parallel_api_test

from matplotlib import pyplot as plt

from pettingzoo import ParallelEnv
import functools

#from utils import draw_grid
from typing import Any

import torch as th
from torch import nn

from stable_baselines3 import PPO
import supersuit as ss

"""
Location: ACL servers.
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

"""

def draw_spline(poses_with_headings):
    # Extract positions and heading indices
    poses = np.array([[i, j] for i, j, _ in poses_with_headings])
    heading_indices = np.array([h for _, _, h in poses_with_headings])

    # Convert heading indices to unit direction vectors
    angle_step = 2 * np.pi / 8
    angles = heading_indices * angle_step
    headings = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    # Parameterize trajectory by index (or distance if you want variable timing)
    t = np.arange(len(poses))

    # Scale heading vectors by local distances to approximate velocity magnitude
    segment_lengths = np.linalg.norm(np.diff(poses, axis=0), axis=1)
    segment_lengths = np.append(segment_lengths, segment_lengths[-1])  # Repeat last

    tangents = headings * segment_lengths[:, None]

    # Create Hermite spline for x and y
    spline_x = CubicHermiteSpline(t, poses[:, 0], tangents[:, 0])
    spline_y = CubicHermiteSpline(t, poses[:, 1], tangents[:, 1])

    # Sample smoothly along the path
    t_fine = np.linspace(t[0], t[-1], 300)
    x_fine = spline_x(t_fine)
    y_fine = spline_y(t_fine)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(x_fine, y_fine, label='Smooth Trajectory', color='blue', linewidth=2)
    plt.scatter(poses[:, 0], poses[:, 1], color='red', label='Original Poses')

    # Draw heading arrows at each pose
    for (x, y), (dx, dy) in zip(poses, headings):
        plt.arrow(x, y, dx * 0.35, dy * 0.35, head_width=0.1, head_length=0.1, fc='green', ec='green')

    plt.grid(True)
    #plt.legend()
    #plt.title("Smooth Trajectory with Discrete Headings")
    #plt.xlabel("i")
    #plt.ylabel("j")
    plt.axis("equal")
    plt.xlim(-1, 7)
    plt.ylim(-1, 6)
    plt.show()
draw_spline(poses_with_headings=poses_with_headings[-2:])

"""

def draw_grid(grid_size, blue_agent_pos=[(2,2,3)], red_agent_pos=[(3,7,7)], blue_flag_loc=(1,1), red_flag_loc = (7,7), blue_flag_img_path="flag_imgs/blue_flag.png", red_flag_img_path="flag_imgs/red_flag.png"):

    def heading_to_direction_vec(heading: int):
        assert heading in range(8)
        direction_vec = None
        if heading == 0: direction_vec = np.array([1, 0])
        elif heading == 1: direction_vec = np.array([1, 1])
        elif heading == 2: direction_vec = np.array([0, 1])
        elif heading == 3: direction_vec = np.array([-1, 1])
        elif heading == 4: direction_vec = np.array([-1, 0])
        elif heading == 5: direction_vec = np.array([-1, -1])
        elif heading == 6: direction_vec = np.array([0, -1])
        elif heading == 7: direction_vec = np.array([1, -1])
        return direction_vec

    ### Add a function to draw agents
    ### Add a function to draw flags
    m, n = grid_size, grid_size
    fig, ax = plt.subplots(figsize=(n, m))

    # Draw grid lines
    for x in range(n+1):
        ax.plot([x, x], [0, m], color='black', linewidth=1)
    for y in range(m+1):
        ax.plot([0, n], [y, y], color='black', linewidth=1)

        # Add markers at intersections
    for agent_iter, agent_pos in enumerate(blue_agent_pos):
        ax.plot(agent_pos[0], agent_pos[1], marker='s', color='blue', markersize=7)
        start_point = np.array([agent_pos[0], agent_pos[1]])
        heading = agent_pos[2]
        end_point = start_point + 0.35*heading_to_direction_vec(heading=heading)
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='blue', linewidth=1.75)
        pass

    for agent_iter, agent_pos in enumerate(red_agent_pos):
        ax.plot(agent_pos[0], agent_pos[1], marker='s', color='red', markersize=7)
        start_point = np.array([agent_pos[0], agent_pos[1]])
        heading = agent_pos[2]
        end_point = start_point + 0.35*heading_to_direction_vec(heading=heading)
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='red', linewidth=1.75)
        pass
    
    if blue_flag_loc is not None:
        blue_flag_img = mpimg.imread(blue_flag_img_path)
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        blue_flag_img = OffsetImage(blue_flag_img, zoom=0.22)
        ab_blue = AnnotationBbox(blue_flag_img, blue_flag_loc, frameon=False)
        ax.add_artist(ab_blue)
    
    if red_flag_loc is not None:
        red_flag_img = mpimg.imread(red_flag_img_path)
        red_flag_img = OffsetImage(red_flag_img, zoom=0.22)
        ab_red = AnnotationBbox(red_flag_img, red_flag_loc, frameon=False)
        ax.add_artist(ab_red)

    #size = 0.5
    #ax.imshow(img=flag_img, extent=[flag_loc[0] - size, flag_loc[0] + size, flag_loc[1] - size, flag_loc[1] + size])
    #ax.text(flag_loc[0], flag_loc[1], "F", fontsize=14, ha='center', va='center', color='purple')

    ax.set_xlim(0, n)
    ax.set_ylim(0, m)
    ax.set_aspect('equal')
    ax.axis('off')
    #plt.show()
    return fig, ax

from scipy.interpolate import CubicHermiteSpline
def draw_spline(ax, poses_with_headings, plot=False, color='blue'):
    """
    Example input for poses_with_headings:
    poses_with_headings = [
    (0, 0, 0),    # Right
    (1, 2, 1),    # Up-right
    (3, 3, 2),    # Up
    (4, 1, 5),    # Down-left
    (6, 4, 6),    # Left
    ]
    """
    def heading_to_direction_vec(heading: int):
        assert heading in range(8)
        direction_vec = None
        if heading == 0: direction_vec = np.array([1, 0])
        elif heading == 1: direction_vec = np.array([1, 1])
        elif heading == 2: direction_vec = np.array([0, 1])
        elif heading == 3: direction_vec = np.array([-1, 1])
        elif heading == 4: direction_vec = np.array([-1, 0])
        elif heading == 5: direction_vec = np.array([-1, -1])
        elif heading == 6: direction_vec = np.array([0, -1])
        elif heading == 7: direction_vec = np.array([1, -1])
        return direction_vec
    
    # Extract positions and heading indices
    poses = np.array([[i, j] for i, j, _ in poses_with_headings])
    heading_indices = np.array([h for _, _, h in poses_with_headings])

    # Convert heading indices to unit direction vectors
    angle_step = 2 * np.pi / 8
    angles = heading_indices * angle_step
    headings = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    # Parameterize trajectory by index (or distance if you want variable timing)
    t = np.arange(len(poses))

    # Scale heading vectors by local distances to approximate velocity magnitude
    segment_lengths = np.linalg.norm(np.diff(poses, axis=0), axis=1)
    segment_lengths = np.append(segment_lengths, segment_lengths[-1])  # Repeat last

    tangents = headings * segment_lengths[:, None]

    # Create Hermite spline for x and y
    spline_x = CubicHermiteSpline(t, poses[:, 0], tangents[:, 0])
    spline_y = CubicHermiteSpline(t, poses[:, 1], tangents[:, 1])

    # Sample smoothly along the path
    t_fine = np.linspace(t[0], t[-1], 300)
    x_fine = spline_x(t_fine)
    y_fine = spline_y(t_fine)

    # Plot
    if plot:
        if color.startswith('r'):
            spline_color = 'tomato'
            scatter_color = spline_color
        elif color.startswith('b'):
            spline_color = 'dodgerblue'
            scatter_color = spline_color
        ax.plot(x_fine, y_fine, label='Smooth Trajectory', color=spline_color, linewidth=2) #plt.plot(x_fine, y_fine, label='Smooth Trajectory', color='blue', linewidth=2)
        ax.scatter(poses[:, 0], poses[:, 1], color=scatter_color, label='Original Poses')

        for pose in poses_with_headings[:-1]:
            start_point = np.array([pose[0], pose[1]])
            heading = pose[2]
            end_point = start_point + 0.2*heading_to_direction_vec(heading=heading) #0.35 as the head for the pose.
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color=color, linewidth=1.75) #scatter_color
        
        """
        # Draw heading arrows at each pose
        for (x, y), (dx, dy) in zip(poses, headings):
            ax.arrow(x, y, dx * 0.35, dy * 0.35, head_width=0.1, head_length=0.1, fc='green', ec='green')
        plt.grid(True)
        plt.axis("equal")
        plt.xlim(-1, 7)
        plt.ylim(-1, 6)
        plt.show()
        """
    return ax, x_fine, y_fine

from stable_baselines3.ppo.policies import MultiInputPolicy
class MaskedMultiInputPolicy(MultiInputPolicy):
    ### !!!!!!!!!!!!
    """
    This modification is required to incorporate / accomodate for action_masks at different states of the system / observations for an agent.
    Takes in as input an obs dict such as obs = {"observation": obs_array, "action_mask": action_mask_arr}. Masks the output of the network with the provided action_mask_arr.

    """
    def forward(self, obs, deterministic=False):
        # 'obs' is a dict with 'obs' and 'action_mask'
        features = self.extract_features(obs)  # handles dict
        latent_pi, latent_vf = self.mlp_extractor(features)

        dist = self._get_action_dist_from_latent(latent_pi)
        # Extract the original logits from the distribution
        logits = dist.distribution.logits

        # Apply mask: set invalid action logits to a very negative value
        action_mask = obs["action_mask"]
        # Avoid log(0) by clamping values
        #mask = (action_mask + 1e-8).log()
        """
        mask = (action_mask + np.exp(-np.inf)).log()
        masked_logits = logits + mask
        # This was crashing for the case of agent_deaths in DefenseCTF_v2 when the action_mask is empty i.e. action_mask = [].
        """
        masked_logits = logits.masked_fill(action_mask == 0, -1e10)
        dist.distribution.logits = masked_logits
        
        actions = dist.get_actions(deterministic=deterministic)

        log_prob = dist.log_prob(actions)
        #return actions, None, log_prob

        values = self.value_net(latent_vf)
        return actions, values, log_prob

import random
import numpy as np
import torch
from gymnasium.utils import seeding

def make_seeded_rngs(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np_random, actual_seed = seeding.np_random(seed)
    torch_rng = torch.Generator().manual_seed(actual_seed)
    return {
        "np_random": np_random,
        "torch_rng": torch_rng,
        "actual_seed": actual_seed
    }

# This is for the Basin-Cooridoor graph...
from dataclasses import dataclass
from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple

@dataclass
class BasinCorridorGraph:
    G: nx.Graph
    pos: Dict[int, np.ndarray]  # node -> (x,y)
    red_flag_L: int
    red_flag_R: int
    blue_flag: int
    # handy regions (optional)
    top_left_basin: List[int]
    top_right_basin: List[int]
    bottom_region: List[int]

from matplotlib.patches import Circle # For rendering...
class GraphCTF(ParallelEnv):
    meta_data = {'name': 'GraphCTF_v0'}
    """
    v0 = known map, unknown occupancy beyond visibility radius.
    v1 = unknown map and unknown occupancy beyond visibility radius.
    """
    def __init__(self,
                 ctf_player_config="2v2",
                 max_num_cycles=30, #35, #50, #200,
                 verbose=False,
                 seed=None,
                 obs_mode={'occupancy': 'partial', 'global_map': 'known', 'flag': 'known'},
                 game_mode='half',
                 obs_version=1,  # 1 for v1 observations, 2 for v2 (faster direct slicing)
                 opp_policy=None,  # Optional: set opponent policy directly (bypasses GraphCoopEnv wrapper)
                 active_team=None,  # 'Red' or 'Blue' - the team being trained (opponent is passive)
                 fixed_flag_hypothesis=0,  # None = sample; 0 = left flag; 1 = right flag
                 flag_hypothesis_weights=None,  # [w0, w1] unnormalized; None = uniform when sampling
                 **kwargs):
        self.fixed_flag_hypothesis = fixed_flag_hypothesis
        if flag_hypothesis_weights is not None:
            import numpy as _np
            w = _np.array(flag_hypothesis_weights, dtype=float)
            self._flag_probs = w / w.sum()
        else:
            self._flag_probs = None
        self.reveal_flag = kwargs.get('reveal_flag', False)  # If True, Blue always has enemy_flag_known=True (bypasses frontier discovery)
        self.obs_version = obs_version  # Controls which get_observation method is used
        self.game_mode = game_mode
        assert self.game_mode in ['half', 'full']
        """
        Description: self.game_mode = 'half' is no blue flag. 'full' is blue flag.
        """

        # Opponent policy integration (avoids GraphCoopEnv wrapper overhead)
        # Note: actual initialization is deferred until after agents are defined
        self._opp_policy = opp_policy
        self._active_team = active_team  # 'Red' or 'Blue'
        self._active_agents = None
        self._passive_agents = None
        self._deferred_opp_policy_init = (opp_policy is not None and active_team is not None)

        import copy
        import networkx as nx

        self.metadata = { "name": "GraphCTF"}
        self.render_mode = "human"
        self.verbose = verbose
        self.max_num_cycles = max_num_cycles # ideally based on graph diameter. Our example has diameter = 20, therefore max_num_cycles = 30.
        self.obs_mode = {'occupancy': 'known', 'global_map': 'known', 'flag': 'unknown'} #copy.deepcopy(obs_mode)
        self.obs_mode = {'occupancy': 'known', 'global_map': 'known', 'flag': 'partial'}
        self.obs_mode = {'occupancy': 'known', 'global_map': 'known', 'flag': 'partial'} # To start with (train on just one flag -- see if you learn to reach the flag under partial information on the flag).

        """
        Note: Agent occupancy known implies we are implicitly assuming fixed number of teammates and opponents / team sizes are known.
        """

        # k-hop ego-graph and crafted features to incorporate adversary information -- adversary occupancy, hops to adversary, degree of node...
        # k = 2, 3
        # 2 layer MPNNs...
        # Freeze MPNNs for later PSRO iterations...
        # Train on p=0.5, 0.5 sampling flag and adversaries (one red defends real red flag and other red attacks blue flag), agents learn a search and coverage problem.
        # Evaluate transfer on p=1,0 and 0,1. Worst-case performance of average ensemble and robust ensemble.
        # Solve true large-scale dec-POSMGs later... Contrast to other game-theory aware methods in literature later.
        # Design an experiment graph -- need not be a RRG with a max_degree (if yes, the max_degree = 5 is sufficient for paper purposes)...
        # Red spawns with the same distribution irrespective of the flag location. Blue can only observe the flag from a certain vantage point and it has to learn to find paths via the vantage point before commiting to a trajectory.

        ### GENERATE RED FLAG LOCATIONS AND VISUALIZE K-HOP VISIBILITY NEIGHBOURHOODS... AS AN INTUITION TO SET UP THE EXPERIMENT...
        ### Then see if your Rob-PSRO makes sense in this case... (given an opponent profile, compute adversarial task distribution and then best respond Blue to that... We don't do this fully, and in turn best respond to the average task distribution with robust distribution of the opponents... See if we gain anything in robustness to distribution shifts of the flag... Compute occupancy frequency of the common vantage point...) 
        ### Red seed behaviours -- guard common vantage point ( == INFORMATION FRONTIER) and guard flag...

        ### January 9, 2026: Once the flag is discovered by one agent in the neighbourhood -- it is instantly visible to all teammates.
        # Note on the above remark: for information discovery patterns of the above kind, credit assignment breaks for IPPO since an agent doesn't know how to assign credit properly. Perhaps for IPPO -- don't share information among agents and yes, share it for MAPPO (allows for more complex coordination).
        ### PURSUIT EVASION GAMES WITH PARTIAL VISIBILITY AND ASYMMETRIC INFORMATION ON RANDOM GEOMETRIC GRAPHS (RGGs is the key here).

        self.graph_gen_seed = kwargs.get('graph_gen_seed', 555) #kwargs.get('graph_gen_seed', 282)
        self.max_degree = kwargs.get('max_degree', 5) #kwargs.get('max_degree', 8)
        self.graph = None
        self.gen_graph(random_graph_family='basin-cooridoor') #self.gen_graph(random_graph_family='random-geometric', max_degree=self.max_degree)
        self.red_flag_node, self.blue_flag_node = None, None
        
        self.r = 1.05
        self.partial_visibility = True
        self.partial_visibility_radius = 1.5 #3*self.r #2.5
        self.partial_visibility_max_hops = 3
        self.ego_graph_max_hops = kwargs.get('ego_graph_max_hops', 3)
        self.partial_visibility_mode = 'radius'
        self.tagging_mode = 'radius'
        assert self.partial_visibility_mode in ['radius', 'k-hop']
        assert self.tagging_mode in ['radius', 'k-hop']
        self.tagging_radius = 1.25*self.r # [1., 1.05, 1.15] are all valid parameters.
        self.tagging_num_hops = 1
        self.flag_visibility_max_hops = 4 #March 6, 2026 for new_map. #7 ## Remark on 02/11/2026: Fine-tuned for the basin-coridoor graph for our paper (seed=2).

        self.reset_spawn_radius = 2*self.r
        self.reset_spawn_max_hops = 2 #### Rethink this. 02/11/2026 !!!!!!!!!!!!!!!

        self.global_embedding_cache = np.zeros((self.num_nodes, 6))
        self.ego_graph_cache = {node_idx: None for node_idx in range(self.num_nodes)}
        self.gen_graph_artifacts() # Max degree = 5. #### Verify this. 02/11/2026

        if seed is not None:
            self.seed(seed=seed)
        else:
            self.seed(seed=0)
            #self._seed = None # dummy value.
            #self.np_random = None
            #self.torch_rng = None

        self.num_teams = 2
        if ctf_player_config == "1v1":
            self.num_agents_blue_team = 1
        elif ctf_player_config == "2v2":
            self.num_agents_blue_team = 2
        self.num_agents_red_team = self.num_agents_blue_team

        self.blue_team_agents = ["Blue_" + "{}".format(i) for i in range(self.num_agents_blue_team)]
        self.red_team_agents = ["Red_" + "{}".format(i) for i in range(self.num_agents_red_team)]
        self.team_to_idx_dict = {'Blue': -2, 'B': -2, 'Red': +2, 'R': +2}
        self.idx_to_team_dict = {-2: 'Blue', +2: 'Red'}

        self.agents = copy.deepcopy(self.blue_team_agents + self.red_team_agents)
        if self.verbose: print("self.agents: {}".format(self.agents))
        self.possible_agents = copy.deepcopy(self.agents)
        if self.verbose: print(self.possible_agents)

        # Deferred opponent policy initialization (needs agents to be defined first)
        if self._deferred_opp_policy_init:
            self.set_opponent_policy(self._opp_policy, self._active_team)

        self.num_agents = self.num_agents()
        self.state = {agent: None for agent in self.agents}
        self.current_step = 0

        self.obs_type = 'full-graph'
        assert self.obs_type in ['full-graph', 'sub-graph']
        self.num_actions = self.max_degree + 2
        self.observation_spaces = {agent: self.observation_space(agent=agent) for agent in self.agents}
        self.action_spaces = {agent: self.action_space(agent=agent) for agent in self.agents}

        self.F = +1 #+100 # Reward on flag capture (zero-sum).
        # Shaped rewards (optional, pass via kwargs to override; set to 0 to disable)
        self.T = kwargs.get('tag_reward', 0.1)  # Tagging penalty/reward (shaped, small relative to F).
        self.flag_approach_shaping = kwargs.get('flag_approach_shaping', 0.01)  # Per-step reward for reducing distance to flag.
        self.tag_avoidance_shaping = kwargs.get('tag_avoidance_shaping', 0.005)  # Per-step penalty for being close to opponents.
        self.frontier_approach_shaping = kwargs.get('frontier_approach_shaping', 0)  # Per-step reward for reducing distance to nearest frontier (useful for stage 0).
        self.home_defense_shaping = kwargs.get('home_defense_shaping', 0.0)  # Per-step reward for Blue when ≤4 hops from home flag while any Red is ≤6 hops from it.
        self.defense_zone_tag_bonus = kwargs.get('defense_zone_tag_bonus', 0.0)  # One-time bonus for Blue when it tags a Red within ≤6 hops of Blue's home flag.
        self.defense_pursuit_shaping = kwargs.get('defense_pursuit_shaping', 0.0)
        # Per-step reward for Blue when it closes distance to nearest Red while Red is
        # ≤6 hops from the Blue flag. Teaches active interception rather than camping.
        self.global_embedding = None

        self._init_args = ()
        self._init_kwargs = {"ctf_player_config":ctf_player_config, "max_num_cycles":max_num_cycles, "verbose":verbose, "seed":seed,
                             "tag_reward":self.T, "flag_approach_shaping":self.flag_approach_shaping, "tag_avoidance_shaping":self.tag_avoidance_shaping,
                             "frontier_approach_shaping":self.frontier_approach_shaping, "home_defense_shaping":self.home_defense_shaping,
                             "defense_zone_tag_bonus":self.defense_zone_tag_bonus,
                             "defense_pursuit_shaping":self.defense_pursuit_shaping,
                             "reveal_flag":self.reveal_flag}

    def __call__(self):
        return GraphCTF(*self._init_args, **self._init_kwargs)

    def seed(self, seed=None):
        self.rngs = make_seeded_rngs(seed)
        self._seed = self.rngs["actual_seed"]
        self.np_random = self.rngs["np_random"]
        self.torch_rng = self.rngs["torch_rng"]
        return [self._seed]

    def valid_actions(self, node_idx):
        # node = (node_idx, node_attr)
        # Given a node, return valid actions by returning neighboring nodes.
        # Summary: action_mask(node) = [neighboring_nodes].
        G = copy.deepcopy(self.graph)
        valid_actions = NotImplemented
        return valid_actions

    def gen_graph(self,
                  random_graph_family='random-geometric',
                  graph_gen_seed=None,
                  max_degree=8
        ):
        """
        Generate a random geometric graph using NetworkX with optional max degree
        and guaranteed single connected component.
        """
        import networkx as nx
        assert random_graph_family in ['random-geometric', 'erdos-renyi', 'basin-cooridoor']

        if graph_gen_seed is None: graph_gen_seed = self.graph_gen_seed
        np.random.seed(graph_gen_seed)

        if random_graph_family == 'random-geometric':
            scene_x_lim = [0, 10]
            scene_y_lim = [0, 10]
            num_points = 250

            x = np.random.uniform(*scene_x_lim, num_points)
            y = np.random.uniform(*scene_y_lim, num_points)
            import matplotlib.pyplot as plt
            plt.plot(x, y, '.')
            pts = [(xx, yy) for xx, yy in zip(x, y)]

            edges = []
            r = 1.05 #1.8
            for i in range(num_points):
                for j in range(num_points):
                    if i == j: continue
                    if np.linalg.norm(np.array(pts[i]) - np.array(pts[j])) <= r:
                        if (i, j) in edges or (j, i) in edges: continue
                        edges.append((i, j))
                        plt.plot([x[i], x[j]], [y[i], y[j]], color='black', linewidth=0.5)

            G = nx.Graph()
            # Add nodes with positions
            for i, pos in enumerate(pts):
                G.add_node(i, pos=pos)

            # Add edges based on distance
            for i in range(num_points):
                for j in range(i + 1, num_points):
                    if np.linalg.norm(np.array(pts[i]) - np.array(pts[j])) <= r:
                        G.add_edge(i, j)

            nodes = list(range(len(pts)))
            node_feats = pts
            edge_list = edges
            print('num of edges: {}'.format(len(edges)))

            # Optionally prune edges to enforce max degree
            if max_degree is not None:
                for node in G.nodes:
                    neighbors = list(G.neighbors(node))
                    if len(neighbors) > max_degree:
                        # Keep only closest neighbors
                        neighbors.sort(key=lambda j: np.linalg.norm(np.array(pts[node]) - np.array(pts[j])))
                        for neighbor in neighbors[max_degree:]:
                            if G.has_edge(node, neighbor):
                                G.remove_edge(node, neighbor)

            # Ensure a single connected component
            components = list(nx.connected_components(G))
            if len(components) > 1:
                print(f"{len(components)} disconnected components detected. Connecting them...")
                main_comp = components[0]
                for comp in components[1:]:
                    # Find closest pair of nodes between main component and this component
                    min_dist = float('inf')
                    closest_pair = None
                    for u in main_comp:
                        for v in comp:
                            dist = np.linalg.norm(np.array(pts[u]) - np.array(pts[v]))
                            if dist < min_dist:
                                min_dist = dist
                                closest_pair = (u, v)
                    # Connect the components
                    G.add_edge(*closest_pair)
                    main_comp = main_comp.union(comp)
            
            
        elif random_graph_family == 'basin-cooridoor':
            bc = self.make_basin_corridor_medium_v1()
            G = bc.G
            self._bc = bc  # preserve basin-corridor metadata for gen_graph_artifacts
            #self._bc_info = build_bc_info(self._bc) # train_blue_curicullum.py has build_bc_info() which uses Structs from basin_corridor_policies.py

        for i, node in enumerate(G.nodes):
            G.nodes[node]['node_idx'] = i # Storing node_idx as a feature...
        self.graph = copy.deepcopy(G)
        self.num_nodes = len(G.nodes)
        self.node_pose_dict = copy.deepcopy(nx.get_node_attributes(G, 'pos'))
        self.node_to_idx = {node: G.nodes[node]['node_idx'] for node in G.nodes()} # Networkx node to index (ordering of Networkx nodes fixed by the lines ~ 469-470 above)
        self.idx_to_node = {node_idx: node for node, node_idx in self.node_to_idx.items()} # reverse mapping of index to Networkx nodes

        # Saving a canonical ordering over edges...
        undirected = True
        edges = []
        for u, v in G.edges():
            i, j = self.node_to_idx[u], self.node_to_idx[v]
            edges.append((i, j))
            if undirected:
                edges.append((j, i))
        edges = sorted(edges)
        self.edges = edges
        
        ## Compute edge_index for the global graph topology...
        self.edge_index = np.array(edges, dtype=np.int64).T

        # Generate neighbours...
        self.neighbours = {idx: [] for idx in range(self.num_nodes)}
        for node_idx in range(self.num_nodes):
            node = self.idx_to_node[node_idx]
            neighbour_idxs = [self.node_to_idx[ngbr] for ngbr in G.neighbors(node)]
            neighbour_idxs = sorted(neighbour_idxs)
            self.neighbours[node_idx] = neighbour_idxs
        
        # Remark: self.edges and self.edge_index are consistent in their ordering... Later helpful for constructing edge_attr()
        self.edge_features = self.construct_edge_attr()

        # GRAPH PROPERTIES...
        self.degrees = G.degree()
        self.max_degree_actual = max(d for _, d in G.degree())
        self.max_degree = self.max_degree_actual
        self.graph_diameter = nx.diameter(G)
        return

    # This is for Basin-Cooridoor Graph...
    @staticmethod
    def _grid_idx(i: int, j: int, ny: int) -> int:
        return i * ny + j

    # This is for Basin-Cooridoor Graph...
    @staticmethod
    def _grid_ij(u: int, ny: int) -> Tuple[int, int]:
        return (u // ny, u % ny)

    # This is for Basin-Cooridoor Graph...
    @staticmethod
    def _bfs_multi_source_dist(G: nx.Graph, sources: Iterable[int]) -> Dict[int, int]:
        """Multi-source unweighted shortest-path distances on an undirected graph."""
        sources = list(sources)
        dist: Dict[int, int] = {}
        dq = deque()
        for s in sources:
            if s in G:
                dist[s] = 0
                dq.append(s)
        while dq:
            u = dq.popleft()
            du = dist[u]
            for v in G.neighbors(u):
                if v not in dist:
                    dist[v] = du + 1
                    dq.append(v)
        return dist

    # This is for Basin-Cooridoor Graph...
    @staticmethod
    def _closest_node_in_G(G: nx.Graph, ny: int, target_ij: Tuple[int, int]) -> int:
        """Pick node in G whose underlying grid coords are closest (L2 in ij space) to target."""
        ti, tj = target_ij
        best_u, best_d = None, 10**18
        for u in G.nodes():
            ui, uj = GraphCTF._grid_ij(u, ny)
            d = (ui - ti) ** 2 + (uj - tj) ** 2
            if d < best_d:
                best_d, best_u = d, u
        assert best_u is not None
        return best_u
    
    # This is for Basin-Cooridoor Graph...
    def make_basin_corridor_medium(
        self,
        seed: int = 101,
        nx_dim: int = 18,
        ny_dim: int = 18,
        rho: float = 0.78,
        ell: float = 4.0,
        jitter: float = 0.35,
    ) -> BasinCorridorGraph:
        """
        Medium instance generator.

        Returns:
        - G: largest connected component after correlated node removal
        - pos: jittered coordinates for plotting
        - red_flag_L / red_flag_R: representative nodes near top-left / top-right
        - basin + bottom regions (for convenience)
        """
        rng = np.random.default_rng(seed)
        N = nx_dim * ny_dim

        # Jittered positions (geometry only; does NOT affect connectivity)
        pos: Dict[int, np.ndarray] = {}
        for i in range(nx_dim):
            for j in range(ny_dim):
                u = GraphCTF._grid_idx(i, j, ny_dim)
                pos[u] = np.array([j + jitter * rng.normal(), -i + jitter * rng.normal()], dtype=float)

        # Grid edges (4-neighborhood)
        edges: List[Tuple[int, int]] = []
        for i in range(nx_dim):
            for j in range(ny_dim):
                u = GraphCTF._grid_idx(i, j, ny_dim)
                if i + 1 < nx_dim:
                    edges.append((u, GraphCTF._grid_idx(i + 1, j, ny_dim)))
                if j + 1 < ny_dim:
                    edges.append((u, GraphCTF._grid_idx(i, j + 1, ny_dim)))

        # Correlated random field on the grid to select "open" nodes
        coords = np.array([[i, j] for i in range(nx_dim) for j in range(ny_dim)], dtype=float)
        D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
        Sigma = np.exp(-D / ell) + 1e-6 * np.eye(N)
        L = np.linalg.cholesky(Sigma)
        z = L @ rng.standard_normal(N)

        tau = np.quantile(z, 1 - rho)
        open_nodes = {u for u in range(N) if z[u] >= tau}

        # Induced subgraph + keep largest connected component
        G = nx.Graph()
        #G.add_nodes_from(open_nodes)
        for u in open_nodes:
            G.add_node(u, pos=pos[u])

        for u, v in edges:
            if u in open_nodes and v in open_nodes:
                G.add_edge(u, v)

        if G.number_of_nodes() == 0:
            raise RuntimeError("No open nodes; adjust rho/ell/seed.")

        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

        # Flag hypothesis nodes near top-left and top-right
        red_flag_L = GraphCTF._closest_node_in_G(G, ny_dim, (0, 1))
        red_flag_R = GraphCTF._closest_node_in_G(G, ny_dim, (0, ny_dim - 2))

        # Convenience regions
        top_left_basin_targets = [(0, 0), (0, 1), (1, 0), (1, 1)]
        top_right_basin_targets = [(0, ny_dim - 2), (0, ny_dim - 1), (1, ny_dim - 2), (1, ny_dim - 1)]
        top_left_basin = list({ GraphCTF._closest_node_in_G(G, ny_dim, t) for t in top_left_basin_targets })
        top_right_basin = list({ GraphCTF._closest_node_in_G(G, ny_dim, t) for t in top_right_basin_targets })

        # bottom region = nodes whose underlying grid-row is in last 2 rows
        bottom_region = [u for u in G.nodes() if GraphCTF._grid_ij(u, ny_dim)[0] >= nx_dim - 2]

        return BasinCorridorGraph(
            G=G,
            pos=pos,
            red_flag_L=red_flag_L,
            red_flag_R=red_flag_R,
            blue_flag=None,
            top_left_basin=top_left_basin,
            top_right_basin=top_right_basin,
            bottom_region=bottom_region,
        )
    
    def make_basin_corridor_medium_v1(
        self,
        seed: int = 30597,
        nx_dim: int = 11,
        ny_dim: int = 11,
        rho: float = 0.80,
        ell: float = 3.2,
        jitter: float = 0.35,
        choke_row_start: int = 5, # rows [start, end] form the wall
        choke_row_end: int = 7, 
        passage_cols: set = {0, 1, 2, 3, 11-4, 11-3, 11-2, 11-1}, # columns kept open through the choke
        funnel_rows: int = 2, # rows above/below choke forced open on passage cols
    ) -> BasinCorridorGraph:
        
        rng = np.random.default_rng(seed)
        N   = nx_dim * ny_dim
        
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
        G = G.subgraph(largest_cc).copy()

        # Passage nodes...
        passage_nodes = [u for u in G.nodes()
                 if choke_row_start <= (u // ny_dim) <= choke_row_end]
        
        # Flag hypothesis nodes near top-left and top-right
        red_flag_L = 24
        red_flag_R = 29

        blue_flag = None
        bottom_basin_anchor_node = 104 # proxy for blue_flag

        # Top basin region computation...
        top_left_basin_ = set(nx.single_source_shortest_path_length(G, red_flag_L, cutoff=4).keys())
        top_left_basin = copy.deepcopy(top_left_basin_)
        for node in top_left_basin_:
            if node in passage_nodes:
                top_left_basin.remove(node)
        top_left_basin = list(top_left_basin)

        top_right_basin_ = set(nx.single_source_shortest_path_length(G, red_flag_R, cutoff=4).keys())
        top_right_basin = copy.deepcopy(top_right_basin_)
        for node in top_right_basin_:
            if node in passage_nodes:
                top_right_basin.remove(node)
        top_right_basin = list(top_right_basin)

        # Bottom basin region computation...
        bottom_region = set(nx.single_source_shortest_path_length(G, bottom_basin_anchor_node, cutoff=3).keys())
        bottom_region = list(bottom_region)

        return BasinCorridorGraph(
            G=G,
            pos=pos,
            red_flag_L=red_flag_L,
            red_flag_R=red_flag_R,
            blue_flag=bottom_basin_anchor_node,
            top_left_basin=top_left_basin,
            top_right_basin=top_right_basin,
            bottom_region=bottom_region,
        )

    def construct_edge_attr(self):
        # Populating edge_attr in the same consistent order...
        G = copy.deepcopy(self.graph)
        node_pos_dict = nx.get_node_attributes(G, 'pos')
        edge_features = []
        for canonical_u, canonical_v in self.edges:
            u, v = self.idx_to_node[canonical_u], self.idx_to_node[canonical_v]
            edge_ft_vec = np.array(node_pos_dict[v]) - np.array(node_pos_dict[u])
            edge_ft_dir = edge_ft_vec / np.linalg.norm(edge_ft_vec)
            edge_ft_len = np.linalg.norm(edge_ft_vec)
            edge_ft_uv = np.concat([edge_ft_dir, [edge_ft_len]])
            edge_features.append(edge_ft_uv)
        edge_features = np.array(edge_features, dtype=np.float32)
        return edge_features
        
    def gen_graph_artifacts(self):
        # Add Red Flag.
        # Add Blue Flag.

        G = copy.deepcopy(self.graph)
        node_pos_dict = nx.get_node_attributes(G, "pos")

        if hasattr(self, '_bc') and self._bc is not None:
            # Basin-corridor graph: use pre-computed flag nodes from make_basin_corridor_medium
            self.red_flag_node = self._bc.red_flag_L           # top-left basin
            self.red_flag_2_node = self._bc.red_flag_R         # top-right basin
            
            hardcoded_map = True
            if hardcoded_map:
                self.blue_flag_node = self._bc.blue_flag
            else:
                blue_target_ij = (10, 5)                            # bottom-center of 11x11 grid
                self.blue_flag_node = GraphCTF._closest_node_in_G(G, 11, blue_target_ij)

            self.red_flag_pos = np.array(node_pos_dict[self.red_flag_node])
            self.red_flag_2_pos = np.array(node_pos_dict[self.red_flag_2_node])
            self.blue_flag_pos = np.array(node_pos_dict[self.blue_flag_node])
        else:
            # Random-geometric graphs: sample flag positions and snap to closest node
            red_flag_x_lim_left = [2,3] #[1, 2] #[0, 2]
            red_flag_x_lim_right = [7,8] #[8, 9] #[8, 10]

            red_flag_x_lim, red_flag_y_lim = red_flag_x_lim_left, [9, 10]
            red_flag_x_lim_2 = red_flag_x_lim_right
            blue_flag_x_lim, blue_flag_y_lim = [4, 6], [0, 1] # Blue Flag Center.
            np.random.seed(seed=self.graph_gen_seed+1)

            red_flag_pos_sampled = (np.random.uniform(*red_flag_x_lim), np.random.uniform(*red_flag_y_lim))
            red_flag_pos_sampled_2 = (np.random.uniform(*red_flag_x_lim_2), np.random.uniform(*red_flag_y_lim))
            blue_flag_pos_sampled = (np.random.uniform(*blue_flag_x_lim), np.random.uniform(*blue_flag_y_lim))

            # Closest node to sampled red flag pos...
            red_flag_node_idx = min(
                node_pos_dict.items(),
                key=lambda item: np.linalg.norm(np.array(item[1]) - red_flag_pos_sampled)
            )[0]
            self.red_flag_node = red_flag_node_idx
            self.red_flag_pos = np.array(node_pos_dict[self.red_flag_node])

            red_flag_2_node_idx = min(
                node_pos_dict.items(),
                key=lambda item: np.linalg.norm(np.array(item[1]) - red_flag_pos_sampled_2)
            )[0]
            self.red_flag_2_node = red_flag_2_node_idx
            self.red_flag_2_pos = np.array(node_pos_dict[self.red_flag_2_node])

            # Closest node to sampled blue flag pos...
            blue_flag_node_idx = min(
                node_pos_dict.items(),
                key=lambda item: np.linalg.norm(np.array(item[1]) - blue_flag_pos_sampled)
            )[0]
            self.blue_flag_node = blue_flag_node_idx
            self.blue_flag_pos = np.array(node_pos_dict[self.blue_flag_node])

        # Computing valid spawn nodes for Red and Blue...
        red_flag_pos = np.array(node_pos_dict[self.red_flag_node])
        red_flag_2_pos = np.array(node_pos_dict[self.red_flag_2_node])
        blue_flag_pos = np.array(node_pos_dict[self.blue_flag_node])
        
        self.red_spawn_valid_nodes_L = [
            node for node, pos in node_pos_dict.items()
            if np.linalg.norm(np.array(pos) - red_flag_pos) <= self.reset_spawn_radius
        ]

        assert hasattr(self, '_bc') and self._bc is not None
        self.red_spawn_valid_nodes_L = self._bc.top_left_basin
        self.red_spawn_valid_nodes_R = self._bc.top_right_basin

        self.red_spawn_valid_nodes_just_L = copy.deepcopy(set(self.red_spawn_valid_nodes_L)) - copy.deepcopy(set(self.red_spawn_valid_nodes_R))
        self.red_spawn_valid_nodes_just_L = list(self.red_spawn_valid_nodes_just_L)
        self.red_spawn_valid_nodes_just_R = copy.deepcopy(set(self.red_spawn_valid_nodes_R)) - copy.deepcopy(set(self.red_spawn_valid_nodes_L))
        self.red_spawn_valid_nodes_just_R = list(self.red_spawn_valid_nodes_just_R)
        
        # Following commented on March 6, 2026 (after the new map with Red defense only and no Blue flag).
        """
        self.red_spawn_valid_nodes_L = [
            node for node, pos in node_pos_dict.items()
            if np.linalg.norm(np.array(pos) - red_flag_pos) <= self.reset_spawn_radius
        ]

        self.red_spawn_valid_nodes_R = [
            node for node, pos in node_pos_dict.items()
            if np.linalg.norm(np.array(pos) - red_flag_2_pos) <= self.reset_spawn_radius
        ]

        # Default Red spawn: top 1/3 of y, middle 1/3 of x.
        # This places Red in the corridor between the two basins, symmetric
        # w.r.t. both flag hypotheses, and away from the bottom (Blue) region.
        all_pos = np.array([node_pos_dict[n] for n in G.nodes()])
        x_min, x_max = all_pos[:, 0].min(), all_pos[:, 0].max()
        y_min, y_max = all_pos[:, 1].min(), all_pos[:, 1].max()
        x_lo = x_min + (x_max - x_min) / 3
        x_hi = x_min + 2 * (x_max - x_min) / 3
        y_top_thresh = y_min + 2 * (y_max - y_min) / 3  # top 1/3
        self.red_spawn_valid_nodes = [
            n for n in G.nodes()
            if x_lo <= node_pos_dict[n][0] <= x_hi
            and node_pos_dict[n][1] >= y_top_thresh
        ]
        assert len(self.red_spawn_valid_nodes) >= 2, \
            f"Red spawn set too small ({len(self.red_spawn_valid_nodes)} nodes). " \
            f"Check graph layout — need nodes in top-1/3 y, mid-1/3 x."

        self.blue_spawn_valid_nodes = [
            node for node, pos in node_pos_dict.items()
            if np.linalg.norm(np.array(pos) - blue_flag_pos) <= self.reset_spawn_radius
        ]
        """
        self.red_spawn_valid_nodes = set(self.red_spawn_valid_nodes_L) | set(self.red_spawn_valid_nodes_R)
        self.red_spawn_valid_nodes = list(self.red_spawn_valid_nodes)
        self.blue_spawn_valid_nodes = list(nx.single_source_shortest_path_length(G, self.blue_flag_node, cutoff=3).keys())

        # GENERATE FRONTIER NODES FOR ALL FLAG HYPOTHESIS...
        # Compute frontier_nodes and frontier_resolution_mass for node in frontier_nodes...
        self.red_flag_hypothesis = [self.red_flag_node, self.red_flag_2_node]
        flag_true = 0 # 0 for left, and 1 for right...
        self.red_flag_true = self.red_flag_hypothesis[flag_true] #### NOTE: Added on January 2026: update later to support input true_red_flag in environment __init__. Need it for flag sampling and adversarial training.
        num_flags = len(self.red_flag_hypothesis)
        
        self.red_flag_reach_sets = [set(self.red_spawn_valid_nodes_L), set(self.red_spawn_valid_nodes_R)]
        flag_reach_dict = {
            flag: self.red_flag_reach_sets[iter_] #set(nx.single_source_shortest_path_length(G, flag, cutoff=self.flag_visibility_max_hops))
            for iter_, flag in enumerate(self.red_flag_hypothesis)
        } # for each flag as key, returns dict of nodes that are backward reachable from the flag.
        self.flag_reach_dict = copy.deepcopy(flag_reach_dict)

        frontier_nodes = set()
        for nodes in flag_reach_dict.values():
            frontier_nodes |= nodes

        is_flag_reachable = lambda node, flag: node in flag_reach_dict[flag] #### NOTE: Added on January 2026: More than flag_reachable, it should be flag_visible.
        self.is_true_flag_reachable = copy.deepcopy(flag_reach_dict[self.red_flag_true])

        self.is_frontier = {node: False for node in G.nodes()}
        self.frontier_resolution_mass = {node: 0 for node in G.nodes()}
        for node in frontier_nodes:
            self.is_frontier[node] = True
            self.frontier_resolution_mass[node] = sum([is_flag_reachable(node, flag) for flag in self.red_flag_hypothesis]) / num_flags

        # Precompute distance from every node to its nearest frontier node (multi-source BFS)
        self.dist_to_nearest_frontier = {node: float('inf') for node in G.nodes()}
        for fn in frontier_nodes:
            for node, d in nx.single_source_shortest_path_length(G, fn).items():
                self.dist_to_nearest_frontier[node] = min(self.dist_to_nearest_frontier[node], d)

        # Precompute distance from every node to its nearest Red flag node (multi-source BFS)
        self.dist_to_nearest_red_flag = {node: float('inf') for node in G.nodes()}
        self.nearest_red_flag = {node: None for node in G.nodes()} # 0 or 1.
        for flag_idx, flag_node in enumerate(self.red_flag_hypothesis):
            for node, d in nx.single_source_shortest_path_length(G, flag_node).items():
                if d < self.dist_to_nearest_red_flag[node]:
                    self.dist_to_nearest_red_flag[node] = d
                    self.nearest_red_flag[node] = flag_idx

        # Precompute BFS distance from every node to the Blue flag (used by defense_zone_tag_bonus).
        self._dist_from_blue_flag = dict(nx.single_source_shortest_path_length(G, self.blue_flag_node))

        # Generate Global Embedding Cache...
        self.gen_global_embedding_cache()
        # Generate Ego Graph Cache...
        self.gen_ego_graph_cache()
        return
        
    def save_graph_scene(self, data_file_name):
        scene_data = {
            'nx_graph': None,
            'flag_locs': None,
        }
        import pickle
        scene_data_save_path = 'graphs/{}.pkl'.format(data_file_name)
        return

    # =========================================================================
    # OPPONENT POLICY INTEGRATION (bypasses GraphCoopEnv wrapper for speed)
    # =========================================================================
    def set_opponent_policy(self, policy, active_team='Blue'):
        """
        Set opponent policy directly in GraphCTF to avoid GraphCoopEnv wrapper overhead.

        Args:
            policy: Callable that takes observation dict and returns action
            active_team: 'Blue' or 'Red' - the team being trained (opponent team uses fixed policy)

        Usage:
            env = GraphCTF(...)
            env.set_opponent_policy(RedGraphPolicy, active_team='Blue')
            # Now env.step() only needs actions for Blue team
            # Red team actions are computed internally
        """
        assert active_team in ['Blue', 'Red'], "active_team must be 'Blue' or 'Red'"
        self._opp_policy = policy
        self._active_team = active_team

        if active_team == 'Blue':
            self._active_agents = self.blue_team_agents
            self._passive_agents = self.red_team_agents
        else:
            self._active_agents = self.red_team_agents
            self._passive_agents = self.blue_team_agents

    def has_opponent_policy(self):
        """Check if opponent policy is configured."""
        return self._opp_policy is not None and self._active_team is not None

    def get_active_agents(self):
        """Get list of agents controlled by the learner (not the fixed policy)."""
        if self.has_opponent_policy():
            return [a for a in self.agents if a in self._active_agents]
        return self.agents

    def get_passive_agents(self):
        """Get list of agents controlled by the fixed opponent policy."""
        if self.has_opponent_policy():
            return [a for a in self.agents if a in self._passive_agents]
        return []

    def _compute_opponent_actions(self):
        """Compute actions for passive agents using the opponent policy."""
        if not self.has_opponent_policy():
            return {}

        passive_agents = self.get_passive_agents()
        if not passive_agents:
            return {}

        # Batch collect observations
        obs_list = []
        for agent in passive_agents:
            if self.obs_version == 3:
                obs_list.append(self.get_observation_v3(agent))
            elif self.obs_version == 2:
                obs_list.append(self.get_observation_v2(agent))
            else:
                obs_list.append(self.get_observation_v1(agent))

        # Batch compute actions if supported
        if hasattr(self._opp_policy, 'batch_action'):
            actions = self._opp_policy.batch_action(obs_list)
        else:
            actions = [self._opp_policy(obs) for obs in obs_list]

        return {agent: action for agent, action in zip(passive_agents, actions)}
    
    def reset(self, seed=None, options=None):
        # Add Red and Blue team agent initial positions -> reset().
        if seed is not None:
            self.seed(seed=seed)

        # --- Per-episode red flag sampling ---
        if self.fixed_flag_hypothesis is not None:
            flag_idx = self.fixed_flag_hypothesis
        elif self._flag_probs is not None:
            flag_idx = int(self.np_random.choice(len(self.red_flag_hypothesis),
                                                  p=self._flag_probs))
        else:
            flag_idx = int(self.np_random.integers(len(self.red_flag_hypothesis)))
        self.red_flag_true = self.red_flag_hypothesis[flag_idx]
        self._episode_flag_idx = flag_idx  # §41/§42: read by step_wait for per-flag reward routing
        self.is_true_flag_reachable = copy.deepcopy(self.flag_reach_dict[self.red_flag_true])

        # Reset opponent policy FSM state if it has one.
        # Pass flag_idx so flag-conditioned opponents (MultiLearnedOpponent) can
        # sample from the matching per-flag weight vector.  Policies whose
        # reset() doesn't accept flag_idx (FSM scripted policies) are called
        # without it via the TypeError fallback.
        if hasattr(self, '_opp_policy') and self._opp_policy is not None and hasattr(self._opp_policy, 'reset'):
            try:
                self._opp_policy.reset(flag_idx=flag_idx)
            except TypeError:
                self._opp_policy.reset()

        self.agents = copy.deepcopy(self.blue_team_agents + self.red_team_agents)
        # Initialize Red Agents...
        self.red_init_pos = np.random.choice(self.red_spawn_valid_nodes, size=self.num_agents_red_team, replace=False)
        for agent_iter, agent in enumerate(self.red_team_agents):
            self.state[agent] = self.red_init_pos[agent_iter]
        
        # Initialize Blue Agents...
        self.blue_init_pos = np.random.choice(self.blue_spawn_valid_nodes, size=self.num_agents_blue_team, replace=False)
        for agent_iter, agent in enumerate(self.blue_team_agents):
            self.state[agent] = self.blue_init_pos[agent_iter]

        # Initialize per-agent distance to flag for reward shaping
        self._prev_dist_to_flag = {}
        self._prev_dist_to_frontier = {}
        for agent in self.agents:
            opp_flag = self.red_flag_true if agent[0] == 'B' else self.blue_flag_node
            try:
                self._prev_dist_to_flag[agent] = nx.shortest_path_length(
                    self.graph, self.state[agent], opp_flag
                )
            except nx.NetworkXNoPath:
                self._prev_dist_to_flag[agent] = self.graph_diameter
            self._prev_dist_to_frontier[agent] = self.dist_to_nearest_frontier.get(self.state[agent], self.graph_diameter)

        self._prev_dist_to_nearest_red = {}
        if self.defense_pursuit_shaping:
            red_agents_at_reset = [a for a in self.agents if a[0] == 'R']
            for agent in self.agents:
                if agent[0] == 'B':
                    if red_agents_at_reset:
                        try:
                            d = min(
                                nx.shortest_path_length(self.graph, self.state[agent], self.state[r])
                                for r in red_agents_at_reset
                            )
                        except nx.NetworkXNoPath:
                            d = self.graph_diameter
                    else:
                        d = self.graph_diameter
                    self._prev_dist_to_nearest_red[agent] = d

        # Obtain obs and info dicts for all agents...
        info = {agent: {} for agent in self.agents}
        obs = {agent: None for agent in self.agents}

        ########################
        self.enemy_flag_known = {
            agent: False if agent.startswith('B') else True
            for agent in self.agents
        } # Used in self.get_observation_v1() and hence self.step().....
        self.red_flag_resolved = self.reveal_flag  # If reveal_flag, Blue knows flag from step 0

        if self.reveal_flag:
            for agent in self.agents:
                if agent.startswith('B'):
                    self.enemy_flag_known[agent] = True
        else:
            for agent in self.agents:
                """
                if agent.startswith('B') and (self.state[agent] in self.is_true_flag_reachable):
                    self.red_flag_resolved = True
                    self.enemy_flag_known[agent] = True ### NOTE: Info sharing between teammates. CRITICAL POINT: public belief within the team. Approximates Team Markov Equilibria. Otherwise private belief, more like Dec-POMDP like Equilibria.
                """
                if agent.startswith('B') and self.is_frontier[self.state[agent]]:
                    self.red_flag_resolved = True # REMARK: This works only for 2 flags since seeing one flag's frontier automatically proves the other to be true... Doesn't work for three flags...
                    self.enemy_flag_known[agent] = True ### NOTE: Info sharing between teammates. CRITICAL POINT: public belief within the team. Approximates Team Markov Equilibria. Otherwise private belief, more like Dec-POMDP like Equilibria.

            # This loop propagates information across the entire Blue team (if any one Blue agent resolves the Red flag, it becomes known to the entire Blue team)...
            for agent in self.agents:
                if agent.startswith('B'):
                    self.enemy_flag_known[agent] = self.red_flag_resolved

        self.all_agent_pairings = [] # Used in self.get_observation_v1() and hence self.step(), to generate min_opp_distance and min_teammate_distance...
        for i, agent in enumerate(self.agents):
            for j, agent_partner in enumerate(self.agents):
                if j <= i: continue
                self.all_agent_pairings.append((agent, agent_partner))
        
        self.min_teammate_distance = {
            agent: 1 for agent in self.agents
        }
        self.min_opp_distance = {
            agent: 1 for agent in self.agents
        }
        
        dist = {}
        for agent_0, agent_1 in self.all_agent_pairings:
            # dist[(agent_0, agent_1)]
            distance = nx.shortest_path_length(self.graph, source=self.state[agent_0], target=self.state[agent_1]) / self.graph_diameter
            dist[(agent_0, agent_1)] = distance
            if agent_0[0] == agent_1[0]:
                #Teammates.
                self.min_teammate_distance[agent_0] = min(distance, self.min_teammate_distance[agent_0]) #distance
                self.min_teammate_distance[agent_1] = min(distance, self.min_teammate_distance[agent_1])
            else:
                # Opponents.
                self.min_opp_distance[agent_0] = min(distance, self.min_opp_distance[agent_0])
                self.min_opp_distance[agent_1] = min(distance, self.min_opp_distance[agent_1])

        ########################
        """
        # Commented on January 26, 2026: Not needed for sub-graph observations.
        # Generate global embedding before computing agent obs...
        self.global_embedding = self.generate_global_embeddings()
        """
        for agent in self.agents:
            if self.obs_version == 3:
                obs[agent] = self.get_observation_v3(agent)
            elif self.obs_version == 2:
                obs[agent] = self.get_observation_v2(agent)
            else:
                obs[agent] = self.get_observation_v1(agent)
        self.current_step = 0  # Reset step counter (was incorrectly += 1)

        return obs, info

    def generate_global_embeddings(self):
        ## Dim1: [-1, 0, +1] ==> -1 is for adversary, 0 for unocuppied, +1 for ally, -2 for away_flag and +2 for home_flag
        ## Dim2: node pose coordinate (just for v0, for v1 we would want -> isomorphism between coordinate transformation, just graph topology and relevant geoemtric factors and not the coordinate system used ==> transfers less within the same team and maybe not at all to the enemy team because of a different coordinate reference ==> )
        ## Dim3: relative distance from home_flag (for v1 -> what's a coordinate isomorphic notion of distance in terms of number of hops et cetera)
        ## Dim4: relative pose of away flag (if visible) otherwise -inf. (Assume visible for v0 --> opponent flag location known)
        
        # Note: Call this after updating state, after every reset() and step().
        G = copy.deepcopy(self.graph)
        node_feats_global = np.zeros((self.num_nodes, 2))

        for node_idx in range(self.num_nodes):
            node = self.idx_to_node[node_idx]
            node_feat = []
            node_agent_occupancy_feature = 0
            for agent in self.agents:
                if self.state[agent] == node:
                    node_agent_occupancy_feature = self.team_to_idx_dict[agent[0]]
                    break
            node_feat.append(node_agent_occupancy_feature)
            
            node_flag_occupancy_feature = 0
            if self.red_flag_node == node: node_flag_occupancy_feature = self.team_to_idx_dict['R']
            elif self.blue_flag_node == node: node_flag_occupancy_feature = self.team_to_idx_dict['B']
            node_feat.append(node_flag_occupancy_feature)

            node_feats_global[node_idx] = node_feat

        node_feats_global = np.array(node_feats_global)
        return node_feats_global

    @staticmethod
    def observation_info_pattern(self):
        """
            [0]  is_ego -> dummy val for global embedding x_global
            [1]  is_frontier -> true val for global embedding x_global
            [2]  is_regular -> dummy val for global embedding x_global
            [3]  is_flag_discovered_here -> true val for global embedding x_global

            [4]  distance_to_ego -> dummy val for global embedding x_global

            [5]  distance_to_home_flag        (ego-only) -> true val for global embedding x_global (red and blue)
            [6]  home_flag_known              (ego-only) -> true val for global embedding x_global
                 OOOOOOOOO redundant info [6]
            [7]  distance_to_enemy_flag       (ego-only) -> dummy val for global embedding x_global
            [8]  enemy_flag_known             (ego-only) -> dummy val for global embedding x_global

            [10] min_enemy_distance_to_ego    (ego-only) -> true val for global embedding x_global: min_enemy distance to each node (both red and blue)
            [11] min_teammate_distance_to_ego (ego-only) -> true val for global embedding x_global: min_ally distance to each node (both red and blue)

            [12] structural_degree            (regular-only, optional but recommended) -> true val for global embedding x_global
            [13] visibility_bit -> dummy val for global embedding x_global

            [9]  frontier_unresolved         (frontier-only) -> dummy val for global embedding x_global -> for first implementation of this sub-graph observation, not relevant as graph is static / any frontier node gives the same observation.
                 XXXXXXXX
        """
        return
    
    @staticmethod
    def nx_graph_to_sparse_edge_representation(graph: nx.Graph, node_to_idx_map):
        # Saving a canonical ordering over edges...
        undirected = True
        edges = []
        for u, v in graph.edges():
            i, j = node_to_idx_map[u], node_to_idx_map[v]
            edges.append((i, j))
            if undirected:
                edges.append((j, i))
        edges = sorted(edges)
        edge_index = np.array(edges, dtype=np.int64).T
        return edge_index

    def gen_ego_graph_cache(self):
        """
        Plan: Generate ego-graph before: can generate frontier or not masks from before. Can generate all structural information like node_degree and even distance from ego
            ## Generate keys for the nodes and an edge_list...
            ## We need global keys to generate node embeddings, and we need a canonical ordering for the sub-graph observations...
        """
        # GENERATE and CACHE VISIBILITY NEIGHBORHOODS FOR ALL NODES...

        self.max_number_ego_nodes = 0
        self.max_number_ego_edges = 0
        
        for node_idx in range(self.num_nodes):
            # node_idx here refers to the ego_node.
            node = self.idx_to_node[node_idx]
            node_ego_graph = nx.ego_graph(self.graph, node, radius=self.ego_graph_max_hops)
            self.max_number_ego_nodes = max(self.max_number_ego_nodes, node_ego_graph.number_of_nodes())
            self.max_number_ego_edges = max(self.max_number_ego_edges, node_ego_graph.number_of_edges())

            node_idxs = [self.node_to_idx[node] for node in node_ego_graph.nodes()]
            node_idxs = sorted(node_idxs) # sorting nodes by their global canonical idxs...
            node_idx_local_map = {node: node_idxs.index(self.node_to_idx[node]) for node in node_ego_graph.nodes()} # u, v -> i_local, j_local
            local_idx_node_map = {local_node_idx: node for node, local_node_idx in node_idx_local_map.items()}
            ego_edge_index = GraphCTF.nx_graph_to_sparse_edge_representation(node_ego_graph, node_idx_local_map)
            
            #### PRE-COMPUTE DISTANCE TO EGO OF EACH SUBNODE HERE...
            hops_to_ego = nx.single_source_shortest_path_length(node_ego_graph, node)

            ### PRE-COMPUTE NEIGHBOR LOCAL INDICES AND MASK
            # self.neighbours[node_idx] contains sorted global neighbor indices
            # We need to convert these to local indices in the subgraph
            global_neighbor_idxs = self.neighbours[node_idx]  # sorted global node indices
            num_neighbors = min(len(global_neighbor_idxs), self.max_degree)  # cap at max_degree

            # Convert global indices -> networkx nodes -> local indices
            neighbor_local_idx = np.zeros(self.max_degree, dtype=np.int64)
            neighbor_mask = np.zeros(self.max_degree, dtype=np.float32)

            for i in range(num_neighbors):
                global_ngbr_idx = global_neighbor_idxs[i]
                ngbr_node = self.idx_to_node[global_ngbr_idx]  # networkx node object
                local_idx = node_idx_local_map[ngbr_node]      # local index in subgraph
                neighbor_local_idx[i] = local_idx
                neighbor_mask[i] = 1.0

            self.ego_graph_cache[node] = {
                'ego_graph': node_ego_graph,
                'node_idx_local_map': node_idx_local_map, # or name as ego_node_to_idx # used to generate x_local from x_global using this local_idxs_map.
                'local_idx_node_map': local_idx_node_map,
                'edge_list': ego_edge_index,
                'num_edges': ego_edge_index.shape[1],
                'hops_to_ego': hops_to_ego,
                'neighbor_local_idx': neighbor_local_idx,
                'neighbor_mask': neighbor_mask
            }
        self.max_number_ego_edges = 2 * self.max_number_ego_edges # to be consistent with the PyG convention of counting each undirected edge twice (also, refer to GraphCTF.nx_graph_to_sparse_edge_representation() method)
        return

    def gen_global_embedding_cache(self):
        """
        (0) degree,
        (1) distance_to_red_flag,
        (2) distance_to_blue_flag,
        (3) is_frontier,
        (4) frontier_resolution_mass,
        (5) distance_to_nearest_frontier.
        (N, 6)
        Do not need to recompute this... Structural properties. Make a part of global subgraph cache...
        """
        G = copy.deepcopy(self.graph)
        x_global = np.zeros((self.num_nodes, 6))
        distance_to_red_flag = nx.single_source_shortest_path_length(G, self.red_flag_node)
        distance_to_blue_flag = nx.single_source_shortest_path_length(G, self.blue_flag_node)
        for node_idx in range(self.num_nodes):
            node = self.idx_to_node[node_idx]
            x_idx = np.zeros(6)
            degree_normalized = G.degree(node) / self.max_degree

            x_idx[0] = degree_normalized
            x_idx[1] = distance_to_red_flag[node] / self.graph_diameter #!!!! which red_flag???? For next version with imperfect flag knowledge / multiple flag hypothesis -- not required for single hypothesis.
            x_idx[2] = distance_to_blue_flag[node] / self.graph_diameter
            x_idx[3] = self.is_frontier[node]
            x_idx[4] = self.frontier_resolution_mass[node]
            x_idx[5] = self.dist_to_nearest_frontier.get(node, self.graph_diameter) / self.graph_diameter
            x_global[node_idx] = x_idx
        x_global = np.array(x_global)
        self.global_embedding_cache = copy.deepcopy(x_global)
        return x_global

    def get_observation(self, agent):
        """
        Returns a permutation-invariant observation for an agent:
        - Its current node features
        - Features of neighboring nodes

        [ally_agent, enemy_agent, home_flag, enemy_flag]
        """
        """
        NOTE (on January 1, 2026):
        Observations assume the agent observes the entire graph topology with masked features within a visibility radius.
        i.e. known map and unknown occupancy.
        For v2: unknown map and unknown occupancy. ==> RAL paper. 4 agents moving a cooridoor.
        Later version: only a subgraph is truly visible.
        """
        # v0 -> IPPO. So we can use Dim2 as node pose coordinates.
        ## Dim1 and 2: [-1, 0, +1] ==> -1 is for adversary, 0 for unocuppied, +1 for ally, -2 for away_flag and +2 for home_flag
        ## Dim3: node pos coordinate (just for v0, for v1 we would want -> isomorphism between coordinate transformation, just graph topology and relevant geoemtric factors and not the coordinate system used ==> transfers less within the same team and maybe not at all to the enemy team because of a different coordinate reference ==> )
        ## Above defined Dim3 == XXXXXXXX
        ## Dim3: relative distance from node
        ## Dim4: relative distance from home_flag (for v1 -> what's a coordinate isomorphic notion of distance in terms of number of hops et cetera)
        ## Dim5: Visibility bit.
        ## Dim6: relative pose of away flag (if visible) otherwise -inf. (Assume visible for v0 --> opponent flag location known)
        assert self.obs_mode == {'occupancy': 'known', 'global_map': 'known', 'flag': 'partial'}
        #### IMPORTANT: Frontier based feature == distance from information frontier... Flag discovered --> no? Information frontier expanded...
        #### k-hop information frontier from the agent position....
        #### Question -- how to share information in the team --> MAPPO.
        #### Average Best Response v0 -> Robust Best Response
        #### IMPORTANT: Perhaps -- left and right and center frontier...

        agent_state = self.state[agent]
        agent_team = agent[0]
        home_flag_pos = self.red_flag_pos*(agent_team == 'R') + self.blue_flag_pos*(agent_team == 'B')
        node_pos_dict = nx.get_node_attributes(self.graph, 'pos')
        
        F = 7 # F = node_feature_dim
        node_feature_matrix = np.zeros((self.num_nodes, F))
        node_feats_global = self.global_embedding
        assert node_feats_global is not None
        is_visible = lambda node: np.linalg.norm(np.array(node_pos_dict[node]) - np.array(node_pos_dict[agent_state])) <= self.partial_visibility_radius

        for node_idx in range(self.num_nodes):
            node = self.idx_to_node[node_idx]
            node_global_embedding = node_feats_global[node_idx]
            if is_visible(node):
                feat_visibility_bit = 1
                if node_global_embedding[0] == 0: feat_agent = 0
                else: feat_agent = +1*(agent_team == self.idx_to_team_dict[node_global_embedding[0]]) + (-1)*(1 - (agent_team == self.idx_to_team_dict[node_global_embedding[0]]))

                if node_global_embedding[1] == 0: feat_flag = 0
                else: feat_flag = +1*(agent_team == self.idx_to_team_dict[node_global_embedding[1]]) + (-1)*(1 - (agent_team == self.idx_to_team_dict[node_global_embedding[1]]))
            else:
                feat_visibility_bit = 0
                feat_agent, feat_flag = 0, 0 # proxy for saying that we don't know what is there at those nodes. Not using arbitrary negative values such as -100 or -10.

            feat_rel_node_pos_to_agent = np.array(node_pos_dict[node]) - np.array(node_pos_dict[agent_state]) # Not rotation invariant (if the map rotates).
            feat_rel_node_pos_to_home_flag = np.array(node_pos_dict[node]) - home_flag_pos
            feature = np.concat([[feat_agent, feat_flag], feat_rel_node_pos_to_agent, feat_rel_node_pos_to_home_flag, [feat_visibility_bit]])
            node_feature_matrix[node_idx] = feature

        node_feature_matrix = np.array(node_feature_matrix, dtype=np.float32) # shape = (num_nodes, F)

        node_visibility_mask = np.ones((self.num_nodes,), dtype=np.float32)
        node_visibility_mask_bool = self.obs_type == 'sub-graph' # For sub-graph observations with different number of nodes, node_visibility_mask is used to pad the observations with dummy nodes. (Since SB3 assumes fixed size observations).
        if node_visibility_mask_bool:
            assert node_visibility_mask is not None
        
        agent_node_mask = np.zeros((self.num_nodes))
        agent_node_mask[self.node_to_idx[self.state[agent]]] = 1
        agent_node_mask = np.array(agent_node_mask, dtype=np.float32)

        action_mask = self.get_action_mask(agent)
        """
        To implement on January 9 2026:
        # Write subgraph_observation(sub_graph radius)
        # What are node features for the subgraph observation -- perfect information over agents, imperfect information over flags (bools for agents and flags, features for distance to agents and distance to flag (if known))
        # Common Graph Extractor for Red and Blue??

        #### Implement GNN training for subgraph policy -- see if you learn to reach the Red flag for a random red, for PPO hyperparams...
        """
        """
        Paper: Average Best Response on Robust Goal-conditioned Opponent Policies.
        Metric: Compare occupancy of common information frontier -- for how many rollouts does the agent learn to reach / discover the flag location (against the most robust mixture of opponents)...
        """
        obs = {
            'x': node_feature_matrix, # return adjacency
            'edge_index': copy.deepcopy(self.edge_index), 
            'edge_attr': copy.deepcopy(self.edge_features),
            'agent_node_mask': agent_node_mask,
            'node_visibility_mask': node_visibility_mask,
            'action_mask': action_mask
        }
        return obs
    
    """
            is_ego = 0
            is_frontier = global_embedding_cache[node_idx][3]
            frontier_resolution_mass = global_embedding_cache[node_idx][4]
            is_regular = 0
            is_flag_discovered_here = 0
            distance_to_ego = self.graph_diameter / self.graph_diameter
            distance_to_blue_flag = global_embedding_cache[node_idx][2]
            distance_to_red_flag = global_embedding_cache[node_idx][1]
            enemy_flag_known = 0
            min_enemy_distance_to_ego = None
            min_teammate_distance_to_ego = None
            structural_degree = global_embedding_cache[node_idx][0]
            visibility_bit = 0

            x_idx[0] = is_ego
            x_idx[1] = is_frontier
            x_idx[2] = frontier_resolution_mass
            x_idx[3] = is_regular
            x_idx[4] = is_flag_discovered_here
            x_idx[5] = distance_to_ego
            x_idx[6] = distance_to_blue_flag
            x_idx[7] = distance_to_red_flag
            x_idx[8] = enemy_flag_known
            x_idx[9] = min_enemy_distance_to_ego
            x_idx[10] = min_teammate_distance_to_ego
            x_idx[11] = structural_degree
            x_idx[12] = visibility_bit

        (1) degree,
        (2) distance_to_red_flag,
        (3) distance_to_blue_flag,
        (4) is_frontier,
        (5) frontier_resolution_mass.
        (N, 5)
        Do not need to recompute this... Structural properties. Make a part of global subgraph cache...

        x_idx[0] = degree_normalized
        x_idx[1] = distance_to_red_flag[node] / self.graph_diameter #!!!! which red_flag???? For next version with imperfect flag knowledge / multiple flag hypothesis -- not required for single hypothesis.
        x_idx[2] = distance_to_blue_flag[node] / self.graph_diameter
        x_idx[3] = self.is_frontier[node]
        x_idx[4] = self.frontier_resolution_mass[node]
        x_global[node_idx] = x_idx

        #### OBSERVATION DICT...
        [0]  is_ego -> dummy val for global embedding x_global
        [1]  is_frontier -> true val for global embedding x_global
        [2]  is_regular -> dummy val for global embedding x_global
        [3]  is_flag_discovered_here -> true val for global embedding x_global

        [4]  distance_to_ego -> dummy val for global embedding x_global

        [5]  distance_to_home_flag        (ego-only) -> true val for global embedding x_global (red and blue)
        [6]  home_flag_known              (ego-only) -> true val for global embedding x_global
                OOOOOOOOO redundant info [6]
        [7]  distance_to_enemy_flag       (ego-only) -> dummy val for global embedding x_global
        [8]  enemy_flag_known             (ego-only) -> dummy val for global embedding x_global

        [10] min_enemy_distance_to_ego    (ego-only) -> true val for global embedding x_global: min_enemy distance to each node (both red and blue)
        [11] min_teammate_distance_to_ego (ego-only) -> true val for global embedding x_global: min_ally distance to each node (both red and blue)

        [12] structural_degree            (regular-only, optional but recommended) -> true val for global embedding x_global
        [13] visibility_bit -> dummy val for global embedding x_global

        [9]  frontier_unresolved         (frontier-only) -> dummy val for global embedding x_global -> for first implementation of this sub-graph observation, not relevant as graph is static / any frontier node gives the same observation.
                XXXXXXXX


        x_idx[0] = degree_normalized
        x_idx[1] = distance_to_red_flag[node] / self.graph_diameter #!!!! which red_flag???? For next version with imperfect flag knowledge / multiple flag hypothesis -- not required for single hypothesis.
        x_idx[2] = distance_to_blue_flag[node] / self.graph_diameter
        x_idx[3] = self.is_frontier[node]
        x_idx[4] = self.frontier_resolution_mass[node]
        
    """
    def get_observation_v1(self, agent):
        # v1: SUBGRAPH OBSERVATIONS...
        """
        Returns a permutation-invariant observation for an agent:
        - Its current node features
        - Features of neighboring nodes
        """
        assert self.obs_mode == {'occupancy': 'known', 'global_map': 'known', 'flag': 'partial'}
        agent_team = agent[0]
        agent_node = self.state[agent]
        agent_ego_data_cache = self.ego_graph_cache[agent_node]
        # Generate 'x', 'edge_attr' and 'edge_index' for this ego-graph...

        agent_ego_graph = agent_ego_data_cache['ego_graph']
        num_ego_nodes = agent_ego_graph.number_of_nodes()
        agent_node_idx_local_map = agent_ego_data_cache['node_idx_local_map'] # used to generate x_local from x_global using this local_idxs_map.
        agent_local_idx_node_map = agent_ego_data_cache['local_idx_node_map']
        agent_ego_edge_index = agent_ego_data_cache['edge_list']
        num_edges = agent_ego_edge_index.shape[1]
        agent_hops_to_ego = agent_ego_data_cache['hops_to_ego']

        # Construct x and edge_index from agent_ego_graph with padding on number of agents --> padding on max number of agents in ego_graph pending!!!!...
        F = 13
        ego_only_feature_idxs = [6, 7, 8, 9, 10]
        x_ego = np.zeros((self.max_number_ego_nodes, F))
        E = self.max_number_ego_edges
        agent_ego_edge_index_padded = np.zeros((2, E))
        agent_ego_edge_index_padded[:, :num_edges] = agent_ego_edge_index

        dummy_val = 0
        x_ego[:, ego_only_feature_idxs] = dummy_val*np.ones(len(ego_only_feature_idxs))

        for ego_node_idx in range(num_ego_nodes):
            node = agent_local_idx_node_map[ego_node_idx] # Networkx ID in global graph frame.
            node_idx = self.node_to_idx[node] # node_idx is the canonical ID in global graph frame.
            x_ego_idx_cache_data = self.global_embedding_cache[node_idx].copy()
            # Opponent flag indicator: Red looks for Blue flag, Blue looks for Red flag
            is_flag_discovered_here = float(node == self.blue_flag_node) if agent_team.startswith('R') else float(node == self.red_flag_true)
            if node == agent_node:
                # Ego-only data...
                is_ego = 1
                agent_node_local_idx = ego_node_idx
                if agent_team.startswith('R'): # Red agent has perfect information of the Blue flag, we do not use frontier_node features for the Red team.
                    enemy_flag_known = 1
                    distance_to_home_flag = x_ego_idx_cache_data[1]
                    distance_to_away_flag = x_ego_idx_cache_data[2]
                else:
                    enemy_flag_known = self.enemy_flag_known[agent]
                    distance_to_home_flag = x_ego_idx_cache_data[2]
                    if enemy_flag_known: distance_to_away_flag = x_ego_idx_cache_data[1]
                    else: distance_to_away_flag = 0
                    
                min_enemy_distance_to_ego = self.min_opp_distance[agent]
                min_teammate_distance_to_ego = self.min_teammate_distance[agent]
                structural_degree = x_ego_idx_cache_data[0]
                visibility_bit = 1
                
                x_ego[ego_node_idx, 0] = is_ego
                x_ego[ego_node_idx, 1] = 0
                x_ego[ego_node_idx, 2] = 0 # (frontier-only)
                x_ego[ego_node_idx, 3] = 0
                x_ego[ego_node_idx, 4] = is_flag_discovered_here
                x_ego[ego_node_idx, 5] = 0
                x_ego[ego_node_idx, 6] = distance_to_home_flag
                x_ego[ego_node_idx, 7] = distance_to_away_flag
                x_ego[ego_node_idx, 8] = enemy_flag_known
                x_ego[ego_node_idx, 9] = min_enemy_distance_to_ego
                x_ego[ego_node_idx, 10] = min_teammate_distance_to_ego
                x_ego[ego_node_idx, 11] = structural_degree
                x_ego[ego_node_idx, 12] = visibility_bit
                continue

            # Frontier and regular node data...
            is_ego = 0
            if agent_team.startswith('R'): # Red agent has perfect information of the Blue flag, we do not use frontier_node features for the Red team.
                is_frontier = 0
                frontier_resolution_mass = 0
            else:
                is_frontier = x_ego_idx_cache_data[3]
                frontier_resolution_mass = x_ego_idx_cache_data[4]
            is_regular = not (is_ego or is_frontier)
            distance_to_ego = agent_hops_to_ego[node] / self.graph_diameter
            structural_degree = x_ego_idx_cache_data[0]
            visibility_bit = 1

            x_ego[ego_node_idx, 0] = is_ego
            x_ego[ego_node_idx, 1] = is_frontier
            x_ego[ego_node_idx, 2] = frontier_resolution_mass # (frontier-only)
            x_ego[ego_node_idx, 3] = is_regular
            x_ego[ego_node_idx, 4] = is_flag_discovered_here
            x_ego[ego_node_idx, 5] = distance_to_ego
            x_ego[ego_node_idx, 11] = structural_degree
            x_ego[ego_node_idx, 12] = visibility_bit
                
        ### AGENT NODE MASK
        agent_node_mask = np.zeros((self.num_nodes))
        agent_node_mask[self.node_to_idx[self.state[agent]]] = 1
        agent_node_mask = np.array(agent_node_mask, dtype=np.float32)

        ### NODE AND EDGE VISIBILITY MASK
        node_visibility_mask = np.zeros(self.max_number_ego_nodes, dtype=np.float32)
        node_visibility_mask[:num_ego_nodes] = 1.0
        node_visibility_mask_bool = self.obs_type == 'sub-graph' # For sub-graph observations with different number of nodes, node_visibility_mask is used to pad the observations with dummy nodes. (Since SB3 assumes fixed size observations).
        if node_visibility_mask_bool:
            assert node_visibility_mask is not None

        edge_visibility_mask = np.zeros(self.max_number_ego_edges, dtype=np.float32)
        if agent_ego_edge_index is not None and len(agent_ego_edge_index) > 0:
            edge_visibility_mask[:num_edges] = 1.0
        
        ### ACTION MASK
        action_mask = self.get_action_mask(agent)
        
        edge_attr = np.zeros((self.max_number_ego_edges, 1))

        ### FETCH NEIGHBOR INFO FROM CACHE (precomputed in gen_ego_graph_cache)
        neighbor_local_idx = agent_ego_data_cache['neighbor_local_idx']  # [max_degree], padded
        neighbor_mask = agent_ego_data_cache['neighbor_mask']            # [max_degree], 1s for valid neighbors

        obs = {
            'x': x_ego.copy(), # return adjacency
            'edge_index': agent_ego_edge_index_padded.copy(), #np.array(agent_ego_edge_index).copy(),
            'edge_attr': edge_attr, #self.edge_features.copy(), #np.array(self.edge_features).copy(), #### Pending....
            'agent_node_mask': agent_node_mask,
            'agent_node_local_idx': agent_node_local_idx, # in the subgraph frame, what is the agent / ego_nodes idx (per x and edge_index)
            'node_visibility_mask': node_visibility_mask,
            'edge_visibility_mask': edge_visibility_mask,
            'action_mask': action_mask,
            'neighbor_local_idx': neighbor_local_idx,
            'neighbor_mask': neighbor_mask,
            # V2 compatibility fields (added for unified observation_space)
            'num_visible_nodes': np.array([num_ego_nodes], dtype=np.int64),
            'num_visible_edges': np.array([num_edges], dtype=np.int64)
        }
        return obs

        """
        Paper: Average Best Response on Robust Goal-conditioned Opponent Policies.
        Metric: Compare occupancy of common information frontier -- for how many rollouts does the agent learn to reach / discover the flag location (against the most robust mixture of opponents)...
        """

    def get_observation_v2(self, agent):
        """
        v2: Fixed-size observations optimized for vectorized batch construction.

        Key difference from v1: Returns num_visible_nodes and num_visible_edges
        so policies can use direct slicing (O(1)) instead of boolean masking (O(N)).

        Visible nodes/edges are already placed first in the arrays (contiguous),
        followed by zero-padding. This enables:
          x_visible = x[:num_visible_nodes]  # Direct slice, no loop needed

        Returns same structure as v1, plus:
          - 'num_visible_nodes': int, number of actual nodes in ego-graph
          - 'num_visible_edges': int, number of actual edges in ego-graph
        """
        assert self.obs_mode == {'occupancy': 'known', 'global_map': 'known', 'flag': 'partial'}
        agent_team = agent[0]
        agent_node = self.state[agent]
        agent_ego_data_cache = self.ego_graph_cache[agent_node]

        agent_ego_graph = agent_ego_data_cache['ego_graph']
        num_ego_nodes = agent_ego_graph.number_of_nodes()
        agent_node_idx_local_map = agent_ego_data_cache['node_idx_local_map']
        agent_local_idx_node_map = agent_ego_data_cache['local_idx_node_map']
        agent_ego_edge_index = agent_ego_data_cache['edge_list']
        num_edges = agent_ego_edge_index.shape[1]
        agent_hops_to_ego = agent_ego_data_cache['hops_to_ego']

        # Pre-allocate fixed-size arrays (visible data first, then padding)
        F = 13
        ego_only_feature_idxs = [6, 7, 8, 9, 10]
        x_ego = np.zeros((self.max_number_ego_nodes, F), dtype=np.float32)
        E = self.max_number_ego_edges
        agent_ego_edge_index_padded = np.zeros((2, E), dtype=np.int64)
        agent_ego_edge_index_padded[:, :num_edges] = agent_ego_edge_index

        dummy_val = 0
        x_ego[:, ego_only_feature_idxs] = dummy_val * np.ones(len(ego_only_feature_idxs))

        agent_node_local_idx = 0  # Will be set when we find ego node

        for ego_node_idx in range(num_ego_nodes):
            node = agent_local_idx_node_map[ego_node_idx]
            node_idx = self.node_to_idx[node]
            x_ego_idx_cache_data = self.global_embedding_cache[node_idx].copy()
            # Opponent flag indicator: Red looks for Blue flag, Blue looks for Red flag
            is_flag_discovered_here = float(node == self.blue_flag_node) if agent_team.startswith('R') else float(node == self.red_flag_true)

            if node == agent_node:
                is_ego = 1
                agent_node_local_idx = ego_node_idx
                if agent_team.startswith('R'):
                    enemy_flag_known = 1
                    distance_to_home_flag = x_ego_idx_cache_data[1]
                    distance_to_away_flag = x_ego_idx_cache_data[2]
                else:
                    enemy_flag_known = self.enemy_flag_known[agent]
                    distance_to_home_flag = x_ego_idx_cache_data[2]
                    if enemy_flag_known:
                        distance_to_away_flag = x_ego_idx_cache_data[1]
                    else:
                        distance_to_away_flag = 0

                min_enemy_distance_to_ego = self.min_opp_distance[agent]
                min_teammate_distance_to_ego = self.min_teammate_distance[agent]
                structural_degree = x_ego_idx_cache_data[0]
                visibility_bit = 1

                x_ego[ego_node_idx, 0] = is_ego
                x_ego[ego_node_idx, 1] = 0
                x_ego[ego_node_idx, 2] = 0
                x_ego[ego_node_idx, 3] = 0
                x_ego[ego_node_idx, 4] = is_flag_discovered_here
                x_ego[ego_node_idx, 5] = 0
                x_ego[ego_node_idx, 6] = distance_to_home_flag
                x_ego[ego_node_idx, 7] = distance_to_away_flag
                x_ego[ego_node_idx, 8] = enemy_flag_known
                x_ego[ego_node_idx, 9] = min_enemy_distance_to_ego
                x_ego[ego_node_idx, 10] = min_teammate_distance_to_ego
                x_ego[ego_node_idx, 11] = structural_degree
                x_ego[ego_node_idx, 12] = visibility_bit
                continue

            is_ego = 0
            if agent_team.startswith('R'):
                is_frontier = 0
                frontier_resolution_mass = 0
            else:
                is_frontier = x_ego_idx_cache_data[3]
                frontier_resolution_mass = x_ego_idx_cache_data[4]
            is_regular = not (is_ego or is_frontier)
            distance_to_ego = agent_hops_to_ego[node] / self.graph_diameter
            structural_degree = x_ego_idx_cache_data[0]
            visibility_bit = 1

            x_ego[ego_node_idx, 0] = is_ego
            x_ego[ego_node_idx, 1] = is_frontier
            x_ego[ego_node_idx, 2] = frontier_resolution_mass
            x_ego[ego_node_idx, 3] = is_regular
            x_ego[ego_node_idx, 4] = is_flag_discovered_here
            x_ego[ego_node_idx, 5] = distance_to_ego
            x_ego[ego_node_idx, 11] = structural_degree
            x_ego[ego_node_idx, 12] = visibility_bit

        # Agent node mask (global frame)
        agent_node_mask = np.zeros((self.num_nodes,), dtype=np.float32)
        agent_node_mask[self.node_to_idx[self.state[agent]]] = 1

        # Visibility masks (for backward compatibility)
        node_visibility_mask = np.zeros(self.max_number_ego_nodes, dtype=np.float32)
        node_visibility_mask[:num_ego_nodes] = 1.0

        edge_visibility_mask = np.zeros(self.max_number_ego_edges, dtype=np.float32)
        if agent_ego_edge_index is not None and len(agent_ego_edge_index) > 0:
            edge_visibility_mask[:num_edges] = 1.0

        action_mask = self.get_action_mask(agent)
        edge_attr = np.zeros((self.max_number_ego_edges, 1), dtype=np.float32)

        # Neighbor info from cache
        neighbor_local_idx = agent_ego_data_cache['neighbor_local_idx']
        neighbor_mask = agent_ego_data_cache['neighbor_mask']

        obs = {
            # Core graph data (visible first, then padding)
            'x': x_ego,
            'edge_index': agent_ego_edge_index_padded,
            'edge_attr': edge_attr,
            # Counts for direct slicing (v2 addition)
            'num_visible_nodes': num_ego_nodes,
            'num_visible_edges': num_edges,
            # Agent info
            'agent_node_mask': agent_node_mask,
            'agent_node_local_idx': agent_node_local_idx,
            # Visibility masks (backward compatibility with v1)
            'node_visibility_mask': node_visibility_mask,
            'edge_visibility_mask': edge_visibility_mask,
            # Action info
            'action_mask': action_mask,
            'neighbor_local_idx': neighbor_local_idx,
            'neighbor_mask': neighbor_mask
        }
        return obs

    def get_observation_v3(self, agent):
        """
        v3: Rich per-node features (F=16). Extends v2 with:
          - Per-node distance_to_home_flag and distance_to_away_flag (were ego-only in v2)
          - Per-node distance_to_frontier (NEW)
          - Per-node has_opponent and has_teammate binary indicators (NEW)
          - Ego-only: enemy_flag_known, min_opp_distance, min_teammate_distance

        Feature layout (F=16):
          0  is_ego                   (per-node)
          1  is_frontier              (per-node)
          2  frontier_resolution_mass (per-node)
          3  is_regular               (per-node)
          4  is_flag_discovered_here  (per-node)
          5  distance_to_ego          (per-node)
          6  distance_to_home_flag    (per-node)  -- was ego-only in v2
          7  distance_to_away_flag    (per-node)  -- was ego-only in v2, 0 if Blue & flag unknown
          8  enemy_flag_known         (ego-only)
          9  distance_to_frontier     (per-node)  -- NEW
          10 has_opponent             (per-node)  -- NEW
          11 has_teammate             (per-node)  -- NEW
          12 min_opp_distance         (ego-only)
          13 min_teammate_distance    (ego-only)
          14 structural_degree        (per-node)
          15 visibility_bit           (per-node)
        """
        assert self.obs_mode == {'occupancy': 'known', 'global_map': 'known', 'flag': 'partial'}
        agent_team = agent[0]
        agent_node = self.state[agent]
        agent_ego_data_cache = self.ego_graph_cache[agent_node]

        agent_ego_graph = agent_ego_data_cache['ego_graph']
        num_ego_nodes = agent_ego_graph.number_of_nodes()
        agent_node_idx_local_map = agent_ego_data_cache['node_idx_local_map']
        agent_local_idx_node_map = agent_ego_data_cache['local_idx_node_map']
        agent_ego_edge_index = agent_ego_data_cache['edge_list']
        num_edges = agent_ego_edge_index.shape[1]
        agent_hops_to_ego = agent_ego_data_cache['hops_to_ego']

        # Pre-allocate fixed-size arrays (visible data first, then padding)
        F = 16
        ego_only_feature_idxs = [8, 12, 13]
        x_ego = np.zeros((self.max_number_ego_nodes, F), dtype=np.float32)
        E = self.max_number_ego_edges
        agent_ego_edge_index_padded = np.zeros((2, E), dtype=np.int64)
        agent_ego_edge_index_padded[:, :num_edges] = agent_ego_edge_index

        dummy_val = 0
        x_ego[:, ego_only_feature_idxs] = dummy_val * np.ones(len(ego_only_feature_idxs))

        # Precompute agent occupancy sets for has_opponent / has_teammate
        opponent_nodes = set()
        teammate_nodes = set()
        for other_agent in self.agents:
            if other_agent == agent:
                continue
            if other_agent[0] == agent_team:
                teammate_nodes.add(self.state[other_agent])
            else:
                opponent_nodes.add(self.state[other_agent])

        agent_node_local_idx = 0  # Will be set when we find ego node

        for ego_node_idx in range(num_ego_nodes):
            node = agent_local_idx_node_map[ego_node_idx]
            node_idx = self.node_to_idx[node]
            x_ego_idx_cache_data = self.global_embedding_cache[node_idx].copy()

            # --- Per-node features (all nodes get these) ---
            # Opponent flag indicator: Red looks for Blue flag, Blue looks for Red flag
            if agent_team == 'R':
                is_flag_discovered_here = float(node == self.blue_flag_node)
            else:
                is_flag_discovered_here = float(node == self.red_flag_true)
            if agent_team == 'R':
                is_frontier = 0
                frontier_resolution_mass = 0
                distance_to_frontier = 0  # Red has perfect flag info, frontier is irrelevant
            else:
                is_frontier = x_ego_idx_cache_data[3]
                frontier_resolution_mass = x_ego_idx_cache_data[4]
                distance_to_frontier = x_ego_idx_cache_data[5]  # Already normalized by graph_diameter

            is_ego = float(node == agent_node)
            is_regular = float(not (is_ego or is_frontier))
            distance_to_ego = 0.0 if node == agent_node else agent_hops_to_ego[node] / self.graph_diameter
            structural_degree = x_ego_idx_cache_data[0]

            # Per-node flag distances (NEW: computed for ALL nodes, not just ego)
            if agent_team == 'R':
                distance_to_home_flag = x_ego_idx_cache_data[1]  # dist to red flag
                distance_to_away_flag = x_ego_idx_cache_data[2]  # dist to blue flag
            else:
                distance_to_home_flag = x_ego_idx_cache_data[2]  # dist to blue flag
                if self.enemy_flag_known.get(agent, False):
                    distance_to_away_flag = x_ego_idx_cache_data[1]  # dist to red flag
                else:
                    distance_to_away_flag = 0  # Blue doesn't know flag location yet

            # Per-node occupancy indicators (NEW)
            has_opponent = float(node in opponent_nodes)
            has_teammate = float(node in teammate_nodes)

            x_ego[ego_node_idx, 0] = is_ego
            x_ego[ego_node_idx, 1] = is_frontier
            x_ego[ego_node_idx, 2] = frontier_resolution_mass
            x_ego[ego_node_idx, 3] = is_regular
            x_ego[ego_node_idx, 4] = is_flag_discovered_here
            x_ego[ego_node_idx, 5] = distance_to_ego
            x_ego[ego_node_idx, 6] = distance_to_home_flag
            x_ego[ego_node_idx, 7] = distance_to_away_flag
            # idx 8 = enemy_flag_known (ego-only, set below)
            x_ego[ego_node_idx, 9] = distance_to_frontier
            x_ego[ego_node_idx, 10] = has_opponent
            x_ego[ego_node_idx, 11] = has_teammate
            # idx 12 = min_opp_distance (ego-only, set below)
            # idx 13 = min_teammate_distance (ego-only, set below)
            x_ego[ego_node_idx, 14] = structural_degree
            x_ego[ego_node_idx, 15] = 1.0  # visibility_bit

            if node == agent_node:
                agent_node_local_idx = ego_node_idx
                # Ego-only features
                x_ego[ego_node_idx, 8] = float(self.enemy_flag_known.get(agent, False)) if agent_team == 'B' else 1.0
                x_ego[ego_node_idx, 12] = self.min_opp_distance[agent]
                x_ego[ego_node_idx, 13] = self.min_teammate_distance[agent]

        # Agent node mask (global frame)
        agent_node_mask = np.zeros((self.num_nodes,), dtype=np.float32)
        agent_node_mask[self.node_to_idx[self.state[agent]]] = 1

        # Visibility masks (for backward compatibility)
        node_visibility_mask = np.zeros(self.max_number_ego_nodes, dtype=np.float32)
        node_visibility_mask[:num_ego_nodes] = 1.0

        edge_visibility_mask = np.zeros(self.max_number_ego_edges, dtype=np.float32)
        if agent_ego_edge_index is not None and len(agent_ego_edge_index) > 0:
            edge_visibility_mask[:num_edges] = 1.0

        action_mask = self.get_action_mask(agent)
        edge_attr = np.zeros((self.max_number_ego_edges, 1), dtype=np.float32)

        # Neighbor info from cache
        neighbor_local_idx = agent_ego_data_cache['neighbor_local_idx']
        neighbor_mask = agent_ego_data_cache['neighbor_mask']

        obs = {
            # Core graph data (visible first, then padding)
            'x': x_ego,
            'edge_index': agent_ego_edge_index_padded,
            'edge_attr': edge_attr,
            # Counts for direct slicing (v2/v3)
            'num_visible_nodes': num_ego_nodes,
            'num_visible_edges': num_edges,
            # Agent info
            'agent_node_mask': agent_node_mask,
            'agent_node_local_idx': agent_node_local_idx,
            # Visibility masks (backward compatibility with v1)
            'node_visibility_mask': node_visibility_mask,
            'edge_visibility_mask': edge_visibility_mask,
            # Action info
            'action_mask': action_mask,
            'neighbor_local_idx': neighbor_local_idx,
            'neighbor_mask': neighbor_mask
        }
        return obs

    def get_action_mask(self, agent):
        agent_node = self.state[agent]
        neighbours = self.neighbours[self.node_to_idx[agent_node]]

        action_mask = np.zeros((self.num_actions,))
        for action in range(len(neighbours)):
            action_mask[action] = 1.

        action_mask[self.num_actions - 2] = 1 # stay action.

        valid_tag = False
        opp_agents = self.red_team_agents if agent[0] == 'B' else self.blue_team_agents
        for opp in opp_agents:
            """
            if np.linalg.norm(np.array(self.node_pose_dict[self.state[opp]]) - np.array(self.node_pose_dict[agent_node])) <= self.tagging_radius:
                valid_tag = True
                break
            """
            if self.state[opp] == agent_node or self.state[opp] in self.graph.neighbors(agent_node):
                valid_tag = True
                break

        action_mask[self.num_actions - 1] = valid_tag
        action_mask = np.array(action_mask, dtype=np.float32)
        return action_mask
        
    def step(self, action_dict):
        """
            NOTES:
            A) Canonical ordering over neighbors as one defined by node_to_idx???
            B) And later verify permutation invariance -- shouldnt leak node_idx anywhere, and different labelling and ordering should lead to the same policy.

            If opponent policy is set via set_opponent_policy(), action_dict only needs
            actions for active agents. Passive agent actions are computed internally.
        """
        # Merge opponent actions if opponent policy is configured (backward compatible)
        if self.has_opponent_policy():
            opp_actions = self._compute_opponent_actions()
            action_dict = {**action_dict, **opp_actions}

        # permutation invariant stepping. Learn a permutation invariant policy. # Remarks: invariant vs equivariant. Tess Schmidt research. Terminology -> inductive bias in the learned GNN policies.
        # {0, 1, ...., max_degree - 1} map to neighbouring nodes.
        # max_degree maps to the staying at the current position action for an agent.
        # max_degree + 1 is the tagging action.

        #### PENDING: valid tagging logic and action_masking.
        agents = copy.deepcopy(self.agents)
        for agent in agents:
            assert action_dict[agent] in range(self.num_actions)

        # Collect tags...
        tagging_agents = []
        for agent in agents:
            if action_dict[agent] == self.num_actions - 1: tagging_agents.append(agent)

        agent_tag_status = {agent: False for agent in agents}
        for tagging_agent in tagging_agents:
            opponent_agents = self.red_team_agents if tagging_agent[0] == 'B' else self.blue_team_agents
            agent_node = self.state[tagging_agent]
            # Compute if any adversary within tagging radius..
            for opp_agent in opponent_agents:
                """
                if np.linalg.norm(np.array(self.node_pose_dict[self.state[opp_agent]]) - np.array(self.node_pose_dict[agent_node])) <= self.tagging_radius:
                    agent_tag_status[opp_agent] = True
                """
                if self.state[opp_agent] == agent_node or self.state[opp_agent] in self.graph.neighbors(agent_node): agent_tag_status[opp_agent] = True

        # Capture Red positions before movement/respawn (used for defense_zone_tag_bonus below).
        _pre_tag_red_pos = {a: self.state[a] for a in self.red_team_agents if a in agents}

        #G = copy.deepcopy(self.graph)
        for agent in agents:
            agent_node = self.state[agent]
            agent_tagged = agent_tag_status[agent]
            if not agent_tagged:
                agent_node_idx = self.node_to_idx[agent_node]
                act = action_dict[agent]
                neighbours = self.neighbours[agent_node_idx]
                if act in range(len(neighbours)):
                    next_node_idx = neighbours[act]
                    next_node = self.idx_to_node[next_node_idx]
                else: next_node = agent_node
            else:
                if agent.startswith('B'): valid_respawn_nodes = self.blue_spawn_valid_nodes
                elif agent.startswith('R'):
                    valid_respawn_regions = [self.red_spawn_valid_nodes_just_L, self.red_spawn_valid_nodes_just_R]
                    if agent_node in (set(self._bc.top_left_basin) | set(self._bc.top_right_basin)):
                        # check agent closer distance to which flag, respawn in the other flags
                        closest_red_flag = self.nearest_red_flag[agent_node]
                        valid_respawn_nodes = valid_respawn_regions[1-closest_red_flag]
                    else: valid_respawn_nodes = self.red_spawn_valid_nodes
                next_node = self.np_random.choice(valid_respawn_nodes)
            self.state[agent] = next_node

        infos = {agent: {} for agent in self.agents}
        obs = {agent: None for agent in self.agents}
        rewards = {agent: 0. for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        
        ########################
        ### Generate distance between agents: both teammates and adversaries before fetching observation from self.get_observation_v1()...
        #!!!!!! Pending: How to set is_flag_discovered here...??? Once the agents are in the backward reachable set of the flag, they discover it -- it becomes known -- frontier play. And is_flag_discovered here is just an indicator for the node containing the flag....

        if self.reveal_flag:
            pass  # Flag already revealed from reset — skip frontier discovery
        else:
            for agent in self.agents:
                """
                if agent.startswith('B') and (self.state[agent] in self.is_true_flag_reachable):
                    self.red_flag_resolved = True
                    self.enemy_flag_known[agent] = True ### NOTE: Info sharing between teammates. CRITICAL POINT: public belief within the team. Approximates Team Markov Equilibria. Otherwise private belief, more like Dec-POMDP like Equilibria.
                """
                if agent.startswith('B') and self.is_frontier[self.state[agent]]:
                    self.red_flag_resolved = True # REMARK: This works only for 2 flags since seeing one flag's frontier automatically proves the other to be true... Doesn't work for three flags...
                    self.enemy_flag_known[agent] = True ### NOTE: Info sharing between teammates. CRITICAL POINT: public belief within the team. Approximates Team Markov Equilibria. Otherwise private belief, more like Dec-POMDP like Equilibria.

            # This loop propagates information across the entire Blue team (if any one Blue agent resolves the Red flag, it becomes known to the entire Blue team)...
            for agent in self.agents:
                if agent.startswith('B'):
                    self.enemy_flag_known[agent] = self.red_flag_resolved

        self.min_teammate_distance = {
            agent: 1 for agent in self.agents
        }
        self.min_opp_distance = {
            agent: 1 for agent in self.agents
        }
        
        dist = {}
        for agent_0, agent_1 in self.all_agent_pairings:
            # dist[(agent_0, agent_1)]
            d = nx.shortest_path_length(self.graph, source=self.state[agent_0], target=self.state[agent_1]) / self.graph_diameter
            dist[(agent_0, agent_1)] = d
            if agent_0[0] == agent_1[0]:
                #Teammates.
                self.min_teammate_distance[agent_0] = min(d, self.min_teammate_distance[agent_0]) #distance
                self.min_teammate_distance[agent_1] = min(d, self.min_teammate_distance[agent_1])
            else:
                # Opponents.
                self.min_opp_distance[agent_0] = min(d, self.min_opp_distance[agent_0])
                self.min_opp_distance[agent_1] = min(d, self.min_opp_distance[agent_1])

        ########################
        """
        # Commented on January 26, 2026: Not needed for sub-graph observations...
        self.global_embedding = self.generate_global_embeddings()
        """
        for agent in self.agents:
            if self.obs_version == 3:
                obs[agent] = self.get_observation_v3(agent)
            elif self.obs_version == 2:
                obs[agent] = self.get_observation_v2(agent)
            else:
                obs[agent] = self.get_observation_v1(agent)

        # Compute truncations, terminations, and rewards...
        self.current_step += 1
        if self.current_step == self.max_num_cycles: truncations = {agent: True for agent in self.agents}

        self.team_goal_reach = {team: False for team in ['R', 'B']}
        for agent in agents:
            # In half mode Red has no offensive objective — skip Red flag-capture check.
            if agent[0] == 'R' and self.game_mode == 'half':
                continue
            opp_flag_node = self.red_flag_true if agent[0] == 'B' else self.blue_flag_node
            if self.team_goal_reach[agent[0]]: continue # VERY IMPORTANT. To avoid double assignment, for instance: Red flag reach true for Red_1 but then False for Red_2.
            elif self.state[agent] == opp_flag_node:
                self.team_goal_reach[agent[0]] = True

        # --- Reward shaping (optional, gated on non-zero coefficients) ---
        # 1. Tagging rewards: penalize being tagged, reward tagging opponent
        if self.T:
            blue_tagged = any(agent_tag_status[a] for a in self.blue_team_agents if a in agent_tag_status)
            red_tagged = any(agent_tag_status[a] for a in self.red_team_agents if a in agent_tag_status)
            for agent in agents:
                if agent[0] == 'B':
                    rewards[agent] -= blue_tagged * self.T
                    rewards[agent] += red_tagged * self.T
                else:
                    rewards[agent] += blue_tagged * self.T
                    rewards[agent] -= red_tagged * self.T

        # 1b. Defense zone tag bonus: extra reward for Blue tagging Red near Blue's home flag.
        #     Rewards active interception — complements home_defense_shaping (which only rewards
        #     passive proximity) and is orthogonal to tag_avoidance_shaping.
        if self.defense_zone_tag_bonus:
            defense_tag_occurred = any(
                agent_tag_status.get(ra, False)
                and self._dist_from_blue_flag.get(_pre_tag_red_pos.get(ra), self.graph_diameter) <= 6
                for ra in self.red_team_agents
            )
            if defense_tag_occurred:
                for agent in agents:
                    if agent[0] == 'B':
                        rewards[agent] += self.defense_zone_tag_bonus

        # 2. Flag approach shaping: reward reducing graph distance to opponent flag
        #    For Blue, only active after frontier discovery (when observation has distance_to_away_flag).
        if self.flag_approach_shaping:
            for agent in agents:
                if agent[0] == 'B' and not self.enemy_flag_known.get(agent, False):
                    continue  # Blue doesn't know flag yet — skip to avoid noisy reward
                opp_flag = self.red_flag_true if agent[0] == 'B' else self.blue_flag_node
                if agent[0] == 'R' and self.game_mode == 'half':
                    continue  # half mode: Red has no offensive flag to approach
                try:
                    curr_dist = nx.shortest_path_length(self.graph, self.state[agent], opp_flag)
                except nx.NetworkXNoPath:
                    curr_dist = self.graph_diameter
                prev_dist = self._prev_dist_to_flag.get(agent, curr_dist)
                # Positive reward for getting closer, negative for moving away
                rewards[agent] += self.flag_approach_shaping * (prev_dist - curr_dist)
                self._prev_dist_to_flag[agent] = curr_dist

        # 3. Tag avoidance shaping: small penalty for being close to opponents
        if self.tag_avoidance_shaping:
            for agent in agents:
                if agent[0] == 'B':
                    rewards[agent] -= self.tag_avoidance_shaping * (1.0 - self.min_opp_distance.get(agent, 1.0))

        # 4. Frontier approach shaping: reward Blue for reducing distance to nearest frontier node
        if self.frontier_approach_shaping:
            for agent in agents:
                if agent[0] == 'B':
                    curr_dist = self.dist_to_nearest_frontier.get(self.state[agent], self.graph_diameter)
                    prev_dist = self._prev_dist_to_frontier.get(agent, curr_dist)
                    rewards[agent] += self.frontier_approach_shaping * (prev_dist - curr_dist)
                    self._prev_dist_to_frontier[agent] = curr_dist

        # 5. Home defense shaping: reward Blue for being near home flag while Red threatens it
        if self.home_defense_shaping:
            red_close_to_home = any(
                self._dist_from_blue_flag.get(self.state[a], self.graph_diameter) <= 6
                for a in agents if a[0] == 'R'
            )
            if red_close_to_home:
                for agent in agents:
                    if agent[0] == 'B':
                        if self._dist_from_blue_flag.get(self.state[agent], self.graph_diameter) <= 4:
                            rewards[agent] += self.home_defense_shaping

        # 6. Defense pursuit shaping: reward Blue for closing on Red when Red is in the defense zone.
        #    Fires only when at least one Red is ≤6 hops from Blue's flag (same zone as
        #    defense_zone_tag_bonus). Always updates _prev_dist_to_nearest_red so the
        #    baseline stays accurate across tag/respawn events.
        if self.defense_pursuit_shaping:
            red_alive = [a for a in agents if a[0] == 'R']
            for agent in agents:
                if agent[0] == 'B':
                    if red_alive:
                        try:
                            curr_d = min(
                                nx.shortest_path_length(self.graph, self.state[agent], self.state[r])
                                for r in red_alive
                            )
                        except nx.NetworkXNoPath:
                            curr_d = self.graph_diameter
                    else:
                        curr_d = self.graph_diameter

                    red_in_defense_zone = any(
                        self._dist_from_blue_flag.get(self.state[r], self.graph_diameter) <= 6
                        for r in red_alive
                    )
                    if red_in_defense_zone:
                        prev_d = self._prev_dist_to_nearest_red.get(agent, curr_d)
                        rewards[agent] += self.defense_pursuit_shaping * (prev_d - curr_d)

                    # Always update so respawn doesn't pollute the next step's baseline
                    self._prev_dist_to_nearest_red[agent] = curr_d

        # --- Terminal rewards (flag capture, zero-sum) ---
        for agent in agents:
            opp_team = 'R' if agent[0] == 'B' else 'B'
            if self.team_goal_reach[agent[0]]:
                terminations[agent] = True
                rewards[agent] = +self.F
            elif self.team_goal_reach[opp_team]:
                terminations[agent] = True
                rewards[agent] = -self.F

        # --- Timeout reward for half mode ---
        # Red successfully defended to timeout: Red +F, Blue -F.
        # Only applies when no flag capture terminated the episode on this step.
        if self.game_mode == 'half' and any(truncations[a] for a in agents):
            if not any(terminations[a] for a in agents):
                for agent in agents:
                    rewards[agent] = self.F if agent[0] == 'R' else -self.F

        agents_to_remove = []
        for agent in self.agents:
            if terminations[agent] or truncations[agent]:
                agents_to_remove.append(agent)

        # Remove done agents
        for agent in agents_to_remove:
            self.agents.remove(agent)
        if self.verbose:
            print("Updated self.agents state: {}".format(self.state))
            print("rewards: {}".format(rewards))
            print("terminations: {}".format(terminations))
            print("truncations: {}".format(truncations))

        return obs, rewards, terminations, truncations, infos
    
    def num_agents(self):
        return len(self.agents)

    @staticmethod
    def draw_graph(G, pos=None):
        fig, ax = plt.subplots(figsize=(6, 6))

        pos_dict = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos=pos_dict, node_size=20, edge_color='black', node_color='#333333', width=1)

        return fig, ax

    def render(self, figname='sim', show_red_flag_frontier=True, flag_hypothesis=0, show_cooridoor=False, show_patrol_region=False, blue_agent_vision=False, **kwargs):
        assert flag_hypothesis in [0, 1]
        # First print graph.
        # Then print all agents.
        # Then plot all agent circles.

        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        fig, ax = GraphCTF.draw_graph(self.graph)
        pos = nx.get_node_attributes(self.graph, 'pos')
    
        blue_flag_img_path="flag_imgs/blue_flag.png"
        red_flag_img_path="flag_imgs/red_flag.png"
        blue_flag_node = self.blue_flag_node
        red_flag_node = [self.red_flag_node, self.red_flag_2_node][flag_hypothesis]
        #red_flag_node_2 = red_flag_2_node
        
        blue_flag_loc, red_flag_loc = pos[blue_flag_node], pos[red_flag_node]
        if blue_flag_loc is not None and self.game_mode != 'half':
            blue_flag_img = mpimg.imread(blue_flag_img_path)
            blue_flag_img = OffsetImage(blue_flag_img, zoom=0.12)
            ab_blue = AnnotationBbox(blue_flag_img, blue_flag_loc, frameon=False)
            ax.add_artist(ab_blue)

        if red_flag_loc is not None:
            red_flag_img = mpimg.imread(red_flag_img_path)
            red_flag_img = OffsetImage(red_flag_img, zoom=0.12)
            ab_red = AnnotationBbox(red_flag_img, red_flag_loc, frameon=False)
            ax.add_artist(ab_red)

        if show_red_flag_frontier:
            k = self.flag_visibility_max_hops
            khop_nodes = nx.single_source_shortest_path_length(self.graph, red_flag_node, cutoff=k).keys()
            khop_nodes = self._bc.top_left_basin # Added on March 7, 2026 for new map.
            for khop_node in khop_nodes:
                ax.scatter(*pos[khop_node], color="yellow", s=35, zorder=6)

        """
        # NOTE: self._bc_info not implemented.
        if show_cooridoor:
            corridor_nodes = self._bc_info.corridor_nodes
            corridor_color = 'green'
            for node in corridor_nodes:
                ax.scatter(*pos[node], color=corridor_color, s=35, zorder=6)
        
        if show_patrol_region:
            patrol_region = self._bc_info.patrol_region
            patrol_color = 'purple'
            for node in patrol_region:
                ax.scatter(*pos[node], color=patrol_color, s=35, zorder=6)
        """

        if kwargs.get('nodes', None) is not None:
            nodes = kwargs.get('nodes', None)
            assert isinstance(nodes, list)
            node_color = kwargs.get('nodes_color', 'green')
            for node in nodes:
                ax.scatter(*pos[node], color=node_color, s=35, zorder=6)

        for agent in self.agents:
            color = 'red' if agent.startswith('R') else 'blue'
            ax.scatter(*pos[self.state[agent]], color=color, s=35, zorder=6)
            circle = Circle(
                pos[self.state[agent]],          # (x, y)
                radius=1.5,
                facecolor=color,  # fill color
                edgecolor=color,  # outline color (optional)
                alpha=0.3,          # transparency (0 = invisible, 1 = opaque)
                linewidth=2
            )
            ax.add_patch(circle)
        
        plt.xlim(-3, 13)
        plt.ylim(-13, 3)
        plt.savefig('images/{}.png'.format(figname))
        plt.show()
        return fig, ax
        
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """
        Observation is a graph:
        x          : (N, F) node features
        edge_index : (2, E) COO edge list
        #edge_attr  : (E, D_e) edge features
        node_mask  : (N,) node validity mask
        """

        N = self.max_number_ego_nodes #self.graph.number_of_nodes() # this was for full graph input.
        F = 16 if self.obs_version == 3 else 13  # v3: rich per-node features (F=16), v1/v2: F=13
        E = self.max_number_ego_edges #self.edge_index.shape[1]. # this was for full graph input.
        D_e = 1 #D_e = 1 for dummy edge_features. #self.edge_features.shape[1] # was for older implementation v0.
        A = self.num_actions

        obs_space = Dict({
            "x": Box(
                low=-np.inf,
                high=np.inf,
                shape=(N, F),
                dtype=np.float32
            ),

            # edge_index is integer indices into nodes
            "edge_index": Box(
                low=0,
                high=N - 1,
                shape=(2, E),
                dtype=np.int64
            ),
            
            "edge_attr": Box(
                low=-np.inf,
                high=np.inf,
                shape=(E, D_e),
                dtype=np.float32
            ),
            
            "agent_node_mask": Box(
                low=0,
                high=1,
                shape=(self.num_nodes,),  # Full graph size, needed for GraphPolicy
                dtype=np.float32
            ),

            "agent_node_local_idx": Box(
                low=0,
                high=self.max_number_ego_nodes,
                shape=(1,),
                dtype=np.int64
            ),
            
            # For full-graph obs: all ones
            # For sub-graph obs later: 0/1 mask
            "node_visibility_mask": Box(
                low=0,
                high=1,
                shape=(N,),
                dtype=np.int8
            ),

            "edge_visibility_mask": Box(
                low=0.0, 
                high=1.0, 
                shape=(E,), 
                dtype=np.float32
            ),

            "action_mask": Box(
                low=0,
                high=1,
                shape=(A,),
                dtype=np.int8
            ),

            "neighbor_local_idx": Box(
                low=0,
                high=self.max_number_ego_nodes - 1,
                shape=(self.max_degree,),
                dtype=np.int64
            ),

            "neighbor_mask": Box(
                low=0.0,
                high=1.0,
                shape=(self.max_degree,),
                dtype=np.float32
            ),

            # V2 additions: counts for direct slicing (optional, for v2 observations)
            "num_visible_nodes": Box(
                low=0,
                high=N,
                shape=(1,),
                dtype=np.int64
            ),

            "num_visible_edges": Box(
                low=0,
                high=E,
                shape=(1,),
                dtype=np.int64
            )
        })

        return obs_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(self.num_actions)

    """
    Comments for self: make a custom ParallelEnv to SB3 VecEnv wrapper.
    Custom ActorCriticPolicy by SB3: takes in batched obs from the VecEnv wrapper and returns batched actions, log_probs and values.
    """

from graphpolicy import GraphPolicy
class GraphCoopEnv(ParallelEnv): #CoopEnv_v0 with policy_paths instead of policies.
    """
    This class inherits from ParallelEnv and provides a Cooperative environment with the opponent team policy fixed (as given by the input PolicySet) in the MixedCompCoop setting (the Capture-the-Flag environment).
    GOAL: This class should pass the ParallelEnv API test and should be a valid ParallelEnv class.
    """
    def __init__(self, MixedCompCoop, Policy: GraphPolicy, verbose=False, seed=None):
        from stable_baselines3 import PPO
        import numpy as np
        
        self.verbose = verbose
        assert isinstance(MixedCompCoop, GraphCTF)
        assert callable(Policy)
        #assert isinstance(Policy, GraphPolicy), "type of Policy: {}".format(Policy)

        self.metadata = { "name": "GraphCoopEnv"}
        self.render_mode = "human"
        self.num_teams = 1
        self.MixedCompCoop = MixedCompCoop
        self.OppPolicy = Policy
        self.OppPolicyType = 'Single'

        self.agent_deaths = False

        if seed is not None:
            self.seed(seed=seed)
        else:
            self._seed = None
            self.np_random = None
            self.torch_rng = None

        if self.OppPolicy.team == "Blue":
            self.passive_agents = copy.deepcopy(self.MixedCompCoop.blue_team_agents)
            self.team = "Red"
            self.num_agents = self.MixedCompCoop.num_agents_red_team
            self.agents = copy.deepcopy(self.MixedCompCoop.red_team_agents)
        elif self.OppPolicy.team == "Red":
            self.passive_agents = copy.deepcopy(self.MixedCompCoop.red_team_agents)
            self.team = "Blue"
            self.num_agents = self.MixedCompCoop.num_agents_blue_team
            self.agents = copy.deepcopy(self.MixedCompCoop.blue_team_agents)
        
        if self.verbose: print("self.agents INIT: {}".format(self.agents))
        self.possible_agents = copy.deepcopy(self.agents)
        if self.verbose: print(self.possible_agents)
        self.actors_in_the_env = copy.deepcopy(self.agents + self.passive_agents)
        
        self.current_step = 0
        self.state = {actor: None for actor in self.actors_in_the_env}

        self.observation_spaces = {agent: self.observation_space(agent=agent) for agent in self.agents}
        self.action_spaces = {agent: self.action_space(agent=agent) for agent in self.agents}

        self._init_args = (MixedCompCoop, Policy)
        self._init_kwargs = {"verbose": verbose, "seed": seed}

    def __call__(self):
        return GraphCoopEnv(*self._init_args, **self._init_kwargs)

    def seed(self, seed=None):
        self.rngs = make_seeded_rngs(seed)
        self._seed = self.rngs["actual_seed"]
        self.np_random = self.rngs["np_random"]
        self.torch_rng = self.rngs["torch_rng"]
        return [self._seed]

    def num_agents(self):
        return len(self.agents)

    def reset(self, seed = None, options = None):
        """
        Pending: if policy is a mixture, just sample a policy out of the mixture at every reset and use it for the rest of the episode.
        Just sample a policy index from the mixture and load that policy.
        Pending: For an implementation with just the addresses of the stored networks, load all the networks in the ensemble and store in a list.
        """
        if seed is not None:
            self.seed(seed=seed)

        if self.OppPolicyType == "mixture":
            # Sample a network idx from the mixture and set the self.OppPolicy global variable (None before setting).
            policy_network_idx = self.np_random.choice(self.OppPolicyMixtureSize, p=self.OppPolicyWeights)
            self.OppPolicy = self.OppPolicyMixture[policy_network_idx]
        
        self.current_step = 0
        self.agents = copy.deepcopy(self.possible_agents)
        obs, info = self.MixedCompCoop.reset()
        
        # Copy the MixedCompCoop state to the current environment state.
        for actor in self.actors_in_the_env:
            self.state[actor] = self.MixedCompCoop.state[actor]
        
        coop_obs = {agent: obs[agent] for agent in self.agents}
        coop_info = {agent: info[agent] for agent in self.agents}
        if self.verbose: print("self.agents: {}".format(self.agents))
        return coop_obs, coop_info
    
    def evalPolicy(self, policy, obs, agent):
        assert agent in self.passive_agents
        assert isinstance(obs, dict)
        assert "observation" in obs and "action_mask" in obs
        assert isinstance(policy, PPO)

        obs_tensor = th.as_tensor(obs["observation"]).unsqueeze(0)
        action_mask_tensor = th.as_tensor(obs["action_mask"]).unsqueeze(0)
        valid_actions = self.MixedCompCoop.valid_actions(self.MixedCompCoop.state, agent)
        
        with th.no_grad():
            features = policy.policy.extract_features({"observation": obs_tensor, "action_mask": action_mask_tensor})
            latent_pi, _ = policy.policy.mlp_extractor(features)
            distribution = policy.policy._get_action_dist_from_latent(latent_pi)  # get the action distribution

            # Apply action mask to the distribution
            logits = distribution.distribution.logits
            # Following commented on July 15 2025 @ 8.56 pm.
            """
            # Avoid log(0) by clamping values
            mask = (action_mask_tensor + 1e-8).log()
            masked_logits = logits + mask
            """
            masked_logits = logits.masked_fill(action_mask_tensor == 0, -1e10)
            distribution.distribution.logits = masked_logits

            # Get the predicted action (from the distribution now)
            dist = distribution.distribution  # For clarity

            if hasattr(dist, 'sample') and 'generator' in dist.sample.__code__.co_varnames:
                sample = dist.sample(generator=self.torch_rng)
            else:
                # fallback if generator is not accepted
                torch.manual_seed(self.torch_rng.initial_seed())
                sample = dist.sample()

            agent_action = sample.cpu().numpy()[0]

            #agent_action = distribution.distribution.sample(generator=self.torch_rng).cpu().numpy()[0] #distribution.get_actions(deterministic=False).cpu().numpy()[0] # Assuming batch_size = 1
            if self.verbose: print(agent_action)

        if agent_action not in valid_actions:
            if self.verbose: print("Policy predicted invalid action for agent !!!!!")
            agent_action = self.np_random.choice(valid_actions)

        return agent_action
    
    def valid_actions(self, agent):
        assert agent in self.agents
        return NotImplementedError #self.MixedCompCoop.valid_actions(self.MixedCompCoop.state[agent])

    def step(self, CoopActionsDict):
        if self.verbose:
            print("self.agents: {}".format(self.agents))
            print("len(CoopActionsDict.keys()) = {}".format(len(CoopActionsDict.keys())))
            print("self.num_agents = {}".format(self.num_agents))
        #assert len(CoopActionsDict.keys()) == self.num_agents

        for agent in self.agents:
            #assert CoopActionsDict[agent] in self.valid_actions(agent=agent), "CoopActionsDict[agent]: {}, self.valid_actions(agent=agent): {}".format(CoopActionsDict[agent], self.valid_actions(agent=agent))
            assert CoopActionsDict[agent] in self.action_space(agent=agent), "CoopActionsDict[agent]: {}, self.action_space(agent=agent): {}".format(CoopActionsDict[agent], self.action_space(agent=agent))
        assert self.OppPolicy is not None
        
        """
        Take in actions dict just for the Coop team, augments with actions for the opponent team from the input Policy and steps through the MixedCompCoop environment.
        """
        ActionsDict = {}

        # Batch opponent observation gathering and action computation
        # This reduces Python function call overhead
        active_passive_agents = []
        passive_obs_list = []

        for actor in self.actors_in_the_env:
            if self.agent_deaths:
                if not self.agent_alive[actor]:
                    continue
            if actor in self.agents:
                ActionsDict[actor] = CoopActionsDict[actor]
            elif actor in self.passive_agents:
                active_passive_agents.append(actor)
                passive_obs_list.append(self.MixedCompCoop.get_observation_v1(agent=actor))

        # Batch action computation for opponents
        if hasattr(self.OppPolicy, 'batch_action') and len(passive_obs_list) > 0:
            passive_actions = self.OppPolicy.batch_action(passive_obs_list)
            for actor, action in zip(active_passive_agents, passive_actions):
                ActionsDict[actor] = action
        else:
            # Fallback to sequential if batch_action not available
            for actor, obs in zip(active_passive_agents, passive_obs_list):
                ActionsDict[actor] = self.OppPolicy(obs)
        
        obs, rewards, terminations_, truncations_, infos_ = self.MixedCompCoop.step(ActionsDict)

        obs_coop = {agent: obs[agent] for agent in self.agents}
        rewards_coop = {agent: rewards[agent] for agent in self.agents}
        terminations = {agent: terminations_[agent] for agent in self.agents}
        truncations = {agent: truncations_[agent] for agent in self.agents}
        infos = {agent: infos_[agent] for agent in self.agents}

        for actor in self.actors_in_the_env:
            self.state[actor] = self.MixedCompCoop.state[actor]
            if self.agent_deaths: self.agent_alive[actor] = self.MixedCompCoop.agent_alive[actor]
        self.current_step += 1

        agents_to_remove = []
        for agent in self.agents:
            if terminations[agent] or truncations[agent]:
                agents_to_remove.append(agent)

        # Remove done agents
        for agent in agents_to_remove:
            self.agents.remove(agent)
        
        if self.verbose:
            print(obs_coop)
            print(rewards_coop)
            print(terminations)
            print(truncations)
            print(infos)

        return obs_coop, rewards_coop, terminations, truncations, infos
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        assert agent in self.agents
        #return super().action_space(agent)
        return self.MixedCompCoop.action_space(agent)
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        assert agent in self.agents
        return self.MixedCompCoop.observation_space(agent)

import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from gymnasium.spaces import Box, Dict

class GraphParallelEnvToSB3VecEnv_v0(VecEnv):
    """
    Wrap multiple ParallelEnvs (GraphCTF) into SB3 VecEnv.
    Each agent in each environment is treated as an independent environment.
    Works with dict observations.
    """

    def __init__(self, parallel_env_class, num_envs=1, env_kwargs=None):
        if env_kwargs is None:
            env_kwargs = {}

        self.envs = [parallel_env_class(**env_kwargs) for _ in range(num_envs)]
        self.num_agents = len(self.envs[0].possible_agents)

        # Flattened environments = num_envs * num_agents
        self.num_envs_total = num_envs * self.num_agents

        # Use first env to infer spaces
        obs_space = self.envs[0].observation_space(self.envs[0].possible_agents[0])
        act_space = self.envs[0].action_space(self.envs[0].possible_agents[0])

        super().__init__(num_envs=self.num_envs_total,
                         observation_space=obs_space,
                         action_space=act_space)

        self._actions = None

    # ---------------------------
    # VecEnv API
    # ---------------------------
    def reset(self, **kwargs):
        obs_batches = []
        for env in self.envs:
            obs_dict, _ = env.reset(**kwargs)
            obs_batches.append(self._stack_agent_obs(obs_dict, env.possible_agents))
        return self._stack_env_obs(obs_batches)

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs_batches = []
        reward_batches = []
        done_batches = []
        info_batches = []

        offset = 0
        for env in self.envs:
            action_dict = {agent: int(self._actions[offset + i])
                        for i, agent in enumerate(env.possible_agents)}
            offset += len(env.possible_agents)

            obs, rewards, terminations, truncations, infos = env.step(action_dict)

            # Ensure infos is always a dict
            if infos is None:
                infos = {agent: {} for agent in env.possible_agents}
            else:
                infos = {agent: infos.get(agent, {}) for agent in env.possible_agents}

            done_dict = {agent: terminations[agent] or truncations[agent]
                        for agent in env.possible_agents}

            obs_batches.append(self._stack_agent_obs(obs, env.possible_agents))
            reward_batches.append(np.array([rewards[a] for a in env.possible_agents], dtype=np.float32))
            done_batches.append(np.array([done_dict[a] for a in env.possible_agents], dtype=np.bool_))

            # GUARANTEE every info is a dict
            for agent in env.possible_agents:
                info_batches.append(infos[agent])

        return (self._stack_env_obs(obs_batches),
                np.concatenate(reward_batches, axis=0),
                np.concatenate(done_batches, axis=0),
                info_batches)  # <-- all entries are guaranteed dicts


    def close(self):
        for env in self.envs:
            env.close()

    # ---------------------------
    # Helpers
    # ---------------------------
    def _stack_agent_obs(self, obs_dict, agents):
        """
        Stack a single environment's agents observations.
        Returns a dict {key -> np.array[num_agents, ...]} or numeric array.
        """
        # Keys that should remain int64 for indexing
        int_keys = {'neighbor_local_idx', 'edge_index', 'agent_node_local_idx'}

        if isinstance(next(iter(obs_dict.values())), dict):
            stacked = {}
            for k in obs_dict[agents[0]].keys():
                dtype = np.int64 if k in int_keys else np.float32
                arrs = [np.array(obs_dict[a][k], dtype=dtype) for a in agents]
                stacked[k] = np.stack(arrs, axis=0)
            return stacked
        else:
            return np.stack([np.array(obs_dict[a], dtype=np.float32) for a in agents], axis=0)

    def _stack_env_obs(self, obs_batches):
        """
        Stack observations across multiple environments.
        obs_batches: list of dicts or arrays
        Returns: dict of arrays (preserving dtypes for index keys)
        """
        # Keys that should remain int64 for indexing
        int_keys = {'neighbor_local_idx', 'edge_index', 'agent_node_local_idx'}

        if isinstance(obs_batches[0], dict):
            stacked = {}
            for k in obs_batches[0].keys():
                dtype = np.int64 if k in int_keys else np.float32
                arrs = [np.array(obs[k], dtype=dtype) for obs in obs_batches]
                stacked[k] = np.concatenate(arrs, axis=0)
            return stacked
        else:
            return np.concatenate([np.array(obs, dtype=np.float32) for obs in obs_batches], axis=0)

        # ---------------------------
    # Helpers
    # ---------------------------
    def env_is_wrapped(self, wrapper_class):
        return np.array([isinstance(env, wrapper_class) for env in self.envs])

    def env_method(self, method_name, *args, indices=None, **kwargs):
        if indices is None:
            indices = range(len(self.envs))
        results = []
        for i in indices:
            method = getattr(self.envs[i], method_name)
            results.append(method(*args, **kwargs))
        return results

    def get_attr(self, attr_name, indices=None):
        if indices is None:
            indices = range(len(self.envs))
        return [getattr(self.envs[i], attr_name) for i in indices]

    def set_attr(self, attr_name, value, indices=None):
        if indices is None:
            indices = range(len(self.envs))
        for i in indices:
            setattr(self.envs[i], attr_name, value)

    def seed(self, seed=None):
        self.rngs = make_seeded_rngs(seed)
        self._seed = self.rngs["actual_seed"]
        self.np_random = self.rngs["np_random"]
        self.torch_rng = self.rngs["torch_rng"]
        return [self._seed]

class GraphParallelEnvToSB3VecEnv_v1(VecEnv):
    """
    Wrap multiple ParallelEnvs (GraphCTF) into a SB3-compatible VecEnv.
    Each agent in each environment is treated as an independent environment.
    Handles early termination/truncation by immediately resetting the env.
    """

    """
    v1 fixes over v0: 
    a) agent renaming for batching in vecenv to avoid key errors in sb3 - rename blue_0 to somethig like env_5_blue_0.
    b) agent deaths in some envs in the batched vecenv -- reset the dead envs in step_wait
    """

    """
    Pending: Correct seeding logic for the VecEnv. Sub-graph observations. Use observation features to learn invariant embeddings, invariant to flip, rotation, reflection, translation -- use more complicated features such as Graph laplacians et cetera.
    """
    def __init__(self, parallel_env_class, num_envs=1, env_args=(), env_kwargs={}):
        if env_kwargs is None:
            env_kwargs = {}

        self.envs = [parallel_env_class(*env_args, **env_kwargs) for _ in range(num_envs)]

        # In BR training the env has an opponent policy set; only the active team
        # is controlled by SB3.  Exposing all 4 agents causes zero-sum reward
        # cancellation (Blue +F, Red -F → mean = 0).  Use only active agents.
        first_env = self.envs[0]
        if first_env.has_opponent_policy():
            _train_agents = first_env.get_active_agents()
        else:
            _train_agents = first_env.possible_agents
        self.num_agents = len(_train_agents)
        self.num_envs = num_envs * self.num_agents

        # Flattened agent keys for the batch
        self.agents_flat = [
            f"env{env_idx}_agent{agent_idx}"
            for env_idx in range(num_envs)
            for agent_idx in range(self.num_agents)
        ]

        # Observation/action space from the first active agent of first env
        obs_space = first_env.observation_space(_train_agents[0])
        act_space = first_env.action_space(_train_agents[0])

        super().__init__(num_envs=self.num_envs,
                         observation_space=obs_space,
                         action_space=act_space)

        self._actions = None

    # ---------------------------
    # VecEnv API
    # ---------------------------
    """
    def reset(self, **kwargs):
        obs_list = []
        for env in self.envs:
            obs_dict, _ = env.reset(**kwargs)
            for agent in env.possible_agents:
                obs_list.append(obs_dict[agent])
        return np.stack(obs_list, axis=0)
    """
    def reset(self, **kwargs):
        obs_batches = []
        for env in self.envs:
            obs_dict, _ = env.reset(**kwargs)
            train_agents = env.get_active_agents() if env.has_opponent_policy() else env.possible_agents
            obs_batches.append(self._stack_agent_obs(obs_dict, train_agents))
        return self._stack_env_obs(obs_batches)

    def step_async(self, actions):
        self._actions = actions

    def step_wait_v0(self):
        obs_list = []
        rewards_list = []
        dones_list = []
        infos_list = []

        offset = 0
        for env in self.envs:
            train_agents = env.get_active_agents() if env.has_opponent_policy() else env.possible_agents

            # Build action dict only for active agents; env.step fills in opponent actions.
            action_dict = {
                agent: int(self._actions[offset + i])
                for i, agent in enumerate(train_agents)
            }
            offset += len(train_agents)

            # Step env
            obs, rewards, terminations, truncations, infos = env.step(action_dict)

            # Ensure infos is never None
            if infos is None:
                infos = {agent: {} for agent in env.possible_agents}
            else:
                infos = {agent: infos.get(agent, {}) for agent in env.possible_agents}

            for agent in train_agents:
                done = terminations[agent] or truncations[agent]
                dones_list.append(done)
                obs_list.append(obs[agent])
                rewards_list.append(rewards[agent])
                infos_list.append(infos[agent])

            # Auto-reset when episode ends.
            # Keep done=True and the terminal reward; replace obs with post-reset obs.
            # Store terminal_observation in infos for correct SB3 value bootstrapping.
            if any(terminations.values()) or any(truncations.values()):
                obs_reset, _ = env.reset()
                n_train = len(train_agents)
                for i, agent in enumerate(train_agents):
                    idx = offset - n_train + i
                    if infos_list[idx] is None:
                        infos_list[idx] = {}
                    infos_list[idx]["terminal_observation"] = obs_list[idx]
                    obs_list[idx] = obs_reset[agent]

        return (
            np.stack(obs_list, axis=0),
            np.array(rewards_list, dtype=np.float32),
            np.array(dones_list, dtype=np.bool_),
            infos_list
        )

    def step_wait(self):
        obs_batches = []
        rewards_list = []
        dones_list = []
        infos_list = []

        offset = 0
        for env in self.envs:
            # In BR training only the active team is controlled by SB3; the
            # opponent's actions are computed internally by env.step().
            train_agents = env.get_active_agents() if env.has_opponent_policy() else env.possible_agents

            # Build action dict only for active agents
            action_dict = {agent: int(self._actions[offset + i])
                        for i, agent in enumerate(train_agents)}
            offset += len(train_agents)

            # Step env (opponent actions merged internally)
            obs, rewards, terminations, truncations, infos = env.step(action_dict)

            # Ensure infos is never None
            if infos is None:
                infos = {agent: {} for agent in env.possible_agents}
            else:
                infos = {agent: infos.get(agent, {}) for agent in env.possible_agents}

            # Inject episode_flag_idx into every step's infos (§42).
            # Read BEFORE any potential env.reset() below so terminal steps
            # still carry the just-completed episode's flag, not the next one.
            flag_idx_ep = getattr(env, '_episode_flag_idx', None)
            if flag_idx_ep is not None:
                for agent in train_agents:
                    infos[agent]['episode_flag_idx'] = flag_idx_ep

            # Convert per-agent obs to numeric dicts/arrays (active agents only)
            obs_batch = self._stack_agent_obs(obs, train_agents)
            obs_batches.append(obs_batch)

            for agent in train_agents:
                done = terminations[agent] or truncations[agent]
                dones_list.append(done)
                rewards_list.append(rewards[agent])
                infos_list.append(infos[agent])

            # Auto-reset when episode ends.
            # Keep done=True and the terminal reward so SB3 and callbacks see
            # the episode boundary correctly (needed for GAE and reward tracking).
            # Store the terminal observation in infos["terminal_observation"] so
            # SB3 can use it for correct value bootstrapping at episode end.
            if any(terminations.values()) or any(truncations.values()):
                terminal_obs_batch = obs_batches[-1]  # obs at terminal step
                obs_reset, _ = env.reset()
                obs_batch_reset = self._stack_agent_obs(obs_reset, train_agents)
                obs_batches[-1] = obs_batch_reset  # post-reset obs for next step
                # Inject terminal_observation into each agent's info dict.
                # episode_flag_idx was already injected above for all steps.
                n_train = len(train_agents)
                for j in range(n_train):
                    idx = -n_train + j
                    if infos_list[idx] is None:
                        infos_list[idx] = {}
                    infos_list[idx]["terminal_observation"] = {
                        k: v[j] for k, v in terminal_obs_batch.items()
                    }

        # Stack all environments
        final_obs = self._stack_env_obs(obs_batches)
        return final_obs, np.array(rewards_list, dtype=np.float32), np.array(dones_list, dtype=np.bool_), infos_list

    def close(self):
        for env in self.envs:
            env.close()

    # ---------------------------
    # Helpers
    # ---------------------------
    def _stack_agent_obs(self, obs_dict, agents):
        """
        Stack a single environment's agents observations.
        Returns a dict {key -> np.array[num_agents, ...]} of numeric arrays.
        """
        # Keys that should remain int64 for indexing
        int_keys = {'neighbor_local_idx', 'edge_index', 'agent_node_local_idx'}

        if isinstance(next(iter(obs_dict.values())), dict):
            stacked = {}
            for k in obs_dict[agents[0]].keys():
                # print("k: {}".format(k))
                dtype = np.int64 if k in int_keys else np.float32
                arrs = []
                for a in agents:
                    # print("agent a: {}".format(a))
                    val = obs_dict[a][k]
                    if isinstance(val, list):
                        val = np.array(val, dtype=dtype)
                    elif np.isscalar(val):
                        val = np.array([val], dtype=dtype)
                    else:
                        val = np.array(val, dtype=dtype)
                    arrs.append(val)
                    # print("val shape: \n {}".format(val.shape))
                stacked[k] = np.stack(arrs, axis=0)
            return stacked
        else:
            arrs = []
            for a in agents:
                val = obs_dict[a]
                if isinstance(val, list):
                    val = np.array(val, dtype=np.float32)
                elif np.isscalar(val):
                    val = np.array([val], dtype=np.float32)
                arrs.append(val)
            return np.stack(arrs, axis=0)


    def _stack_env_obs(self, obs_batches):
        """
        Stack observations across multiple environments.
        obs_batches: list of dicts or arrays
        Returns: dict of arrays (preserving dtypes for index keys)
        """
        # Keys that should remain int64 for indexing
        int_keys = {'neighbor_local_idx', 'edge_index', 'agent_node_local_idx'}

        if isinstance(obs_batches[0], dict):
            stacked = {}
            for k in obs_batches[0].keys():
                dtype = np.int64 if k in int_keys else np.float32
                arrs = []
                for obs in obs_batches:
                    val = obs[k]
                    if isinstance(val, list):
                        val = np.array(val, dtype=dtype)
                    arrs.append(val)
                stacked[k] = np.concatenate(arrs, axis=0)
            return stacked
        else:
            arrs = []
            for obs in obs_batches:
                val = obs
                if isinstance(val, list):
                    val = np.array(val, dtype=np.float32)
                arrs.append(val)
            return np.concatenate(arrs, axis=0)

    def env_is_wrapped(self, wrapper_class):
        return np.array([isinstance(env, wrapper_class) for env in self.envs])

    def env_method(self, method_name, *args, indices=None, **kwargs):
        if indices is None:
            indices = range(len(self.envs))
        results = []
        for i in indices:
            method = getattr(self.envs[i], method_name)
            results.append(method(*args, **kwargs))
        return results

    def get_attr(self, attr_name, indices=None):
        if indices is None:
            indices = range(len(self.envs))
        return [getattr(self.envs[i], attr_name) for i in indices]

    def set_attr(self, attr_name, value, indices=None):
        if indices is None:
            indices = range(len(self.envs))
        for i in indices:
            setattr(self.envs[i], attr_name, value)


# =========================================================
# MAPPO VecEnv Wrapper
# =========================================================

_TEAMMATE_COPY_KEYS = (
    'x', 'node_visibility_mask', 'edge_visibility_mask',
    'edge_index', 'edge_attr', 'agent_node_local_idx',
)


class GraphParallelEnvToSB3VecEnv_MAPPO(GraphParallelEnvToSB3VecEnv_v1):
    """
    Extends GraphParallelEnvToSB3VecEnv_v1 for MAPPO (Centralized Training,
    Decentralized Execution).

    Each agent's observation is augmented with:
      - teammate_{key}  for each key in _TEAMMATE_COPY_KEYS
      - agent_id        team-local integer (0 or 1) for actor-head routing

    The observation_space is extended accordingly.

    Assumptions:
      - Exactly 2 agents per team (Blue_0/Blue_1 or Red_0/Red_1).
      - Agent names start with 'B' (Blue) or 'R' (Red).
      - `possible_agents` ordering is stable across all envs in the pool.
    """

    def __init__(self, parallel_env_class, num_envs=1, env_args=(), env_kwargs={},
                 active_team='Blue'):
        super().__init__(parallel_env_class, num_envs, env_args, env_kwargs)
        first_env = self.envs[0]

        # Select active-team agents by name at construction time, independently
        # of has_opponent_policy() (opponents are set after construction).
        # active_team[0] gives 'B' or 'R' for prefix matching.
        active_agents = [a for a in first_env.possible_agents
                         if a.startswith(active_team[0])]
        n_active = len(active_agents)

        # Build _teammate_idx_map and _agent_local_id using loop indices 0..n-1.
        # _stack_agent_obs loops `for i in range(n_active)`, so keys must be
        # those indices — not positions inside possible_agents.
        self._teammate_idx_map = {i: (i + 1) % n_active for i in range(n_active)}
        self._agent_local_id   = {i: i for i in range(n_active)}

        # Fix num_agents / num_envs / agents_flat at construction time so that
        # VecNormalize and the PPO rollout buffer always see the right sizes,
        # regardless of whether has_opponent_policy() was True at super().__init__.
        n_underlying = len(self.envs)
        self.num_agents  = n_active
        self.num_envs    = n_underlying * n_active
        self.agents_flat = [
            f"env{env_idx}_agent{agent_idx}"
            for env_idx in range(n_underlying)
            for agent_idx in range(n_active)
        ]

        # Extend observation_space with teammate_* keys + agent_id
        from gymnasium.spaces import Dict as GymDict, Box as GymBox
        base  = self.observation_space
        extra = {
            f'teammate_{k}': base.spaces[k]
            for k in _TEAMMATE_COPY_KEYS
            if k in base.spaces
        }
        extra['agent_id'] = GymBox(low=0, high=1, shape=(1,), dtype=np.int64)
        self.observation_space = GymDict({**base.spaces, **extra})

    def _stack_agent_obs(self, obs_dict, agents):
        """
        Calls the base stacker then injects teammate observations and agent_id.

        The base class returns stacked[key] with shape [n, ...] where index i
        corresponds to the agent at position i in `agents` (= possible_agents).
        We copy each key from the teammate's slot to produce teammate_{key}.
        """
        stacked = super()._stack_agent_obs(obs_dict, agents)
        n = len(agents)

        # Inject teammate observations
        for key in _TEAMMATE_COPY_KEYS:
            if key not in stacked:
                continue
            src    = stacked[key]  # [n, ...]
            tm_arr = np.empty_like(src)
            for i in range(n):
                tm_arr[i] = src[self._teammate_idx_map.get(i, (i + 1) % n)]
            stacked[f'teammate_{key}'] = tm_arr

        # Inject team-local agent id
        stacked['agent_id'] = np.array(
            [[self._agent_local_id.get(i, 0)] for i in range(n)],
            dtype=np.int64
        )
        return stacked


class CustomCTF_v0(ParallelEnv): # v0: HOMOGENOUS OBSERVATION SPACES FOR RED AND BLUE TEAMS: Full information case.
    """
    Blue agents policy should be independent of the red flag location. It should just be agents.
    Pending: Debug the partial information case.
    Pending: Blue agent policy should not be a function of the red_flag_location. Blue agent does not know where the red_flag_is: POSMG. It's policy is just a function of home flag location and all the agent locations, not the opponent team flag location.
    !!!!! PENDING: Change the observation space of Blue Team agents for the deceptive CTF environment, or the Partially Observable CTF environment.
    #### Pending: Look at methods for partially observable Games == what changes for PSRO.
    #### Pending: Blue team not knowing Red Flag location at all and learning a behaviour just from interactions versus it having a belief over red_flag_locations.
    """
    meta_data = { "name": "custom_ctf_v0"}
    def __init__(self, grid_size=8, ctf_player_config="2v2", max_num_cycles=300, red_flag_locs=(6,6), verbose=False, seed=None):
        # Two swarms: red and blue.
        # Nash DQN as initial guess. Then warm start policy synthesis.
        # CTF on a grid of size [grid_size \times grid_size]
        self.metadata = { "name": "custom_ctf_v0"}
        self.render_mode = "human"
        self.verbose = verbose

        if seed is not None:
            self.seed(seed=seed)
        else:
            self._seed = None # dummy value.
            self.np_random = None
            self.torch_rng = None

        self.grid_size = grid_size #this defines the state space of the Markov Game and hence the observation_space.
        self.num_headings = 8  # Possible headings in degrees:(0, 45, 90, 135, 180, 225, 270, 315)
        self.num_actions = 5
        self.max_num_cycles = max_num_cycles

        self.num_teams = 2
        if ctf_player_config == "1v1":
            self.num_agents_blue_team = 1
        elif ctf_player_config == "2v2":
            self.num_agents_blue_team = 2
        self.num_agents_red_team = self.num_agents_blue_team
        self.num_agents = self.num_agents()

        self.blue_team_agents = ["Blue_" + "{}".format(i) for i in range(self.num_agents_blue_team)]
        self.red_team_agents = ["Red_" + "{}".format(i) for i in range(self.num_agents_red_team)]

        #self.agents = ["Blue_" + "{}".format(i) for i in range(self.num_agents_blue_team)] + ["Red_" + "{}".format(i) for i in range(self.num_agents_red_team)]
        self.agents = copy.deepcopy(self.blue_team_agents + self.red_team_agents)
        if self.verbose: print("self.agents: {}".format(self.agents))
        self.possible_agents = copy.deepcopy(self.agents)
        if self.verbose: print(self.possible_agents)

        self.observation_spaces = {agent: self.observation_space(agent=agent)["observation"] for agent in self.agents}
        self.action_spaces = {agent: self.action_space(agent=agent) for agent in self.agents}

        """
        For v1 of the CustomCTF environment: Can add logic to have a jitter / randomness in the flag locations.
        """
        self.deceptive_env = False #Note: Deception refers to uncertainty in the red_flag_location.
        self.blue_flag_location = (1, 1)
        if red_flag_locs is None: self.red_flag_location = (self.grid_size - 2, self.grid_size - 2)
        elif (red_flag_locs is not None) and isinstance(red_flag_locs, tuple):
            self.red_flag_location = red_flag_locs
        elif isinstance(red_flag_locs, dict):
            assert list(red_flag_locs.keys()) == ['locs', 'p']
            assert isinstance(red_flag_locs['locs'], list) and isinstance(red_flag_locs['locs'][0], tuple) and isinstance(red_flag_locs['p'], list)
            assert len(red_flag_locs['locs']) == len(red_flag_locs['p'])
            assert abs(sum(red_flag_locs['p']) - 1) <= 1e-10
            """
            For instance, red_flag_locs = { 'locs': [(6,6), (3,7), (5,5)], 'p': [0.7, 0.2, 0.1] }
            """
            self.deceptive_env = True
            self.red_flag_locs = red_flag_locs
            self.red_flag_location = None

        self.blue_init_spawn_y_lim = 2 # Measured from bottom of the grid.
        self.possible_init_headings_blue_team = [0, 1, 2, 3] # possible init headings = (0, 45, 90, 135)
        self.red_init_spawn_y_lim = 2 # Measured from top of the grid.
        self.possible_init_headings_red_team = [0, 7, 6, 5] # possible init headings = (0, -45, -90, -135)

        # Tag spawning area is a rectangle, away from the flag in a corner.
        self.blue_tag_spawn_area = {"x_lim":(5, 7), "y_lim": (0, 1)} # (x, y) coordinate tuple -- x_low y_low and x_high y_high
        self.red_tag_spawn_area = {"x_lim":(0, 2), "y_lim": (6, 7)} # (x, y) coordinate tuple -- x_low y_low and x_high y_high

        self.current_step = 0
        self.state = { agent: None for agent in self.agents } #self.state = np.array([None for _ in self.agents])
        #self.reset()

        self.F = 100 #self.capture_flag_reward = 100
        self.tag_reward_shape = True
        self.T = 10 #10 #self.tagging_penalty = 10

        self.blue_flag_img_path = 'flag_imgs/blue_flag.png'
        self.red_flag_img_path = 'flag_imgs/red_flag.png'

        self.blue_flag_img = mpimg.imread('flag_imgs/blue_flag.png')
        self.red_flag_img = mpimg.imread('flag_imgs/red_flag.png')

        self.flag_capture_team = None

        self._init_args = ()
        self._init_kwargs = {"grid_size":grid_size, "ctf_player_config":ctf_player_config, "max_num_cycles":max_num_cycles, "red_flag_locs":red_flag_locs, "verbose":verbose, "seed":seed}
    
    def __call__(self):
        return CustomCTF_v0(*self._init_args, **self._init_kwargs)

    def seed(self, seed=None):
        self.rngs = make_seeded_rngs(seed)
        self._seed = self.rngs["actual_seed"]
        self.np_random = self.rngs["np_random"]
        self.torch_rng = self.rngs["torch_rng"]
        return [self._seed]

    def agent_idx(self, agent_name):
        # Returns the agent_idx according to the self.agents list for the input agent_name.
        return

    def state(self):
        # Returns the global state as a Dict or an array?
        # I think the answer depends on the type of Agent names, if those are int then array otherwise Dict.
        return self.state

    def _sample_init_heading(self, agent_team, x, y):
        assert agent_team == "Blue" or agent_team == "Red"
        if agent_team == "Blue":
            possible_init_headings = copy.deepcopy(self.possible_init_headings_blue_team)
            if x == 0: possible_init_headings.remove(3)
            if x == self.grid_size - 1:
                possible_init_headings.remove(0)
                possible_init_headings.remove(1)
            heading = self.np_random.choice(possible_init_headings) #np.random.choice(possible_init_headings)
        elif agent_team == "Red":
            possible_init_headings = copy.deepcopy(self.possible_init_headings_red_team)
            if x == 0: possible_init_headings.remove(5)
            if x == 1:
                possible_init_headings.remove(0)
                possible_init_headings.remove(7)
            heading = self.np_random.choice(possible_init_headings)
        return heading

    """
    def clip_to_valid_heading(self, pos):
        x, y, heading = pos
        if x == 0:
          if y > 0 and y < self.grid_size - 1:
            if heading == 3: heading = 2
            if heading == 4: heading = 6
            if heading == 5: heading = 6
          elif y == 0: pass
          elif y == self.grid_:
            if heading == 3: heading = 2
        if x == self.grid_size - 1: pass
        return pos
    """

    def reset(self, seed = None, options = None):
      #### Debugging prints: print valid actions for each agent at the step. => a) actions available at the walls b) tagging action available or not.
        if seed is not None:
            self.seed(seed=seed)
            """
            np.random.seed(seed=seed)
            import torch
            torch.manual_seed(seed=seed)
            random.seed(seed=seed)
            """
        self.current_step = 0
        self.agents = copy.deepcopy(self.blue_team_agents + self.red_team_agents) # !!! 04/25 3.56 pm: DON'T UNDERSTAND THIS BUG.
        """
        NOTE: In the current implementation, v0, flag locations are parameters of the Markov game and are hence not a part of the state, agent-wise or global.
        For v1: Can add logic to have a jitter / randomness in the flag locations.
        """
        #self.blue_flag_location = (1, 1)
        #self.red_flag_location = (self.grid_size - 2, self.grid_size - 2) # We assume the grid_size > 3 or something like that for the flag placements to be a non-trivial game scenario.

        self.blue_flag_location = self.blue_flag_location
        if self.deceptive_env:
            red_flag_idx = self.np_random.choice(len(self.red_flag_locs['locs']), p=self.red_flag_locs['p'])
            self.red_flag_location = self.red_flag_locs['locs'][red_flag_idx]

        blue_team_init_xys = []
        for agent_iter, blue_agent_num in enumerate(range(self.num_agents_blue_team)):
            if agent_iter == 0:
                rand_x = self.np_random.integers(0, self.grid_size) #self.np_random.randint(0, self.grid_size)
                rand_y = self.np_random.integers(0, self.blue_init_spawn_y_lim + 1)
                rand_theta = self._sample_init_heading(agent_team="Blue", x=rand_x, y=rand_y)
                self.state["Blue_{}".format(blue_agent_num)] = np.array([rand_x, rand_y, rand_theta])
                blue_team_init_xys.append((rand_x, rand_y))
                continue
            # A while loop to break ties in the x-y location to prevent collision during agent initialization.
            while (rand_x, rand_y) in blue_team_init_xys:
                rand_x = self.np_random.integers(0, self.grid_size)
                rand_y = self.np_random.integers(0, self.blue_init_spawn_y_lim + 1)
            blue_team_init_xys.append((rand_x, rand_y))
            rand_theta = self._sample_init_heading(agent_team="Blue", x=rand_x, y=rand_y)
            self.state["Blue_{}".format(blue_agent_num)] = np.array([rand_x, rand_y, rand_theta])

        red_team_init_xys = []
        for agent_iter, red_agent_num in enumerate(range(self.num_agents_red_team)):
            if agent_iter == 0:
                rand_x = self.np_random.integers(0, self.grid_size)
                rand_y = self.np_random.integers(self.grid_size - self.red_init_spawn_y_lim - 1, self.grid_size)
                rand_theta = self._sample_init_heading(agent_team="Red", x=rand_x, y=rand_y)
                self.state["Red_{}".format(red_agent_num)] = np.array([rand_x, rand_y, rand_theta])
                red_team_init_xys.append((rand_x, rand_y))
                continue
            # A while loop to break ties in the x-y location to prevent collision during agent initialization.
            while (rand_x, rand_y) in red_team_init_xys:
                rand_x = self.np_random.integers(0, self.grid_size)
                rand_y = self.np_random.integers(self.grid_size - self.red_init_spawn_y_lim - 1, self.grid_size)
            red_team_init_xys.append((rand_x, rand_y))
            rand_theta = self._sample_init_heading(agent_team="Red", x=rand_x, y=rand_y)
            self.state["Red_{}".format(red_agent_num)] = np.array([rand_x, rand_y, rand_theta])

        #global_state = [self.state[agent] for agent in self.agents]
        #obs = {agent: global_state for agent in self.agents} # global_state dict. The prisoner guard example on ParallelEnv has this as a tuple, and you have this as a List currently.
        obs = self._observations()
        info = {agent: {} for agent in self.agents}
        #print("self.agents: {}".format(self.agents))
        return obs, info

    def step(self, actions):
      #### Debugging prints: print valid actions for each agent at the step. => a) actions available at the walls b) tagging action available or not.
        ### v0: No collision logic.
        # TO-DO for v1: Resolve collisions and tagging and re-spawning. # Collisions within a team prevented via action_masking. Where to write actions_available_to_an_agent?

        # The input actions is a Dict from agent_name to actions. Add an assert statement for Type of actions.
        """
        Action keys:
        0 = same pos, same heading
        1 = same pos, minus heading
        2 = same pos, plus heading
        3 = advance pos, same heading
        4 = tagging action
        """
        if self.verbose: print("Current Step: {}".format(self.current_step))
        if self.verbose: print("Before stepping self.agents state: {}".format(self.state))
        if self.verbose: print("Actions: {}".format(actions))

        infos = {agent: {} for agent in self.agents}
        for agent in self.agents:
            infos[agent]["CTF"] = None # "CTF" descriptor is for the last step of the game on which team captured the flag if any!
        tag_bool = {agent: False for agent in self.agents}
        reach_goal_bool = {agent: False for agent in self.agents}
        new_state = {agent: None for agent in self.agents}

        rewards = {agent: 0. for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        for agent in self.agents:
          assert actions[agent] in self.valid_actions(self.state, agent)

        # Actions dict = actions.
        for agent in self.agents:
            assert actions[agent] in self.action_space(agent=agent)
            tag_bool[agent], reach_goal_bool[agent], new_state[agent] = self.dynamics(agent=agent, actions=actions)
            #print("agent: {}, tag_bool_agent: {}".format(agent, tag_bool[agent]))

        for agent in self.agents:
            self.state[agent] = new_state[agent]

        blue_tag = np.sum([ tag_bool[agent] for agent in self.blue_team_agents ])
        red_tag = np.sum([ tag_bool[agent] for agent in self.red_team_agents ])
        
        if self.verbose:
            for agent in self.blue_team_agents: print("blue tag check : {}".format(tag_bool[agent]))
            for agent in self.red_team_agents: print("red tag check : {}".format(tag_bool[agent]))
            for agent in self.blue_team_agents: print("blue reach_goal check : {}".format(reach_goal_bool[agent]))
            for agent in self.red_team_agents: print("red reach_goal check : {}".format(reach_goal_bool[agent]))
        
        blue_reach_goal = np.sum([ reach_goal_bool[agent] for agent in self.blue_team_agents ])
        red_reach_goal = np.sum([ reach_goal_bool[agent] for agent in self.red_team_agents ])

        if blue_reach_goal + red_reach_goal == 0:
            # Flag not captured.
            terminate = False
            terminations = {agent: terminate for agent in self.agents}
            # Assign tagging rewards if any.
            for agent in self.blue_team_agents:
                rewards[agent] +=  - blue_tag * (self.T)
                rewards[agent] += + red_tag * (self.T)
            for agent in self.red_team_agents:
                rewards[agent] += + blue_tag * (self.T)
                rewards[agent] += - red_tag * (self.T)
        else:
            # Flag captured.
            terminate = True
            terminations = {agent: terminate for agent in self.agents}
            for agent in self.blue_team_agents:
                if blue_reach_goal:
                    rewards[agent] += self.F
                    infos[agent]["CTF"] = True
                    self.flag_capture_team = "Blue"
                elif red_reach_goal:
                    rewards[agent] += - self.F
                    infos[agent]["CTF"] = False
                    self.flag_capture_team = "Red"
            for agent in self.red_team_agents:
                if blue_reach_goal:
                    rewards[agent] += - self.F
                    infos[agent]["CTF"] = False
                elif red_reach_goal:
                    rewards[agent] += self.F
                    infos[agent]["CTF"] = True
                pass

        # TO-DO: Determine rewards. Reward structure: team reward: agents share the same reward within a team.
        obs = self._observations()

        self.current_step += 1
        if self.current_step == self.max_num_cycles: truncations = {agent: True for agent in self.agents}

        agents_to_remove = []
        for agent in self.agents:
            if terminations[agent] or truncations[agent]:
                agents_to_remove.append(agent)

        # Remove done agents
        for agent in agents_to_remove:
            self.agents.remove(agent)
        if self.verbose:
            print("Updated self.agents state: {}".format(self.state))
            print("rewards: {}".format(rewards))
            print("terminations: {}".format(terminations))
            print("truncations: {}".format(truncations))
        return obs, rewards, terminations, truncations, infos

    def global_state_to_agent_local_obs(self, agent):
      """
      Hard-coded for a 2v2 scenario.
      For more agents == GNN.
      ### For the second iteration of the PSRO loop: CTDE? Fixing one team behaviour, have a CTDE scheme for the other team.
      """
      """
      Idea: Permutation invariance for better learning: ordering of adversaries shouldn't matter. Same strategy.
      """
      # coordinate of agent, heading, home_flag_rel_location, away_flag_rel_location, friend_rel_location, friend_rel_pose, adversary_1_rel_location, adversary_1_rel_pose, adversary_2_rel_location, adversary_2_rel_pose
      # 3 = self coordinate, 2 = home flag location, 2 = away flag location, 3 = friend rel pos, 3 = adversary 1 rel pos, 3 = adversary 2 rel pos
      assert agent in self.agents
      agent_type = agent[0]

      blue_team_agents = copy.deepcopy(self.blue_team_agents)
      red_team_agents = copy.deepcopy(self.red_team_agents)

      if agent_type == "B":
        blue_team_agents.remove(agent)
        ally_agent = blue_team_agents[0] #Hard-coded for 2v2 scenario.
        adv_agents = red_team_agents
      if agent_type == "R":
        red_team_agents.remove(agent)
        ally_agent = red_team_agents[0] #Hard-coded for 2v2 scenario.
        adv_agents = blue_team_agents

      obs = np.zeros(16, dtype=np.int8)
      #obs.append(self.state[agent])
      obs[:3] = self.state[agent]
      if agent[0] == "R":
        obs[3:5] = np.array(self.red_flag_location) - np.array(self.state[agent][:2])
        obs[5:7] = np.array(self.blue_flag_location) - np.array(self.state[agent][:2])
      elif agent[0] == "B":
        obs[3:5] = np.array(self.blue_flag_location) - np.array(self.state[agent][:2])
        obs[5:7] = np.array(self.red_flag_location) - np.array(self.state[agent][:2])

      obs[7:9] = np.array(self.state[ally_agent][:2]) - np.array(self.state[agent][:2]) # Friend rel xy position
      obs[9] = (self.state[ally_agent][2] - self.state[agent][2]) % (self.num_headings) # Friend rel heading

      obs[10:12] = np.array(self.state[adv_agents[0]][:2]) - np.array(self.state[agent][:2]) # Adversary 1 rel xy position
      obs[12] = (self.state[adv_agents[0]][2] - self.state[agent][2]) % (self.num_headings) # Adversary 1 rel heading

      obs[13:15] = np.array(self.state[adv_agents[1]][:2]) - np.array(self.state[agent][:2]) # Adversary 2 rel xy position
      obs[15] = (self.state[adv_agents[1]][2] - self.state[agent][2]) % (self.num_headings) # Adversary 2 rel heading

      ## Some simple reward shaping for some behaviour learned for 2v2 CTF policies. Then later optimize the reward shaping parameters via Bayesian optimization.
      return obs

    def _observations(self):
        # Implements the action_masks from the global state.
        obs = {}
        global_state = [self.state[agent] for agent in self.agents]
        for agent in self.agents:
            ##agent_obs = global_state # Had to comment due to some API glue-ing error that Google Colab pointed out.
            #agent_obs = np.concatenate([s for s in global_state]).astype(np.int8)

            agent_obs = self.global_state_to_agent_local_obs(agent=agent) # v2 observations for better policy sharing.

            valid = self.valid_actions(global_state_dict=self.state, agent=agent)
            action_mask = np.zeros(self.num_actions, dtype=np.int8)
            action_mask[valid] = 1
            obs[agent] = { "observation": agent_obs, "action_mask": action_mask }
        return obs

    def dynamics(self, agent, actions):
        # Single agent dynamics.
        ### Action keys:
        ## 0 = same pos, same heading
        ## 1 = same pos, minus heading
        ## 2 = same pos, plus heading
        ## 3 = advance pos, same heading
        ## 4 = tagging action

        assert type(agent) == str
        agent_team = None
        if agent[0] == "R": agent_team = "Red"
        elif agent[0] == "B": agent_team = "Blue"

        tag_bool, reach_goal_bool = False, False

        x, y, heading = self.state[agent]
        action = actions[agent]
        x_, y_, heading_ = None, None, None

        # Check if there are incoming tagging actions from nearby agents.
        potential_incoming_tagging_agents = []
        incoming_tagging_agents = []
        for neighbouring_agent in self.agents:
            if neighbouring_agent == agent: continue
            else:
                x_neighbour, y_neighbour, heading_neighbour = self.state[neighbouring_agent]
                if abs(x - x_neighbour) <= 1 and abs(y - y_neighbour) <= 1:
                    neighbour_point_to_x, neighbour_point_to_y =  self.state[neighbouring_agent][:2] + self._heading_to_direction_vector(heading=heading_neighbour)
                    if (neighbour_point_to_x, neighbour_point_to_y) == (x, y): potential_incoming_tagging_agents.append(neighbouring_agent)
                    else: continue

        for agent_iter in potential_incoming_tagging_agents:
            if actions[agent_iter] == 4: incoming_tagging_agents.append(agent_iter)
            else: continue

        if len(incoming_tagging_agents) == 0:
            # No incoming tags.
            if action == 0:
                # same pos, same heading
                x_, y_, heading_ = x, y, heading
                pass
            elif action == 1:
                # same pos, minus heading
                x_, y_, heading_ = x, y, (heading - 1) % self.num_headings
                pass
            elif action == 2:
                # same pos, plus heading
                x_, y_, heading_ = x, y, (heading + 1) % self.num_headings
                pass
            elif action == 3:
                # advance pos, same heading
                x_, y_, heading_ = x, y, heading
                if heading == 0:
                    x_, y_ = x + 1, y
                elif heading == 1:
                    x_, y_ = x + 1, y + 1
                elif heading == 2:
                    x_, y_ = x, y + 1
                elif heading == 3:
                    x_, y_ = x - 1, y + 1
                elif heading == 4:
                    x_, y_ = x - 1, y
                elif heading == 5:
                    x_, y_ = x - 1, y - 1
                elif heading == 6:
                    x_, y_ = x, y - 1
                elif heading == 7:
                    x_, y_ = x + 1, y - 1
            elif action == 4: # Tagging action.
                x_, y_, heading_ = x, y, heading

            flag_location = None
            #### HAD A HUGE BUG HERE. Now fixed on 04/29 2.45 pm. Interchanged flag locations defining the goal_reach_variable.
            if agent_team == "Red": flag_location = self.blue_flag_location
            elif agent_team == "Blue": flag_location = self.red_flag_location

            if (x_, y_) == flag_location:
                reach_goal_bool = True

        else:
            # Incoming tags. Spawn in the tagging area.
            # Multiple tags = single penalty.
            if self.verbose: print("TAAAAAAAAAAGGGGGGGGGGGGGGGG..............")
            tag_bool = True
            spawn_area = None
            """
            Spawning area is a rectangle, away from the flag in a corner.
            blue_tag_spawn_area = {"x_lim":(5, 7), "y_lim": (0, 1)} # (x, y) coordinate tuple -- x_low y_low and x_high y_high
            red_tag_spawn_area = {"x_lim":(0, 2), "y_lim": (6, 7)} # (x, y) coordinate tuple -- x_low y_low and x_high y_high
            """
            if agent_team == "Red":
                spawn_area = self.red_tag_spawn_area
                possible_spawn_headings = [0, 7, 6, 5] # [0, 7, 6, 5], possible init headings = (0, -45, -90, -135)
            elif agent_team == "Blue":
                spawn_area = self.blue_tag_spawn_area
                possible_spawn_headings = [0, 1, 2, 3] # possible init headings = (0, 45, 90, 135)
            x_prime = self.np_random.integers( spawn_area["x_lim"][0], spawn_area["x_lim"][1] + 1)
            y_prime = self.np_random.integers( spawn_area["y_lim"][0], spawn_area["y_lim"][1] + 1)

            if x_prime == 0: # Implied that agent is red_team.
                try: possible_spawn_headings.remove(5)
                except: pass
            if x_prime == self.grid_size - 1: # Implied that agent is blue_team.
                try: possible_spawn_headings.remove(0)
                except: pass
                try: possible_spawn_headings.remove(1)
                except: pass

            heading_prime = self.np_random.choice(possible_spawn_headings)
            x_, y_, heading_ = x_prime, y_prime, heading_prime

        new_state = (x_, y_, heading_)
        return tag_bool, reach_goal_bool, new_state

    def _heading_to_direction_vector(self, heading):
        assert heading in range(self.num_headings)
        vec = None
        if heading == 0:
            vec = np.array([ +1, 0 ])
        elif heading == 1:
            vec = np.array([ +1, +1 ])
        elif heading == 2:
            vec = np.array([ 0, +1 ])
        elif heading == 3:
            vec = np.array([ -1, +1 ])
        elif heading == 4:
            vec = np.array([ -1, 0 ])
        elif heading == 5:
            vec = np.array([ -1, -1 ])
        elif heading == 6:
            vec = np.array([ 0, -1 ])
        elif heading == 7:
            vec = np.array([ +1, -1 ])
        return vec

    def valid_actions(self, global_state_dict, agent):
        """
        ### Action keys:
        ## 0 = same pos, same heading
        ## 1 = same pos, minus heading
        ## 2 = same pos, plus heading
        ## 3 = advance pos, same heading
        ## 4 = tagging action
        """

        # NOTE: No collision logic in v0.
        # Takes in as input global_state of the world and the agent name and returns valid actions for the agent in the global_state of the world.
        # Utility: to decide action_masking for the agent in the global_state.

        agent_state = copy.deepcopy(global_state_dict[agent])
        # Action masking due to walls of the environment.
        x, y, heading = agent_state
        point_to_x, point_to_y =  agent_state[:2] + self._heading_to_direction_vector(heading=heading)

        valid_actions = [0, 1, 2, 3]
        invalid_action_x = None
        invalid_action_y = None

        if x == 0 and heading == 2:
            invalid_action_x = 2
        if x == 0 and heading == 6:
            invalid_action_x = 1
        if x == 0 and heading in [3, 4, 5]:
          invalid_action_x = 3
        if x == self.grid_size - 1 and heading == 2:
            invalid_action_x = 1
        if x == self.grid_size - 1 and heading == 6:
            invalid_action_x = 2
        if x == self.grid_size - 1 and heading in [0, 1, 7]:
          invalid_action_x = 3
        try:
            valid_actions.remove(invalid_action_x)
        except: pass

        if y == 0 and heading == 0:
            invalid_action_y = 1
        if y == 0 and heading == 4:
            invalid_action_y = 2
        if y == 0 and heading in [5, 6, 7]:
            invalid_action_y = 3
        if y == self.grid_size - 1 and heading == 0:
            invalid_action_y = 2
        if y == self.grid_size - 1 and heading == 4:
            invalid_action_y = 1
        if y == self.grid_size - 1 and heading in [1, 2, 3]:
            invalid_action_y = 3
        try:
            valid_actions.remove(invalid_action_y)
        except: pass

        # Asking masking on the Tagging space due to nearby agents.
        # Logic to see nearby agents position-wise: once a list of nearby agents is generated: check if heading is aligned with the neighbouring agent for potential tagging action.
        target_tagging_agents = []
        for potential_nearby_agent in self.agents:
            if potential_nearby_agent[0] == agent[0]: continue # Cannot tag same team member. HAD A BUG HERE previously. Corrected now.
            else:
                x_neighbour, y_neighbour, heading_neighbour = global_state_dict[potential_nearby_agent]
                if abs(x - x_neighbour) <= 1 and abs(y - y_neighbour) <= 1:
                    if (x_neighbour, y_neighbour) == (point_to_x, point_to_y): target_tagging_agents.append(potential_nearby_agent)
                    else: continue
                pass
            pass
        if len(target_tagging_agents) > 0: valid_actions.append(4) #Tagging action available.

        return valid_actions

    def state_space(self, agent):
        # The state of the agent on the grid: (x, y, theta).
        return MultiDiscrete([self.grid_size, self.grid_size, self.num_headings])

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Each agent can observe the state of all the agents.
        # obs_type = MultiDiscrete([self.grid_size, self.grid_size, self.num_headings, self.grid_size, self.grid_size, self.num_headings, self.grid_size, self.grid_size, self.num_headings, self.grid_size, self.grid_size, self.num_headings]) #Tuple([ self.state_space(agent) for agent in self.agents ])
        # coordinate of agent, heading, home_flag_rel_location, away_flag_rel_location, friend_rel_location, friend_rel_pose, adversary_1_rel_location, adversary_1_rel_pose, adversary_2_rel_location, adversary_2_rel_pose
        # 3 = self coordinate, 2 = home flag location, 2 = away flag location, 3 = friend rel pos, 3 = adversary 1 rel pos, 3 = adversary 2 rel pos
        obs_type = Box(low=0, high=self.grid_size, shape=(2 + 2 + len(self.possible_agents) * 3,), dtype=np.int8)
        #obs_type = Box(low=0, high=self.grid_size, shape=(len(self.possible_agents) * 3,), dtype=np.int8)
        obs_space_type = Dict({"observation": obs_type, "action_mask": Box(low=0, high=1, shape=(self.num_actions,), dtype=np.int8)})
        return obs_space_type

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        #Simple dynamics. #same heading same place, heading plus, heading minus, move forward. And a tagging action.
        return Discrete(self.num_actions)

    def num_agents(self):
        return self.num_teams * self.num_agents_blue_team

    def render(self):
        # Note: Once you have the state and the flag locations, this module can be written disjoint of any other logic in the whole program.
        blue_agent_pos = [self.state[agent] for agent in self.blue_team_agents]
        red_agent_pos = [self.state[agent] for agent in self.red_team_agents]
        fig, ax = draw_grid(grid_size=self.grid_size, blue_agent_pos=blue_agent_pos, red_agent_pos=red_agent_pos, blue_flag_loc=self.blue_flag_location, red_flag_loc=self.red_flag_location, blue_flag_img_path=self.blue_flag_img_path, red_flag_img_path=self.red_flag_img_path)
        plt.show()
        return fig, ax

class CustomCTF_v1(ParallelEnv): # v1: HETEROGENOUS OBSERVATION SPACES FOR RED AND BLUE TEAMS: Partial information case.
    # How to have an environment with other agents behaviour fixed in the PSRO loop? How do the PSRO folks do it? Do they create a new environment instance that inherits from the original environment and pass the other agents behaviour?
    """
    Blue agents policy should be independent of the red flag location. It should just be agents.
    Pending: Debug the partial information case.
    Pending: Blue agent policy should not be a function of the red_flag_location. Blue agent does not know where the red_flag_is: POSMG. It's policy is just a function of home flag location and all the agent locations, not the opponent team flag location.
    !!!!! PENDING: Change the observation space of Blue Team agents for the deceptive CTF environment, or the Partially Observable CTF environment.
    #### Pending: Look at methods for partially observable Games == what changes for PSRO.
    #### Pending: Blue team not knowing Red Flag location at all and learning a behaviour just from interactions versus it having a belief over red_flag_locations.
    """
    meta_data = { "name": "custom_ctf_v0"}
    def __init__(self, grid_size=8, ctf_player_config="2v2", max_num_cycles=300, red_flag_locs=(6,6), obs_mode='hetero', verbose=False, seed=None, **kwargs):
        # Two swarms: red and blue.
        # Nash DQN as initial guess. Then warm start policy synthesis.
        # CTF on a grid of size [grid_size \times grid_size]
        self.metadata = { "name": "custom_ctf_v0"}
        self.render_mode = "human"
        self.verbose = verbose

        self.grid_size = grid_size #this defines the state space of the Markov Game and hence the observation_space.
        self.num_headings = 8  # Possible headings in degrees:(0, 45, 90, 135, 180, 225, 270, 315)
        self.max_num_cycles = max_num_cycles

        if seed is not None:
            self.seed(seed=seed)
        else:
            self._seed = None # dummy value.
            self.np_random = None
            self.torch_rng = None

        self.num_teams = 2
        if ctf_player_config == "1v1":
            self.num_agents_blue_team = 1
        elif ctf_player_config == "2v2":
            self.num_agents_blue_team = 2
        self.num_agents_red_team = self.num_agents_blue_team
        self.num_agents = self.num_agents()

        self.blue_team_agents = ["Blue_" + "{}".format(i) for i in range(self.num_agents_blue_team)]
        self.red_team_agents = ["Red_" + "{}".format(i) for i in range(self.num_agents_red_team)]

        #self.agents = ["Blue_" + "{}".format(i) for i in range(self.num_agents_blue_team)] + ["Red_" + "{}".format(i) for i in range(self.num_agents_red_team)]
        self.agents = copy.deepcopy(self.blue_team_agents + self.red_team_agents)
        if self.verbose: print("self.agents: {}".format(self.agents))
        self.possible_agents = copy.deepcopy(self.agents)
        if self.verbose: print(self.possible_agents)
        
        """ Set observation and action spaces. """
        self.action_space_mode = kwargs.get('action_space_mode', 'default')
        assert self.action_space_mode in ['default', 'smooth4', 'smooth8']
        if self.action_space_mode == 'default': self.num_actions = 5
        elif self.action_space_mode == 'smooth4': self.num_actions = 5
        elif self.action_space_mode == 'smooth8': self.num_actions = 7
        self.obs_mode = obs_mode
        assert self.obs_mode in [ 'hetero', 'homo_blue', 'homo_red' ]

        self.observation_spaces = {agent: self.observation_space(agent=agent)["observation"] for agent in self.agents}
        self.action_spaces = {agent: self.action_space(agent=agent) for agent in self.agents}

        """
        For v1 of the CustomCTF environment: Can add logic to have a jitter / randomness in the flag locations.
        """
        self.deceptive_env = False #Note: Deception refers to uncertainty in the red_flag_location.
        self.blue_flag_location = kwargs.get('blue_flag_location', (1, 1))
        if red_flag_locs is None: self.red_flag_location = (self.grid_size - 2, self.grid_size - 2)
        elif (red_flag_locs is not None) and isinstance(red_flag_locs, tuple):
            self.red_flag_location = red_flag_locs
        elif isinstance(red_flag_locs, dict):
            assert list(red_flag_locs.keys()) == ['locs', 'p']
            assert isinstance(red_flag_locs['locs'], list) and isinstance(red_flag_locs['locs'][0], tuple) and isinstance(red_flag_locs['p'], list)
            assert len(red_flag_locs['locs']) == len(red_flag_locs['p'])
            assert abs(sum(red_flag_locs['p']) - 1) <= 1e-10
            """
            For instance, red_flag_locs = { 'locs': [(6,6), (3,7), (5,5)], 'p': [0.7, 0.2, 0.1] }
            """
            self.deceptive_env = True
            self.red_flag_locs = red_flag_locs
            self.red_flag_location = None

        self.blue_init_spawn_y_lim = 2 # Measured from bottom of the grid.
        self.possible_init_headings_blue_team = [0, 1, 2, 3] # possible init headings = (0, 45, 90, 135)
        self.red_init_spawn_y_lim = 2 # Measured from top of the grid.
        self.possible_init_headings_red_team = [0, 7, 6, 5] # possible init headings = (0, -45, -90, -135)

        # Tag spawning area is a rectangle, away from the flag in a corner.
        self.blue_tag_spawn_area = {"x_lim":(5, 7), "y_lim": (0, 1)} # (x, y) coordinate tuple -- x_low y_low and x_high y_high
        self.red_tag_spawn_area = {"x_lim":(0, 2), "y_lim": (6, 7)} # (x, y) coordinate tuple -- x_low y_low and x_high y_high

        self.current_step = 0
        self.state = {agent: None for agent in self.agents} #self.state = np.array([None for _ in self.agents])
        self.state_hist = {agent: [] for agent in self.agents}
        self.state_interpolate_hist = {agent: None for agent in self.agents}

        self.F = 100 #self.capture_flag_reward = 100
        self.tag_reward_shape = True
        self.T = 10 #10 #self.tagging_penalty = 10

        self.blue_flag_img_path = 'flag_imgs/blue_flag.png'
        self.red_flag_img_path = 'flag_imgs/red_flag.png'

        self.blue_flag_img = mpimg.imread('flag_imgs/blue_flag.png')
        self.red_flag_img = mpimg.imread('flag_imgs/red_flag.png')

        self.flag_capture_team = None

        self._init_args = ()
        self._init_kwargs = {"grid_size":grid_size, "ctf_player_config":ctf_player_config, "max_num_cycles":max_num_cycles, "red_flag_locs":red_flag_locs, "obs_mode":obs_mode, "verbose":verbose, "seed":seed}

    def __call__(self):
        return CustomCTF_v1(*self._init_args, **self._init_kwargs)

    def seed(self, seed=None):
        self.rngs = make_seeded_rngs(seed)
        self._seed = self.rngs["actual_seed"]
        self.np_random = self.rngs["np_random"]
        self.torch_rng = self.rngs["torch_rng"]
        return [self._seed]

    def set_obs_mode(self, obs_mode):
        assert obs_mode in [ 'hetero', 'homo_blue', 'homo_red' ]
        self.obs_mode = obs_mode
        return

    def agent_idx(self, agent_name):
        # Returns the agent_idx according to the self.agents list for the input agent_name.
        return

    def state(self):
        # Returns the global state as a Dict or an array?
        # I think the answer depends on the type of Agent names, if those are int then array otherwise Dict.
        return self.state

    def _sample_init_heading(self, agent_team, x, y):
        assert agent_team == "Blue" or agent_team == "Red"
        if agent_team == "Blue":
            possible_init_headings = copy.deepcopy(self.possible_init_headings_blue_team)
            if x == 0: possible_init_headings.remove(3)
            if x == self.grid_size - 1:
                possible_init_headings.remove(0)
                possible_init_headings.remove(1)
            heading = self.np_random.choice(possible_init_headings)
        elif agent_team == "Red":
            possible_init_headings = copy.deepcopy(self.possible_init_headings_red_team)
            if x == 0: possible_init_headings.remove(5)
            if x == 1:
                possible_init_headings.remove(0)
                possible_init_headings.remove(7)
            heading = self.np_random.choice(possible_init_headings)
        return heading

    def reset(self, seed = None, options = None):
      #### Debugging prints: print valid actions for each agent at the step. => a) actions available at the walls b) tagging action available or not.
        if seed is not None:
            self.seed(seed=seed)

        self.current_step = 0
        self.agents = copy.deepcopy(self.blue_team_agents + self.red_team_agents) # !!! 04/25 3.56 pm: DON'T UNDERSTAND THIS BUG.
        """
        NOTE: In the current implementation, v0, flag locations are parameters of the Markov game and are hence not a part of the state, agent-wise or global.
        For v1: Can add logic to have a jitter / randomness in the flag locations.
        """
        #self.blue_flag_location = (1, 1)
        #self.red_flag_location = (self.grid_size - 2, self.grid_size - 2) # We assume the grid_size > 3 or something like that for the flag placements to be a non-trivial game scenario.

        self.blue_flag_location = self.blue_flag_location
        if self.deceptive_env:
            red_flag_idx = self.np_random.choice(len(self.red_flag_locs['locs']), p=self.red_flag_locs['p'])
            self.red_flag_location = self.red_flag_locs['locs'][red_flag_idx]
            print("RED FLAG LOCATION: {}".format(self.red_flag_location))

        blue_team_init_xys = []
        for agent_iter, blue_agent_num in enumerate(range(self.num_agents_blue_team)):
            if agent_iter == 0:
                rand_x = self.np_random.integers(3, 7) #self.np_random.integers(0, self.grid_size)
                rand_y = self.np_random.integers(0, self.blue_init_spawn_y_lim + 1)
                rand_theta = self._sample_init_heading(agent_team="Blue", x=rand_x, y=rand_y)
                self.state["Blue_{}".format(blue_agent_num)] = np.array([rand_x, rand_y, rand_theta])
                blue_team_init_xys.append((rand_x, rand_y))
                continue
            # A while loop to break ties in the x-y location to prevent collision during agent initialization.
            while (rand_x, rand_y) in blue_team_init_xys:
                rand_x = self.np_random.integers(3, 7) #self.np_random.integers(0, self.grid_size)
                rand_y = self.np_random.integers(0, self.blue_init_spawn_y_lim + 1)
            blue_team_init_xys.append((rand_x, rand_y))
            rand_theta = self._sample_init_heading(agent_team="Blue", x=rand_x, y=rand_y)
            self.state["Blue_{}".format(blue_agent_num)] = np.array([rand_x, rand_y, rand_theta])

        red_team_init_xys = []
        for agent_iter, red_agent_num in enumerate(range(self.num_agents_red_team)):
            if agent_iter == 0:
                rand_x = self.np_random.integers(3, 7) #self.np_random.integers(0, self.grid_size)
                rand_y = self.np_random.integers(self.grid_size - self.red_init_spawn_y_lim - 1, self.grid_size)
                rand_theta = self._sample_init_heading(agent_team="Red", x=rand_x, y=rand_y)
                self.state["Red_{}".format(red_agent_num)] = np.array([rand_x, rand_y, rand_theta])
                red_team_init_xys.append((rand_x, rand_y))
                continue
            # A while loop to break ties in the x-y location to prevent collision during agent initialization.
            while (rand_x, rand_y) in red_team_init_xys:
                rand_x = self.np_random.integers(3, 7) #self.np_random.integers(0, self.grid_size)
                rand_y = self.np_random.integers(self.grid_size - self.red_init_spawn_y_lim - 1, self.grid_size)
            red_team_init_xys.append((rand_x, rand_y))
            rand_theta = self._sample_init_heading(agent_team="Red", x=rand_x, y=rand_y)
            self.state["Red_{}".format(red_agent_num)] = np.array([rand_x, rand_y, rand_theta])

        #global_state = [self.state[agent] for agent in self.agents]
        #obs = {agent: global_state for agent in self.agents} # global_state dict. The prisoner guard example on ParallelEnv has this as a tuple, and you have this as a List currently.
        self.state_hist = {agent: [self.state[agent]] for agent in self.agents}
        obs = self._observations()
        #info = {}
        info = {agent: {} for agent in self.agents} # September 1 2025: info = self.state?
        info['game_state_dict'] = self.state
        #print("self.agents: {}".format(self.agents))
        return obs, info

    def step(self, actions):
      #### Debugging prints: print valid actions for each agent at the step. => a) actions available at the walls b) tagging action available or not.
        ### v0: No collision logic.
        # TO-DO for v1: Resolve collisions and tagging and re-spawning. # Collisions within a team prevented via action_masking. Where to write actions_available_to_an_agent?

        # The input actions is a Dict from agent_name to actions. Add an assert statement for Type of actions.
        """
        Action keys:
        0 = same pos, same heading
        1 = same pos, minus heading
        2 = same pos, plus heading
        3 = advance pos, same heading
        4 = tagging action
        """
        if self.verbose: print("Current Step: {}".format(self.current_step))
        if self.verbose: print("Before stepping self.agents state: {}".format(self.state))
        if self.verbose: print("Actions: {}".format(actions))

        infos = {agent: {} for agent in self.agents}
        for agent in self.agents:
            infos[agent]["CTF"] = None # "CTF" descriptor is for the last step of the game on which team captured the flag if any!
        tag_bool = {agent: False for agent in self.agents}
        reach_goal_bool = {agent: False for agent in self.agents}
        new_state = {agent: None for agent in self.agents}

        rewards = {agent: 0. for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        for agent in self.agents:
          assert actions[agent] in self.valid_actions(self.state, agent)

        # Actions dict = actions.
        for agent in self.agents:
            assert actions[agent] in self.action_space(agent=agent)
            tag_bool[agent], reach_goal_bool[agent], new_state[agent] = self.dynamics(agent=agent, actions=actions)
            #print("agent: {}, tag_bool_agent: {}".format(agent, tag_bool[agent]))

        for agent in self.agents:
            self.state[agent] = new_state[agent]

        blue_tag = np.sum([ tag_bool[agent] for agent in self.blue_team_agents ])
        red_tag = np.sum([ tag_bool[agent] for agent in self.red_team_agents ])
        
        if self.verbose:
            for agent in self.blue_team_agents: print("agent: {}, blue tag check : {}".format(agent, tag_bool[agent]))
            for agent in self.red_team_agents: print("agent: {}, red tag check : {}".format(agent, tag_bool[agent]))
            for agent in self.blue_team_agents: print("agent: {}, blue reach_goal check : {}".format(agent, reach_goal_bool[agent]))
            for agent in self.red_team_agents: print("agent: {}, red reach_goal check : {}".format(agent, reach_goal_bool[agent]))
        
        blue_reach_goal = np.sum([ reach_goal_bool[agent] for agent in self.blue_team_agents ])
        red_reach_goal = np.sum([ reach_goal_bool[agent] for agent in self.red_team_agents ])

        if blue_reach_goal + red_reach_goal == 0:
            # Flag not captured.
            terminate = False
            terminations = {agent: terminate for agent in self.agents}
            # Assign tagging rewards if any.
            for agent in self.blue_team_agents:
                rewards[agent] +=  - blue_tag * (self.T)
                rewards[agent] += + red_tag * (self.T)
            for agent in self.red_team_agents:
                rewards[agent] += + blue_tag * (self.T)
                rewards[agent] += - red_tag * (self.T)
        else:
            # Flag captured.
            terminate = True
            terminations = {agent: terminate for agent in self.agents}
            for agent in self.blue_team_agents:
                if blue_reach_goal:
                    rewards[agent] += self.F
                    infos[agent]["CTF"] = True
                    self.flag_capture_team = "Blue"
                elif red_reach_goal:
                    rewards[agent] += - self.F
                    infos[agent]["CTF"] = False
                    self.flag_capture_team = "Red"
            for agent in self.red_team_agents:
                if blue_reach_goal:
                    rewards[agent] += - self.F
                    infos[agent]["CTF"] = False
                elif red_reach_goal:
                    rewards[agent] += self.F
                    infos[agent]["CTF"] = True
                pass
        infos['game_state_dict'] = self.state

        # TO-DO: Determine rewards. Reward structure: team reward: agents share the same reward within a team.
        obs = self._observations()
        for agent in self.agents:
            if tag_bool[agent]:
                # self.state_hist[agent] stores agent_pos_history since last tagging.
                self.state_hist[agent] = [self.state[agent]]
                continue
            self.state_hist[agent].append(self.state[agent])

        self.current_step += 1
        if self.current_step == self.max_num_cycles: truncations = {agent: True for agent in self.agents}

        agents_to_remove = []
        for agent in self.agents:
            if terminations[agent] or truncations[agent]:
                agents_to_remove.append(agent)

        # Remove done agents
        for agent in agents_to_remove:
            self.agents.remove(agent)
        if self.verbose:
            print("Updated self.agents state: {}".format(self.state))
            print("rewards: {}".format(rewards))
            print("terminations: {}".format(terminations))
            print("truncations: {}".format(truncations))
        return obs, rewards, terminations, truncations, infos

    def global_state_to_agent_local_obs(self, agent):
        """
        Hard-coded for a 2v2 scenario.
        For more agents == GNN.
        ### For the second iteration of the PSRO loop: CTDE? Fixing one team behaviour, have a CTDE scheme for the other team.
        """
        """
        Idea: Permutation invariance for better learning: ordering of adversaries shouldn't matter. Same strategy.
        """
        # coordinate of agent, heading, home_flag_rel_location, away_flag_rel_location, friend_rel_location, friend_rel_pose, adversary_1_rel_location, adversary_1_rel_pose, adversary_2_rel_location, adversary_2_rel_pose
        # 3 = self coordinate, 2 = home flag location, 2 = away flag location, 3 = friend rel pos, 3 = adversary 1 rel pos, 3 = adversary 2 rel pos
        assert agent in self.agents
        agent_type = agent[0]

        blue_team_agents = copy.deepcopy(self.blue_team_agents)
        red_team_agents = copy.deepcopy(self.red_team_agents)

        if agent_type == "B":
            blue_team_agents.remove(agent)
            ally_agent = blue_team_agents[0] #Hard-coded for 2v2 scenario.
            adv_agents = red_team_agents
        if agent_type == "R":
            red_team_agents.remove(agent)
            ally_agent = red_team_agents[0] #Hard-coded for 2v2 scenario.
            adv_agents = blue_team_agents

        obs = np.zeros(16, dtype=np.int8)
        #obs.append(self.state[agent])
        obs[:3] = self.state[agent]
        if agent[0] == "R":
            obs[3:5] = np.array(self.red_flag_location) - np.array(self.state[agent][:2]) # HOME FLAG.
            obs[5:7] = np.array(self.blue_flag_location) - np.array(self.state[agent][:2]) # AWAY FLAG.
        elif agent[0] == "B":
            obs[3:5] = np.array(self.blue_flag_location) - np.array(self.state[agent][:2]) # HOME FLAG.
            obs[5:7] = np.array(self.red_flag_location) - np.array(self.state[agent][:2]) # AWAY FLAG.

        obs[7:9] = np.array(self.state[ally_agent][:2]) - np.array(self.state[agent][:2]) # Friend rel xy position
        obs[9] = (self.state[ally_agent][2] - self.state[agent][2]) % (self.num_headings) # Friend rel heading

        obs[10:12] = np.array(self.state[adv_agents[0]][:2]) - np.array(self.state[agent][:2]) # Adversary 1 rel xy position
        obs[12] = (self.state[adv_agents[0]][2] - self.state[agent][2]) % (self.num_headings) # Adversary 1 rel heading

        obs[13:15] = np.array(self.state[adv_agents[1]][:2]) - np.array(self.state[agent][:2]) # Adversary 2 rel xy position
        obs[15] = (self.state[adv_agents[1]][2] - self.state[agent][2]) % (self.num_headings) # Adversary 2 rel heading

        ## Some simple reward shaping for some behaviour learned for 2v2 CTF policies. Then later optimize the reward shaping parameters via Bayesian optimization.
        
        obs_mask = np.ones(16, dtype=np.bool)
        if self.obs_mode == 'hetero':
            if agent[0] == 'R': pass
            elif agent[0] == 'B': obs_mask[5:7] = np.array([False, False])
        elif self.obs_mode == 'homo_blue':
            obs_mask[5:7] = np.array([False, False])
        elif self.obs_mode == 'homo_red':
            pass
        obs = obs[obs_mask]
        return obs

    def _observations(self):
        """
        PENDING: 06/01/2025: Latest change pending to be made.
        """
        # Implements the action_masks from the global state.
        obs = {}
        global_state = [self.state[agent] for agent in self.agents]
        for agent in self.agents:
            ##agent_obs = global_state # Had to comment due to some API glue-ing error that Google Colab pointed out.
            #agent_obs = np.concatenate([s for s in global_state]).astype(np.int8)

            agent_obs = self.global_state_to_agent_local_obs(agent=agent) # v2 observations for better policy sharing.

            valid = self.valid_actions(global_state_dict=self.state, agent=agent)
            action_mask = np.zeros(self.num_actions, dtype=np.int8)
            action_mask[valid] = 1
            obs[agent] = { "observation": agent_obs, "action_mask": action_mask }
        return obs

    def dynamics(self, agent, actions):
        # Single agent dynamics.
        """
        ### Action keys:
        ## 0 = same pos, same heading
        ## 1 = same pos, minus heading
        ## 2 = same pos, plus heading
        ## 3 = advance pos, same heading
        ## 4 = tagging action
        ## 5 = minus two headings (if available)
        ## 6 = plus two headings (if available)
        """

        assert type(agent) == str
        agent_team = None
        if agent[0] == "R": agent_team = "Red"
        elif agent[0] == "B": agent_team = "Blue"

        tag_bool, reach_goal_bool = False, False

        x, y, heading = self.state[agent]
        action = actions[agent]
        x_, y_, heading_ = None, None, None

        # Check if there are incoming tagging actions from nearby agents.
        potential_incoming_tagging_agents = []
        incoming_tagging_agents = []
        for neighbouring_agent in self.agents:
            if neighbouring_agent == agent: continue
            else:
                x_neighbour, y_neighbour, heading_neighbour = self.state[neighbouring_agent]
                if abs(x - x_neighbour) <= 1 and abs(y - y_neighbour) <= 1:
                    neighbour_point_to_x, neighbour_point_to_y =  self.state[neighbouring_agent][:2] + self._heading_to_direction_vector(heading=heading_neighbour)
                    if (neighbour_point_to_x, neighbour_point_to_y) == (x, y): potential_incoming_tagging_agents.append(neighbouring_agent)
                    else: continue

        for agent_iter in potential_incoming_tagging_agents:
            if actions[agent_iter] == 4: incoming_tagging_agents.append(agent_iter)
            else: continue

        x_point, y_point = self.state[agent][:2] + self._heading_to_direction_vector(heading=heading)

        if len(incoming_tagging_agents) == 0:
            # No incoming tags.
            if action == 0:
                # same pos, same heading
                x_, y_, heading_ = x, y, heading
            elif action == 1:
                # same pos, minus heading
                if self.action_space_mode == 'default': x_, y_, heading_ = x, y, (heading - 1) % self.num_headings
                else:
                    heading_ = (heading - 1) % self.num_headings
                    x_, y_ = self.state[agent][:2] + self._heading_to_direction_vector(heading=heading_)
            elif action == 2:
                # same pos, plus heading
                if self.action_space_mode == 'default': x_, y_, heading_ = x, y, (heading + 1) % self.num_headings
                else:
                    heading_ = (heading + 1) % self.num_headings
                    x_, y_ = self.state[agent][:2] + self._heading_to_direction_vector(heading=heading_)
            elif action == 3:
                # advance pos, same heading
                x_, y_, heading_ = x, y, heading
                if heading == 0:
                    x_, y_ = x + 1, y
                elif heading == 1:
                    x_, y_ = x + 1, y + 1
                elif heading == 2:
                    x_, y_ = x, y + 1
                elif heading == 3:
                    x_, y_ = x - 1, y + 1
                elif heading == 4:
                    x_, y_ = x - 1, y
                elif heading == 5:
                    x_, y_ = x - 1, y - 1
                elif heading == 6:
                    x_, y_ = x, y - 1
                elif heading == 7:
                    x_, y_ = x + 1, y - 1
            elif action == 4: # Tagging action.
                x_, y_, heading_ = x, y, heading
            elif action == 5:
                heading_ = (heading - 1) % self.num_headings
                x_, y_ = self.state[agent][:2] + self._heading_to_direction_vector(heading=heading_)
                heading_ = (heading_ - 1) % self.num_headings
            elif action == 6:
                heading_ = (heading + 1) % self.num_headings
                x_, y_ = self.state[agent][:2] + self._heading_to_direction_vector(heading=heading_)
                heading_ = (heading_ + 1) % self.num_headings
                
            flag_location = None
            #### HAD A HUGE BUG HERE. Now fixed on 04/29 2.45 pm. Interchanged flag locations defining the goal_reach_variable.
            if agent_team == "Red": flag_location = self.blue_flag_location
            elif agent_team == "Blue": flag_location = self.red_flag_location

            if (x_, y_) == flag_location:
                reach_goal_bool = True

        else:
            # Incoming tags. Spawn in the tagging area.
            # Multiple tags = single penalty.
            if self.verbose: print("TAAAAAAAAAAGGGGGGGGGGGGGGGG..............")
            tag_bool = True
            spawn_area = None
            """
            Spawning area is a rectangle, away from the flag in a corner.
            blue_tag_spawn_area = {"x_lim":(5, 7), "y_lim": (0, 1)} # (x, y) coordinate tuple -- x_low y_low and x_high y_high
            red_tag_spawn_area = {"x_lim":(0, 2), "y_lim": (6, 7)} # (x, y) coordinate tuple -- x_low y_low and x_high y_high
            """
            if agent_team == "Red":
                spawn_area = self.red_tag_spawn_area
                possible_spawn_headings = [0, 7, 6, 5] # [0, 7, 6, 5], possible init headings = (0, -45, -90, -135)
            elif agent_team == "Blue":
                spawn_area = self.blue_tag_spawn_area
                possible_spawn_headings = [0, 1, 2, 3] # possible init headings = (0, 45, 90, 135)
            x_prime = self.np_random.integers(3, 7) #self.np_random.integers( spawn_area["x_lim"][0], spawn_area["x_lim"][1] + 1)
            if agent[0] == "R": y_prime = self.np_random.integers(self.grid_size - self.red_init_spawn_y_lim - 1, self.grid_size) #y_prime = self.np_random.integers( spawn_area["y_lim"][0], spawn_area["y_lim"][1] + 1)
            elif agent[0] == "B": y_prime = self.np_random.integers(0, self.blue_init_spawn_y_lim + 1)
            if self.verbose: print("respawn y_prime for agent {}: {}".format(agent, y_prime))
            if x_prime == 0: # Implied that agent is red_team.
                try: possible_spawn_headings.remove(5)
                except: pass
            if x_prime == self.grid_size - 1: # Implied that agent is blue_team.
                try: possible_spawn_headings.remove(0)
                except: pass
                try: possible_spawn_headings.remove(1)
                except: pass

            heading_prime = self.np_random.choice(possible_spawn_headings)
            x_, y_, heading_ = x_prime, y_prime, heading_prime

        new_state = (x_, y_, heading_)
        return tag_bool, reach_goal_bool, new_state
    
    @staticmethod
    def dynamics_step(state, action, action_space_mode='smooth8', num_headings=8):
        # Single agent dynamics.
        assert action_space_mode in ['default', 'smooth4', 'smooth8']
        """
        ### Action keys:
        ## 0 = same pos, same heading
        ## 1 = same pos, minus heading
        ## 2 = same pos, plus heading
        ## 3 = advance pos, same heading
        ## 4 = tagging action
        ## 5 = minus two headings (if available)
        ## 6 = plus two headings (if available)
        """

        x, y, heading = state
        x_, y_, heading_ = None, None, None

        if action == 0:
            # same pos, same heading
            x_, y_, heading_ = x, y, heading
        elif action == 1:
            # same pos, minus heading
            if action_space_mode == 'default': x_, y_, heading_ = x, y, (heading - 1) % num_headings
            else:
                heading_ = (heading - 1) % num_headings
                x_, y_ = state[:2] + CustomCTF_v1.heading_to_direction_vector_static(heading=heading_)
        elif action == 2:
            # same pos, plus heading
            if action_space_mode == 'default': x_, y_, heading_ = x, y, (heading + 1) % num_headings
            else:
                heading_ = (heading + 1) % num_headings
                x_, y_ = state[:2] + CustomCTF_v1.heading_to_direction_vector_static(heading=heading_)
        elif action == 3:
            # advance pos, same heading
            x_, y_, heading_ = x, y, heading
            if heading == 0:
                x_, y_ = x + 1, y
            elif heading == 1:
                x_, y_ = x + 1, y + 1
            elif heading == 2:
                x_, y_ = x, y + 1
            elif heading == 3:
                x_, y_ = x - 1, y + 1
            elif heading == 4:
                x_, y_ = x - 1, y
            elif heading == 5:
                x_, y_ = x - 1, y - 1
            elif heading == 6:
                x_, y_ = x, y - 1
            elif heading == 7:
                x_, y_ = x + 1, y - 1
        elif action == 4: # Tagging action.
            x_, y_, heading_ = x, y, heading
        elif action == 5:
            heading_ = (heading - 1) % num_headings
            x_, y_ = state[:2] + CustomCTF_v1.heading_to_direction_vector_static(heading=heading_)
            heading_ = (heading_ - 1) % num_headings
        elif action == 6:
            heading_ = (heading + 1) % num_headings
            x_, y_ = state[:2] + CustomCTF_v1.heading_to_direction_vector_static(heading=heading_)
            heading_ = (heading_ + 1) % num_headings

        new_state = (x_, y_, heading_)
        return new_state
    
    @staticmethod
    def heading_to_direction_vector_static(heading):
        #assert heading in range(self.num_headings)
        assert heading in range(8)
        vec = None
        if heading == 0:
            vec = np.array([ +1, 0 ])
        elif heading == 1:
            vec = np.array([ +1, +1 ])
        elif heading == 2:
            vec = np.array([ 0, +1 ])
        elif heading == 3:
            vec = np.array([ -1, +1 ])
        elif heading == 4:
            vec = np.array([ -1, 0 ])
        elif heading == 5:
            vec = np.array([ -1, -1 ])
        elif heading == 6:
            vec = np.array([ 0, -1 ])
        elif heading == 7:
            vec = np.array([ +1, -1 ])
        return vec
    
    def _heading_to_direction_vector(self, heading):
        assert heading in range(self.num_headings)
        vec = None
        if heading == 0:
            vec = np.array([ +1, 0 ])
        elif heading == 1:
            vec = np.array([ +1, +1 ])
        elif heading == 2:
            vec = np.array([ 0, +1 ])
        elif heading == 3:
            vec = np.array([ -1, +1 ])
        elif heading == 4:
            vec = np.array([ -1, 0 ])
        elif heading == 5:
            vec = np.array([ -1, -1 ])
        elif heading == 6:
            vec = np.array([ 0, -1 ])
        elif heading == 7:
            vec = np.array([ +1, -1 ])
        return vec

    @staticmethod
    def valid_actions_static(state, action_space_mode='smooth8', num_actions=7, grid_size=10):
        # Action masking due to walls of the environment.
        assert action_space_mode in ['default', 'smooth4', 'smooth8']
        """
        ### Action keys:
        ## 0 = same pos, same heading
        ## 1 = same pos, minus heading
        ## 2 = same pos, plus heading
        ## 3 = advance pos, same heading
        ## 4 = tagging action
        ## 5 = minus two headings (if available)
        ## 6 = plus two headings (if available)
        """
        x, y, heading = state

        valid_actions = list(range(num_actions)) #[0, 1, 2, 3]
        valid_actions.remove(4)
        invalid_actions_x = []
        invalid_actions_y = []

        if x == 0 and heading == 2:
            if action_space_mode == 'default': invalid_actions_x = [2]
            elif action_space_mode == 'smooth4': invalid_actions_x = [2]
            elif action_space_mode == 'smooth8': invalid_actions_x = [2, 6]
        if x == 0 and heading == 6:
            if action_space_mode == 'default': invalid_actions_x = [1]
            elif action_space_mode == 'smooth4': invalid_actions_x = [1]
            elif action_space_mode == 'smooth8': invalid_actions_x = [1, 5]
        if x == 0 and heading in [3, 4, 5]:
            if action_space_mode == 'default': invalid_actions_x = [3]
            elif action_space_mode == 'smooth4': invalid_actions_x = [1, 2, 3]
            elif action_space_mode == 'smooth8': invalid_actions_x = [1, 2, 3, 5, 6]
        if x == grid_size - 1 and heading == 2:
            if action_space_mode == 'default': invalid_actions_x = [1]
            elif action_space_mode == 'smooth4': invalid_actions_x = [1]
            elif action_space_mode == 'smooth8': invalid_actions_x = [1, 5]
        if x == grid_size - 1 and heading == 6:
            if action_space_mode == 'default': invalid_actions_x = [2]
            elif action_space_mode == 'smooth4': invalid_actions_x = [2]
            elif action_space_mode == 'smooth8': invalid_actions_x = [2, 6]
        if x == grid_size - 1 and heading in [0, 1, 7]:
            if action_space_mode == 'default': invalid_actions_x = [3]
            elif action_space_mode == 'smooth4': invalid_actions_x = [1, 2, 3]
            elif action_space_mode == 'smooth8': invalid_actions_x = [1, 2, 3, 5, 6]
        
        for action in invalid_actions_x:
            try:
                valid_actions.remove(action)
            except: pass

        if y == 0 and heading == 0:
            if action_space_mode == 'default': invalid_actions_y = [1]
            elif action_space_mode == 'smooth4': invalid_actions_y = [1]
            elif action_space_mode == 'smooth8': invalid_actions_y = [1, 5]
        if y == 0 and heading == 4:
            if action_space_mode == 'default': invalid_actions_y = [2]
            elif action_space_mode == 'smooth4': invalid_actions_y = [2]
            elif action_space_mode == 'smooth8': invalid_actions_y = [2, 6]
        if y == 0 and heading in [5, 6, 7]:
            if action_space_mode == 'default': invalid_actions_y = [3]
            elif action_space_mode == 'smooth4': invalid_actions_y = [1, 2, 3]
            elif action_space_mode == 'smooth8': invalid_actions_y = [1, 2, 3, 5, 6]
        if y == grid_size - 1 and heading == 0:
            if action_space_mode == 'default': invalid_actions_y = [2]
            elif action_space_mode == 'smooth4': invalid_actions_y = [2]
            elif action_space_mode == 'smooth8': invalid_actions_y = [2, 6]
        if y == grid_size - 1 and heading == 4:
            if action_space_mode == 'default': invalid_actions_y = [1]
            elif action_space_mode == 'smooth4': invalid_actions_y = [1]
            elif action_space_mode == 'smooth8': invalid_actions_y = [1, 5]
        if y == grid_size - 1 and heading in [1, 2, 3]:
            if action_space_mode == 'default': invalid_actions_y = [3]
            elif action_space_mode == 'smooth4': invalid_actions_y = [1, 2, 3]
            elif action_space_mode == 'smooth8': invalid_actions_y = [1, 2, 3, 5, 6]
        
        for action in invalid_actions_y:
            try:
                valid_actions.remove(action)
            except: pass

        return valid_actions

    def valid_actions(self, global_state_dict, agent):
        """
        ### Action keys:
        ## 0 = same pos, same heading
        ## 1 = same pos, minus heading
        ## 2 = same pos, plus heading
        ## 3 = advance pos, same heading
        ## 4 = tagging action
        ## 5 = minus two headings (if available)
        ## 6 = plus two headings (if available)
        """

        # NOTE: No collision logic in v0.
        # Takes in as input global_state of the world and the agent name and returns valid actions for the agent in the global_state of the world.
        # Utility: to decide action_masking for the agent in the global_state.

        agent_state = copy.deepcopy(global_state_dict[agent])
        # Action masking due to walls of the environment.
        x, y, heading = agent_state
        point_to_x, point_to_y =  agent_state[:2] + self._heading_to_direction_vector(heading=heading)

        valid_actions = list(range(self.num_actions)) #[0, 1, 2, 3]
        valid_actions.remove(4)
        invalid_actions_x = []
        invalid_actions_y = []

        if x == 0 and heading == 2:
            if self.action_space_mode == 'default': invalid_actions_x = [2]
            elif self.action_space_mode == 'smooth4': invalid_actions_x = [2]
            elif self.action_space_mode == 'smooth8': invalid_actions_x = [2, 6]
        if x == 0 and heading == 6:
            if self.action_space_mode == 'default': invalid_actions_x = [1]
            elif self.action_space_mode == 'smooth4': invalid_actions_x = [1]
            elif self.action_space_mode == 'smooth8': invalid_actions_x = [1, 5]
        if x == 0 and heading in [3, 4, 5]:
            if self.action_space_mode == 'default': invalid_actions_x = [3]
            elif self.action_space_mode == 'smooth4': invalid_actions_x = [1, 2, 3]
            elif self.action_space_mode == 'smooth8': invalid_actions_x = [1, 2, 3, 5, 6]
        if x == self.grid_size - 1 and heading == 2:
            if self.action_space_mode == 'default': invalid_actions_x = [1]
            elif self.action_space_mode == 'smooth4': invalid_actions_x = [1]
            elif self.action_space_mode == 'smooth8': invalid_actions_x = [1, 5]
        if x == self.grid_size - 1 and heading == 6:
            if self.action_space_mode == 'default': invalid_actions_x = [2]
            elif self.action_space_mode == 'smooth4': invalid_actions_x = [2]
            elif self.action_space_mode == 'smooth8': invalid_actions_x = [2, 6]
        if x == self.grid_size - 1 and heading in [0, 1, 7]:
            if self.action_space_mode == 'default': invalid_actions_x = [3]
            elif self.action_space_mode == 'smooth4': invalid_actions_x = [1, 2, 3]
            elif self.action_space_mode == 'smooth8': invalid_actions_x = [1, 2, 3, 5, 6]
        
        for action in invalid_actions_x:
            try:
                valid_actions.remove(action)
            except: pass

        if y == 0 and heading == 0:
            if self.action_space_mode == 'default': invalid_actions_y = [1]
            elif self.action_space_mode == 'smooth4': invalid_actions_y = [1]
            elif self.action_space_mode == 'smooth8': invalid_actions_y = [1, 5]
        if y == 0 and heading == 4:
            if self.action_space_mode == 'default': invalid_actions_y = [2]
            elif self.action_space_mode == 'smooth4': invalid_actions_y = [2]
            elif self.action_space_mode == 'smooth8': invalid_actions_y = [2, 6]
        if y == 0 and heading in [5, 6, 7]:
            if self.action_space_mode == 'default': invalid_actions_y = [3]
            elif self.action_space_mode == 'smooth4': invalid_actions_y = [1, 2, 3]
            elif self.action_space_mode == 'smooth8': invalid_actions_y = [1, 2, 3, 5, 6]
        if y == self.grid_size - 1 and heading == 0:
            if self.action_space_mode == 'default': invalid_actions_y = [2]
            elif self.action_space_mode == 'smooth4': invalid_actions_y = [2]
            elif self.action_space_mode == 'smooth8': invalid_actions_y = [2, 6]
        if y == self.grid_size - 1 and heading == 4:
            if self.action_space_mode == 'default': invalid_actions_y = [1]
            elif self.action_space_mode == 'smooth4': invalid_actions_y = [1]
            elif self.action_space_mode == 'smooth8': invalid_actions_y = [1, 5]
        if y == self.grid_size - 1 and heading in [1, 2, 3]:
            if self.action_space_mode == 'default': invalid_actions_y = [3]
            elif self.action_space_mode == 'smooth4': invalid_actions_y = [1, 2, 3]
            elif self.action_space_mode == 'smooth8': invalid_actions_y = [1, 2, 3, 5, 6]
        
        for action in invalid_actions_y:
            try:
                valid_actions.remove(action)
            except: pass

        # Logic to see nearby agents position-wise: once a list of nearby agents is generated: check if heading is aligned with the neighbouring agent for potential tagging action.
        target_tagging_agents = []
        for potential_nearby_agent in self.agents:
            if potential_nearby_agent[0] == agent[0]: continue # Cannot tag same team member. HAD A BUG HERE previously. Corrected now.
            else:
                x_neighbour, y_neighbour, heading_neighbour = global_state_dict[potential_nearby_agent]
                if abs(x - x_neighbour) <= 1 and abs(y - y_neighbour) <= 1:
                    if (x_neighbour, y_neighbour) == (point_to_x, point_to_y): target_tagging_agents.append(potential_nearby_agent)
                    else: continue
                pass
            pass
        if len(target_tagging_agents) > 0: valid_actions.append(4) #Tagging action available.

        return valid_actions

    def state_space(self, agent):
        # The state of the agent on the grid: (x, y, theta).
        return MultiDiscrete([self.grid_size, self.grid_size, self.num_headings])

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Each agent can observe the state of all the agents.
        # obs_type = MultiDiscrete([self.grid_size, self.grid_size, self.num_headings, self.grid_size, self.grid_size, self.num_headings, self.grid_size, self.grid_size, self.num_headings, self.grid_size, self.grid_size, self.num_headings]) #Tuple([ self.state_space(agent) for agent in self.agents ])
        # coordinate of agent, heading, home_flag_rel_location, away_flag_rel_location, friend_rel_location, friend_rel_pose, adversary_1_rel_location, adversary_1_rel_pose, adversary_2_rel_location, adversary_2_rel_pose
        # 3 = self coordinate, 2 = home flag location, 2 = away flag location, 3 = friend rel pos, 3 = adversary 1 rel pos, 3 = adversary 2 rel pos
        
        obs_type_full = Box(low=0, high=self.grid_size, shape=(2 + 2 + len(self.possible_agents) * 3,), dtype=np.int8) # Away flag is visible.
        obs_type_partial = Box(low=0, high=self.grid_size, shape=(2 + 0 + len(self.possible_agents) * 3,), dtype=np.int8) # Away flag not visible.
        if self.obs_mode == 'hetero':
            if agent[0] == 'R': obs_type = obs_type_full
            elif agent[0] == 'B': obs_type = obs_type_partial
        elif self.obs_mode == 'homo_blue':
            obs_type = obs_type_partial
        elif self.obs_mode == 'homo_red':
            obs_type = obs_type_full
        obs_space_type = Dict({"observation": obs_type, "action_mask": Box(low=0, high=1, shape=(self.num_actions,), dtype=np.int8)})
        return obs_space_type

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        #Simple dynamics. #same heading same place, heading plus, heading minus, move forward. And a tagging action.
        return Discrete(self.num_actions)

    def num_agents(self):
        return self.num_teams * self.num_agents_blue_team

    def render(self, plot_agent_trajs=False):
        # Note: Once you have the state and the flag locations, this module can be written disjoint of any other logic in the whole program.
        blue_agent_pos = [self.state[agent] for agent in self.blue_team_agents]
        red_agent_pos = [self.state[agent] for agent in self.red_team_agents]

        # Draw flags and draw agent current poses.
        fig, ax = draw_grid(grid_size=self.grid_size, blue_agent_pos=blue_agent_pos, red_agent_pos=red_agent_pos, blue_flag_loc=self.blue_flag_location, red_flag_loc=self.red_flag_location, blue_flag_img_path=self.blue_flag_img_path, red_flag_img_path=self.red_flag_img_path)
        
        if plot_agent_trajs:
            for agent in self.agents:
                # Interpolate from self.state_hist[agent].
                # Plot the interpolated trajectory.
                # Plot only if alive. Not relevant for CustomCTF_v1 but will be relevant for CustomCTF_v2.
                agent_pose_hist = self.state_hist[agent]
                if len(agent_pose_hist) == 1: continue
                #if len(agent_pose_hist) > 5: agent_pose_hist = agent_pose_hist[:] #agent_pose_hist[-5:]
                if len(agent_pose_hist) > 10: agent_pose_hist = agent_pose_hist[-10:] #agent_pose_hist[-5:]
                if agent.startswith('R') or agent.startswith('r'): color='red'
                elif agent.startswith('B') or agent.startswith('b'): color='blue'
                ax, x_spline, y_spline = draw_spline(ax=ax, poses_with_headings=agent_pose_hist, plot=True, color=color) #Take ax and plot agent_pose_hist and interpolated trajectory.

        plt.show()
        return fig, ax

class CustomCTF_v2(ParallelEnv): # v2: v1 + agent_deaths (max_respawns).
    meta_data = { "name": "custom_ctf_v2"}
    def __init__(self, grid_size=8, ctf_player_config="2v2", max_num_cycles=300, red_flag_locs=(6,6), obs_mode='hetero', verbose=False, seed=None, **kwargs):
        # Two swarms: red and blue.
        # Nash DQN as initial guess. Then warm start policy synthesis.
        # CTF on a grid of size [grid_size \times grid_size]
        self.metadata = { "name": "custom_ctf_v0"}
        self.render_mode = "human"
        self.verbose = verbose

        self.grid_size = grid_size #this defines the state space of the Markov Game and hence the observation_space.
        self.num_headings = 8  # Possible headings in degrees:(0, 45, 90, 135, 180, 225, 270, 315)
        self.num_actions = 5
        self.max_num_cycles = max_num_cycles

        if seed is not None:
            self.seed(seed=seed)
        else:
            self._seed = None
            self.np_random = None
            self.torch_rng = None

        self.num_teams = 2
        if ctf_player_config == "1v1":
            self.num_agents_blue_team = 1
        elif ctf_player_config == "2v2":
            self.num_agents_blue_team = 2
        self.num_agents_red_team = self.num_agents_blue_team
        self.num_agents = self.num_agents()

        self.blue_team_agents = ["Blue_" + "{}".format(i) for i in range(self.num_agents_blue_team)]
        self.red_team_agents = ["Red_" + "{}".format(i) for i in range(self.num_agents_red_team)]

        #self.agents = ["Blue_" + "{}".format(i) for i in range(self.num_agents_blue_team)] + ["Red_" + "{}".format(i) for i in range(self.num_agents_red_team)]
        self.agents = copy.deepcopy(self.blue_team_agents + self.red_team_agents)
        if self.verbose: print("self.agents: {}".format(self.agents))
        self.possible_agents = copy.deepcopy(self.agents)
        if self.verbose: print(self.possible_agents)

        self.obs_mode = obs_mode
        assert self.obs_mode in [ 'hetero', 'homo_blue', 'homo_red' ]
        self.observation_spaces = {agent: self.observation_space(agent=agent)["observation"] for agent in self.agents}
        self.action_spaces = {agent: self.action_space(agent=agent) for agent in self.agents}

        """
        For v1 of the CustomCTF environment: Can add logic to have a jitter / randomness in the flag locations.
        """
        self.deceptive_env = False #Note: Deception refers to uncertainty in the red_flag_location.
        self.blue_flag_location = kwargs.get('blue_flag_location', (1, 1)) #self.blue_flag_location = (1, 1)
        if red_flag_locs is None: self.red_flag_location = (self.grid_size - 2, self.grid_size - 2)
        elif (red_flag_locs is not None) and isinstance(red_flag_locs, tuple):
            self.red_flag_location = red_flag_locs
        elif isinstance(red_flag_locs, dict):
            assert list(red_flag_locs.keys()) == ['locs', 'p']
            assert isinstance(red_flag_locs['locs'], list) and isinstance(red_flag_locs['locs'][0], tuple) and isinstance(red_flag_locs['p'], list)
            assert len(red_flag_locs['locs']) == len(red_flag_locs['p'])
            assert abs(sum(red_flag_locs['p']) - 1) <= 1e-10
            """
            For instance, red_flag_locs = { 'locs': [(6,6), (3,7), (5,5)], 'p': [0.7, 0.2, 0.1] }
            """
            self.deceptive_env = True
            self.red_flag_locs = red_flag_locs
            self.red_flag_location = None

        self.blue_init_spawn_y_lim = 2 # Measured from bottom of the grid.
        self.possible_init_headings_blue_team = [0, 1, 2, 3] # possible init headings = (0, 45, 90, 135)
        self.red_init_spawn_y_lim = 2 # Measured from top of the grid.
        self.possible_init_headings_red_team = [0, 7, 6, 5] # possible init headings = (0, -45, -90, -135)

        # Tag spawning area is a rectangle, away from the flag in a corner.
        self.blue_tag_spawn_area = {"x_lim":(5, 7), "y_lim": (0, 1)} # (x, y) coordinate tuple -- x_low y_low and x_high y_high
        self.red_tag_spawn_area = {"x_lim":(0, 2), "y_lim": (6, 7)} # (x, y) coordinate tuple -- x_low y_low and x_high y_high

        self.current_step = 0
        self.state = { agent: None for agent in self.agents } #self.state = np.array([None for _ in self.agents])
        self.state_hist = { agent: [] for agent in self.agents }
        #self.reset()

        self.F = 100 #self.capture_flag_reward = 100
        self.tag_reward_shape = True
        self.T = 10 #10 #self.tagging_penalty = 10

        self.blue_flag_img_path = 'flag_imgs/blue_flag.png'
        self.red_flag_img_path = 'flag_imgs/red_flag.png'

        self.blue_flag_img = mpimg.imread('flag_imgs/blue_flag.png')
        self.red_flag_img = mpimg.imread('flag_imgs/red_flag.png')

        self.flag_capture_team = None

        self._init_args = ()
        self._init_kwargs = {"grid_size":grid_size, "ctf_player_config":ctf_player_config, "max_num_cycles":max_num_cycles, "red_flag_locs":red_flag_locs, "obs_mode":obs_mode, "verbose":verbose, "seed":seed}

    def __call__(self):
        return CustomCTF_v2(*self._init_args, **self._init_kwargs)

    def seed(self, seed=None):
        self.rngs = make_seeded_rngs(seed)
        self._seed = self.rngs["actual_seed"]
        self.np_random = self.rngs["np_random"]
        self.torch_rng = self.rngs["torch_rng"]
        return [self._seed]

    def set_obs_mode(self, obs_mode):
        assert obs_mode in [ 'hetero', 'homo_blue', 'homo_red' ]
        self.obs_mode = obs_mode
        return

    def agent_idx(self, agent_name):
        # Returns the agent_idx according to the self.agents list for the input agent_name.
        return

    def state(self):
        # Returns the global state as a Dict or an array?
        # I think the answer depends on the type of Agent names, if those are int then array otherwise Dict.
        return self.state

    def _sample_init_heading(self, agent_team, x, y):
        assert agent_team == "Blue" or agent_team == "Red"
        if agent_team == "Blue":
            possible_init_headings = copy.deepcopy(self.possible_init_headings_blue_team)
            if x == 0: possible_init_headings.remove(3)
            if x == self.grid_size - 1:
                possible_init_headings.remove(0)
                possible_init_headings.remove(1)
            heading = self.np_random.choice(possible_init_headings)
        elif agent_team == "Red":
            possible_init_headings = copy.deepcopy(self.possible_init_headings_red_team)
            if x == 0: possible_init_headings.remove(5)
            if x == 1:
                possible_init_headings.remove(0)
                possible_init_headings.remove(7)
            heading = self.np_random.choice(possible_init_headings)
        return heading

    def reset(self, seed = None, options = None):
      #### Debugging prints: print valid actions for each agent at the step. => a) actions available at the walls b) tagging action available or not.
        if seed is not None:
            self.seed(seed=seed)

        self.current_step = 0
        self.agents = copy.deepcopy(self.blue_team_agents + self.red_team_agents) # !!! 04/25 3.56 pm: DON'T UNDERSTAND THIS BUG.
        """
        NOTE: In the current implementation, v0, flag locations are parameters of the Markov game and are hence not a part of the state, agent-wise or global.
        For v1: Can add logic to have a jitter / randomness in the flag locations.
        """
        #self.blue_flag_location = (1, 1)
        #self.red_flag_location = (self.grid_size - 2, self.grid_size - 2) # We assume the grid_size > 3 or something like that for the flag placements to be a non-trivial game scenario.

        self.blue_flag_location = self.blue_flag_location
        if self.deceptive_env:
            red_flag_idx = self.np_random.choice(len(self.red_flag_locs['locs']), p=self.red_flag_locs['p'])
            self.red_flag_location = self.red_flag_locs['locs'][red_flag_idx]
            print("RED FLAG LOCATION: {}".format(self.red_flag_location))

        blue_team_init_xys = []
        for agent_iter, blue_agent_num in enumerate(range(self.num_agents_blue_team)):
            if agent_iter == 0:
                rand_x = self.np_random.integers(0, self.grid_size)
                rand_y = self.np_random.integers(0, self.blue_init_spawn_y_lim + 1)
                rand_theta = self._sample_init_heading(agent_team="Blue", x=rand_x, y=rand_y)
                self.state["Blue_{}".format(blue_agent_num)] = np.array([rand_x, rand_y, rand_theta])
                blue_team_init_xys.append((rand_x, rand_y))
                continue
            # A while loop to break ties in the x-y location to prevent collision during agent initialization.
            while (rand_x, rand_y) in blue_team_init_xys:
                rand_x = self.np_random.integers(0, self.grid_size)
                rand_y = self.np_random.integers(0, self.blue_init_spawn_y_lim + 1)
            blue_team_init_xys.append((rand_x, rand_y))
            rand_theta = self._sample_init_heading(agent_team="Blue", x=rand_x, y=rand_y)
            self.state["Blue_{}".format(blue_agent_num)] = np.array([rand_x, rand_y, rand_theta])

        red_team_init_xys = []
        for agent_iter, red_agent_num in enumerate(range(self.num_agents_red_team)):
            if agent_iter == 0:
                rand_x = self.np_random.integers(0, self.grid_size)
                rand_y = self.np_random.integers(self.grid_size - self.red_init_spawn_y_lim - 1, self.grid_size)
                rand_theta = self._sample_init_heading(agent_team="Red", x=rand_x, y=rand_y)
                self.state["Red_{}".format(red_agent_num)] = np.array([rand_x, rand_y, rand_theta])
                red_team_init_xys.append((rand_x, rand_y))
                continue
            # A while loop to break ties in the x-y location to prevent collision during agent initialization.
            while (rand_x, rand_y) in red_team_init_xys:
                rand_x = self.np_random.integers(0, self.grid_size)
                rand_y = self.np_random.integers(self.grid_size - self.red_init_spawn_y_lim - 1, self.grid_size)
            red_team_init_xys.append((rand_x, rand_y))
            rand_theta = self._sample_init_heading(agent_team="Red", x=rand_x, y=rand_y)
            self.state["Red_{}".format(red_agent_num)] = np.array([rand_x, rand_y, rand_theta])

        self.state_hist = {agent: [self.state[agent]] for agent in self.agents}
        
        #global_state = [self.state[agent] for agent in self.agents]
        #obs = {agent: global_state for agent in self.agents} # global_state dict. The prisoner guard example on ParallelEnv has this as a tuple, and you have this as a List currently.
        obs = self._observations()
        info = {agent: {} for agent in self.agents}
        #print("self.agents: {}".format(self.agents))
        return obs, info

    def step(self, actions):
      #### Debugging prints: print valid actions for each agent at the step. => a) actions available at the walls b) tagging action available or not.
        ### v0: No collision logic.
        # TO-DO for v1: Resolve collisions and tagging and re-spawning. # Collisions within a team prevented via action_masking. Where to write actions_available_to_an_agent?

        # The input actions is a Dict from agent_name to actions. Add an assert statement for Type of actions.
        """
        Action keys:
        0 = same pos, same heading
        1 = same pos, minus heading
        2 = same pos, plus heading
        3 = advance pos, same heading
        4 = tagging action
        """
        if self.verbose: print("Current Step: {}".format(self.current_step))
        if self.verbose: print("Before stepping self.agents state: {}".format(self.state))
        if self.verbose: print("Actions: {}".format(actions))

        infos = {agent: {} for agent in self.agents}
        for agent in self.agents:
            infos[agent]["CTF"] = None # "CTF" descriptor is for the last step of the game on which team captured the flag if any!
        tag_bool = {agent: False for agent in self.agents}
        reach_goal_bool = {agent: False for agent in self.agents}
        new_state = {agent: None for agent in self.agents}

        rewards = {agent: 0. for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        for agent in self.agents:
          assert actions[agent] in self.valid_actions(self.state, agent)

        # Actions dict = actions.
        for agent in self.agents:
            assert actions[agent] in self.action_space(agent=agent)
            tag_bool[agent], reach_goal_bool[agent], new_state[agent] = self.dynamics(agent=agent, actions=actions)
            #print("agent: {}, tag_bool_agent: {}".format(agent, tag_bool[agent]))

        for agent in self.agents:
            self.state[agent] = new_state[agent]

        blue_tag = np.sum([ tag_bool[agent] for agent in self.blue_team_agents ])
        red_tag = np.sum([ tag_bool[agent] for agent in self.red_team_agents ])
        
        if self.verbose:
            for agent in self.blue_team_agents: print("agent: {}, blue tag check : {}".format(agent, tag_bool[agent]))
            for agent in self.red_team_agents: print("agent: {}, red tag check : {}".format(agent, tag_bool[agent]))
            for agent in self.blue_team_agents: print("agent: {}, blue reach_goal check : {}".format(agent, reach_goal_bool[agent]))
            for agent in self.red_team_agents: print("agent: {}, red reach_goal check : {}".format(agent, reach_goal_bool[agent]))
        
        blue_reach_goal = np.sum([ reach_goal_bool[agent] for agent in self.blue_team_agents ])
        red_reach_goal = np.sum([ reach_goal_bool[agent] for agent in self.red_team_agents ])

        if blue_reach_goal + red_reach_goal == 0:
            # Flag not captured.
            terminate = False
            terminations = {agent: terminate for agent in self.agents}
            # Assign tagging rewards if any.
            for agent in self.blue_team_agents:
                rewards[agent] +=  - blue_tag * (self.T)
                rewards[agent] += + red_tag * (self.T)
            for agent in self.red_team_agents:
                rewards[agent] += + blue_tag * (self.T)
                rewards[agent] += - red_tag * (self.T)
        else:
            # Flag captured.
            terminate = True
            terminations = {agent: terminate for agent in self.agents}
            for agent in self.blue_team_agents:
                if blue_reach_goal:
                    rewards[agent] += self.F
                    infos[agent]["CTF"] = True
                    self.flag_capture_team = "Blue"
                elif red_reach_goal:
                    rewards[agent] += - self.F
                    infos[agent]["CTF"] = False
                    self.flag_capture_team = "Red"
            for agent in self.red_team_agents:
                if blue_reach_goal:
                    rewards[agent] += - self.F
                    infos[agent]["CTF"] = False
                elif red_reach_goal:
                    rewards[agent] += self.F
                    infos[agent]["CTF"] = True
                pass

        # TO-DO: Determine rewards. Reward structure: team reward: agents share the same reward within a team.
        obs = self._observations()
        for agent in self.agents:
            if tag_bool[agent]:
                self.state_hist[agent] = [self.state[agent]]
                continue
            self.state_hist[agent].append(self.state[agent])

        self.current_step += 1
        if self.current_step == self.max_num_cycles: truncations = {agent: True for agent in self.agents}

        agents_to_remove = []
        for agent in self.agents:
            if terminations[agent] or truncations[agent]:
                agents_to_remove.append(agent)

        # Remove done agents
        for agent in agents_to_remove:
            self.agents.remove(agent)
        if self.verbose:
            print("Updated self.agents state: {}".format(self.state))
            print("rewards: {}".format(rewards))
            print("terminations: {}".format(terminations))
            print("truncations: {}".format(truncations))
        return obs, rewards, terminations, truncations, infos

    def global_state_to_agent_local_obs(self, agent):
        """
        Hard-coded for a 2v2 scenario.
        For more agents == GNN.
        ### For the second iteration of the PSRO loop: CTDE? Fixing one team behaviour, have a CTDE scheme for the other team.
        """
        """
        Idea: Permutation invariance for better learning: ordering of adversaries shouldn't matter. Same strategy.
        """
        # coordinate of agent, heading, home_flag_rel_location, away_flag_rel_location, friend_rel_location, friend_rel_pose, adversary_1_rel_location, adversary_1_rel_pose, adversary_2_rel_location, adversary_2_rel_pose
        # 3 = self coordinate, 2 = home flag location, 2 = away flag location, 3 = friend rel pos, 3 = adversary 1 rel pos, 3 = adversary 2 rel pos
        assert agent in self.agents
        agent_type = agent[0]

        blue_team_agents = copy.deepcopy(self.blue_team_agents)
        red_team_agents = copy.deepcopy(self.red_team_agents)

        if agent_type == "B":
            blue_team_agents.remove(agent)
            ally_agent = blue_team_agents[0] #Hard-coded for 2v2 scenario.
            adv_agents = red_team_agents
        if agent_type == "R":
            red_team_agents.remove(agent)
            ally_agent = red_team_agents[0] #Hard-coded for 2v2 scenario.
            adv_agents = blue_team_agents

        obs = np.zeros(16, dtype=np.int8)
        #obs.append(self.state[agent])
        obs[:3] = self.state[agent]
        if agent[0] == "R":
            obs[3:5] = np.array(self.red_flag_location) - np.array(self.state[agent][:2]) # HOME FLAG.
            obs[5:7] = np.array(self.blue_flag_location) - np.array(self.state[agent][:2]) # AWAY FLAG.
        elif agent[0] == "B":
            obs[3:5] = np.array(self.blue_flag_location) - np.array(self.state[agent][:2]) # HOME FLAG.
            obs[5:7] = np.array(self.red_flag_location) - np.array(self.state[agent][:2]) # AWAY FLAG.

        obs[7:9] = np.array(self.state[ally_agent][:2]) - np.array(self.state[agent][:2]) # Friend rel xy position
        obs[9] = (self.state[ally_agent][2] - self.state[agent][2]) % (self.num_headings) # Friend rel heading

        obs[10:12] = np.array(self.state[adv_agents[0]][:2]) - np.array(self.state[agent][:2]) # Adversary 1 rel xy position
        obs[12] = (self.state[adv_agents[0]][2] - self.state[agent][2]) % (self.num_headings) # Adversary 1 rel heading

        obs[13:15] = np.array(self.state[adv_agents[1]][:2]) - np.array(self.state[agent][:2]) # Adversary 2 rel xy position
        obs[15] = (self.state[adv_agents[1]][2] - self.state[agent][2]) % (self.num_headings) # Adversary 2 rel heading

        ## Some simple reward shaping for some behaviour learned for 2v2 CTF policies. Then later optimize the reward shaping parameters via Bayesian optimization.
        
        obs_mask = np.ones(16, dtype=np.bool)
        if self.obs_mode == 'hetero':
            if agent[0] == 'R': pass
            elif agent[0] == 'B': obs_mask[5:7] = np.array([False, False])
        elif self.obs_mode == 'homo_blue':
            obs_mask[5:7] = np.array([False, False])
        elif self.obs_mode == 'homo_red':
            pass
        obs = obs[obs_mask]
        return obs

    def _observations(self):
        """
        PENDING: 06/01/2025: Latest change pending to be made.
        """
        # Implements the action_masks from the global state.
        obs = {}
        global_state = [self.state[agent] for agent in self.agents]
        for agent in self.agents:
            ##agent_obs = global_state # Had to comment due to some API glue-ing error that Google Colab pointed out.
            #agent_obs = np.concatenate([s for s in global_state]).astype(np.int8)

            agent_obs = self.global_state_to_agent_local_obs(agent=agent) # v2 observations for better policy sharing.

            valid = self.valid_actions(global_state_dict=self.state, agent=agent)
            action_mask = np.zeros(self.num_actions, dtype=np.int8)
            action_mask[valid] = 1
            obs[agent] = { "observation": agent_obs, "action_mask": action_mask }
        return obs

    def dynamics(self, agent, actions):
        # Single agent dynamics.
        ### Action keys:
        ## 0 = same pos, same heading
        ## 1 = same pos, minus heading
        ## 2 = same pos, plus heading
        ## 3 = advance pos, same heading
        ## 4 = tagging action

        assert type(agent) == str
        agent_team = None
        if agent[0] == "R": agent_team = "Red"
        elif agent[0] == "B": agent_team = "Blue"

        tag_bool, reach_goal_bool = False, False

        x, y, heading = self.state[agent]
        action = actions[agent]
        x_, y_, heading_ = None, None, None

        # Check if there are incoming tagging actions from nearby agents.
        potential_incoming_tagging_agents = []
        incoming_tagging_agents = []
        for neighbouring_agent in self.agents:
            if neighbouring_agent == agent: continue
            else:
                x_neighbour, y_neighbour, heading_neighbour = self.state[neighbouring_agent]
                if abs(x - x_neighbour) <= 1 and abs(y - y_neighbour) <= 1:
                    neighbour_point_to_x, neighbour_point_to_y =  self.state[neighbouring_agent][:2] + self._heading_to_direction_vector(heading=heading_neighbour)
                    if (neighbour_point_to_x, neighbour_point_to_y) == (x, y): potential_incoming_tagging_agents.append(neighbouring_agent)
                    else: continue

        for agent_iter in potential_incoming_tagging_agents:
            if actions[agent_iter] == 4: incoming_tagging_agents.append(agent_iter)
            else: continue

        if len(incoming_tagging_agents) == 0:
            # No incoming tags.
            if action == 0:
                # same pos, same heading
                x_, y_, heading_ = x, y, heading
                pass
            elif action == 1:
                # same pos, minus heading
                x_, y_, heading_ = x, y, (heading - 1) % self.num_headings
                pass
            elif action == 2:
                # same pos, plus heading
                x_, y_, heading_ = x, y, (heading + 1) % self.num_headings
                pass
            elif action == 3:
                # advance pos, same heading
                x_, y_, heading_ = x, y, heading
                if heading == 0:
                    x_, y_ = x + 1, y
                elif heading == 1:
                    x_, y_ = x + 1, y + 1
                elif heading == 2:
                    x_, y_ = x, y + 1
                elif heading == 3:
                    x_, y_ = x - 1, y + 1
                elif heading == 4:
                    x_, y_ = x - 1, y
                elif heading == 5:
                    x_, y_ = x - 1, y - 1
                elif heading == 6:
                    x_, y_ = x, y - 1
                elif heading == 7:
                    x_, y_ = x + 1, y - 1
            elif action == 4: # Tagging action.
                x_, y_, heading_ = x, y, heading

            flag_location = None
            #### HAD A HUGE BUG HERE. Now fixed on 04/29 2.45 pm. Interchanged flag locations defining the goal_reach_variable.
            if agent_team == "Red": flag_location = self.blue_flag_location
            elif agent_team == "Blue": flag_location = self.red_flag_location

            if (x_, y_) == flag_location:
                reach_goal_bool = True

        else:
            # Incoming tags. Spawn in the tagging area.
            # Multiple tags = single penalty.
            if self.verbose: print("TAAAAAAAAAAGGGGGGGGGGGGGGGG..............")
            tag_bool = True
            spawn_area = None
            """
            Spawning area is a rectangle, away from the flag in a corner.
            blue_tag_spawn_area = {"x_lim":(5, 7), "y_lim": (0, 1)} # (x, y) coordinate tuple -- x_low y_low and x_high y_high
            red_tag_spawn_area = {"x_lim":(0, 2), "y_lim": (6, 7)} # (x, y) coordinate tuple -- x_low y_low and x_high y_high
            """
            if agent_team == "Red":
                spawn_area = self.red_tag_spawn_area
                possible_spawn_headings = [0, 7, 6, 5] # [0, 7, 6, 5], possible init headings = (0, -45, -90, -135)
            elif agent_team == "Blue":
                spawn_area = self.blue_tag_spawn_area
                possible_spawn_headings = [0, 1, 2, 3] # possible init headings = (0, 45, 90, 135)
            x_prime = self.np_random.integers( spawn_area["x_lim"][0], spawn_area["x_lim"][1] + 1)
            y_prime = self.np_random.integers( spawn_area["y_lim"][0], spawn_area["y_lim"][1] + 1)

            if x_prime == 0: # Implied that agent is red_team.
                try: possible_spawn_headings.remove(5)
                except: pass
            if x_prime == self.grid_size - 1: # Implied that agent is blue_team.
                try: possible_spawn_headings.remove(0)
                except: pass
                try: possible_spawn_headings.remove(1)
                except: pass

            heading_prime = self.np_random.choice(possible_spawn_headings)
            x_, y_, heading_ = x_prime, y_prime, heading_prime

        new_state = (x_, y_, heading_)
        return tag_bool, reach_goal_bool, new_state

    def _heading_to_direction_vector(self, heading):
        assert heading in range(self.num_headings)
        vec = None
        if heading == 0:
            vec = np.array([ +1, 0 ])
        elif heading == 1:
            vec = np.array([ +1, +1 ])
        elif heading == 2:
            vec = np.array([ 0, +1 ])
        elif heading == 3:
            vec = np.array([ -1, +1 ])
        elif heading == 4:
            vec = np.array([ -1, 0 ])
        elif heading == 5:
            vec = np.array([ -1, -1 ])
        elif heading == 6:
            vec = np.array([ 0, -1 ])
        elif heading == 7:
            vec = np.array([ +1, -1 ])
        return vec

    def valid_actions(self, global_state_dict, agent):
        """
        ### Action keys:
        ## 0 = same pos, same heading
        ## 1 = same pos, minus heading
        ## 2 = same pos, plus heading
        ## 3 = advance pos, same heading
        ## 4 = tagging action
        """

        # NOTE: No collision logic in v0.
        # Takes in as input global_state of the world and the agent name and returns valid actions for the agent in the global_state of the world.
        # Utility: to decide action_masking for the agent in the global_state.

        agent_state = copy.deepcopy(global_state_dict[agent])
        # Action masking due to walls of the environment.
        x, y, heading = agent_state
        point_to_x, point_to_y =  agent_state[:2] + self._heading_to_direction_vector(heading=heading)

        valid_actions = [0, 1, 2, 3]
        invalid_action_x = None
        invalid_action_y = None

        if x == 0 and heading == 2:
            invalid_action_x = 2
        if x == 0 and heading == 6:
            invalid_action_x = 1
        if x == 0 and heading in [3, 4, 5]:
          invalid_action_x = 3
        if x == self.grid_size - 1 and heading == 2:
            invalid_action_x = 1
        if x == self.grid_size - 1 and heading == 6:
            invalid_action_x = 2
        if x == self.grid_size - 1 and heading in [0, 1, 7]:
          invalid_action_x = 3
        try:
            valid_actions.remove(invalid_action_x)
        except: pass

        if y == 0 and heading == 0:
            invalid_action_y = 1
        if y == 0 and heading == 4:
            invalid_action_y = 2
        if y == 0 and heading in [5, 6, 7]:
            invalid_action_y = 3
        if y == self.grid_size - 1 and heading == 0:
            invalid_action_y = 2
        if y == self.grid_size - 1 and heading == 4:
            invalid_action_y = 1
        if y == self.grid_size - 1 and heading in [1, 2, 3]:
            invalid_action_y = 3
        try:
            valid_actions.remove(invalid_action_y)
        except: pass

        # Asking masking on the Tagging space due to nearby agents.
        # Logic to see nearby agents position-wise: once a list of nearby agents is generated: check if heading is aligned with the neighbouring agent for potential tagging action.
        target_tagging_agents = []
        for potential_nearby_agent in self.agents:
            if potential_nearby_agent[0] == agent[0]: continue # Cannot tag same team member. HAD A BUG HERE previously. Corrected now.
            else:
                x_neighbour, y_neighbour, heading_neighbour = global_state_dict[potential_nearby_agent]
                if abs(x - x_neighbour) <= 1 and abs(y - y_neighbour) <= 1:
                    if (x_neighbour, y_neighbour) == (point_to_x, point_to_y): target_tagging_agents.append(potential_nearby_agent)
                    else: continue
                pass
            pass
        if len(target_tagging_agents) > 0: valid_actions.append(4) #Tagging action available.

        return valid_actions

    def state_space(self, agent):
        # The state of the agent on the grid: (x, y, theta).
        return MultiDiscrete([self.grid_size, self.grid_size, self.num_headings])

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Each agent can observe the state of all the agents.
        # obs_type = MultiDiscrete([self.grid_size, self.grid_size, self.num_headings, self.grid_size, self.grid_size, self.num_headings, self.grid_size, self.grid_size, self.num_headings, self.grid_size, self.grid_size, self.num_headings]) #Tuple([ self.state_space(agent) for agent in self.agents ])
        # coordinate of agent, heading, home_flag_rel_location, away_flag_rel_location, friend_rel_location, friend_rel_pose, adversary_1_rel_location, adversary_1_rel_pose, adversary_2_rel_location, adversary_2_rel_pose
        # 3 = self coordinate, 2 = home flag location, 2 = away flag location, 3 = friend rel pos, 3 = adversary 1 rel pos, 3 = adversary 2 rel pos
        
        obs_type_full = Box(low=0, high=self.grid_size, shape=(2 + 2 + len(self.possible_agents) * 3,), dtype=np.int8) # Away flag is visible.
        obs_type_partial = Box(low=0, high=self.grid_size, shape=(2 + 0 + len(self.possible_agents) * 3,), dtype=np.int8) # Away flag not visible.
        if self.obs_mode == 'hetero':
            if agent[0] == 'R': obs_type = obs_type_full
            elif agent[0] == 'B': obs_type = obs_type_partial
        elif self.obs_mode == 'homo_blue':
            obs_type = obs_type_partial
        elif self.obs_mode == 'homo_red':
            obs_type = obs_type_full
        obs_space_type = Dict({"observation": obs_type, "action_mask": Box(low=0, high=1, shape=(self.num_actions,), dtype=np.int8)})
        return obs_space_type

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        #Simple dynamics. #same heading same place, heading plus, heading minus, move forward. And a tagging action.
        return Discrete(self.num_actions)

    def num_agents(self):
        return self.num_teams * self.num_agents_blue_team

    def render(self):
        # Note: Once you have the state and the flag locations, this module can be written disjoint of any other logic in the whole program.
        blue_agent_pos = [self.state[agent] for agent in self.blue_team_agents]
        red_agent_pos = [self.state[agent] for agent in self.red_team_agents]
        fig, ax = draw_grid(grid_size=self.grid_size, blue_agent_pos=blue_agent_pos, red_agent_pos=red_agent_pos, blue_flag_loc=self.blue_flag_location, red_flag_loc=self.red_flag_location, blue_flag_img_path=self.blue_flag_img_path, red_flag_img_path=self.red_flag_img_path)
        plt.show()
        return fig, ax

class DefenseCTF(ParallelEnv):
    """
    Note on Wimbledon Final day, July 13 2025, 06.16 pm: some weird bug with observation space dimension mismatch. Fixing it temporarily for now because if I remember heterogenous observation spaces worked for CustomCTF_v1 just fine without agent_deaths.
    """
    meta_data = { "name": "DefenseCTF"}
    def __init__(self, defense_team, grid_size=8, ctf_player_config="2v2", max_num_cycles=300, red_flag_locs=(6,6), max_respawns=2, tag_reward_shaping=True, reward_shaping={}, verbose=False, obs_mode=None, seed=None):
        assert defense_team in ['Blue', 'Red']
        assert isinstance(reward_shaping, dict)

        self.metadata = { "name": "DefenseCTF"}
        self.render_mode = "human"
        self.verbose = verbose
        
        if seed is not None:
            self.seed(seed=seed)
        else:
            self._seed = None
            self.np_random = None
            self.torch_rng = None

        self.defense_team = defense_team
        if obs_mode is not None:
            if self.defense_team == 'Blue': obs_mode = 'hetero'
            elif self.defense_team == 'Red': obs_mode = 'homo_red'
            self.set_obs_mode(obs_mode=obs_mode)
        else: self.set_obs_mode(obs_mode='hetero') #self.set_obs_mode(obs_mode=obs_mode)
        if self.defense_team == 'Red': self.away_team = 'Blue'
        else: self.away_team = 'Red'

        """
        Note: Defense_team influences obs_mode. For eg, if blue is defending, then the red team has to have the blue_flag i.e. the away flag as a part of the policy. This means Blue team has obs_mode = homo_blue but Red team has obs_mode = hetero.
        Note: If Red is defending, it has to be trained against a Blue that knows the Red flag location.
        If defense_mode = 'Blue':
            obs_mode of Blue agents = obs_partial
            obs_mode of Red agents = obs_full

        If defense_mode = 'Red'
            obs_mode of Blue agents = obs_full
            obs_mode of Red agents = obs_full
        """
        self.grid_size = grid_size #this defines the state space of the Markov Game and hence the observation_space.
        self.num_headings = 8  # Possible headings in degrees:(0, 45, 90, 135, 180, 225, 270, 315)
        self.num_actions = 5
        self.max_respawns = max_respawns
        self.max_num_cycles = max_num_cycles

        self.num_teams = 2
        if ctf_player_config == "1v1":
            self.num_agents_blue_team = 1
        elif ctf_player_config == "2v2":
            self.num_agents_blue_team = 2
        self.num_agents_red_team = self.num_agents_blue_team
        self.num_agents = self.num_agents()

        self.blue_team_agents = ["Blue_" + "{}".format(i) for i in range(self.num_agents_blue_team)]
        self.red_team_agents = ["Red_" + "{}".format(i) for i in range(self.num_agents_red_team)]

        self.possible_agents = copy.deepcopy(self.blue_team_agents + self.red_team_agents)
        self.agents = copy.deepcopy(self.possible_agents)
        if self.verbose: print("self.agents: {}".format(self.agents))
        if self.verbose: print("self.possible_agents: {}".format(self.possible_agents))
        self.agent_alive = {agent: None for agent in self.possible_agents} # set to True for each agent on resetting.

        self.observation_spaces = {agent: self.observation_space(agent=agent)["observation"] for agent in self.possible_agents}
        self.action_spaces = {agent: self.action_space(agent=agent) for agent in self.possible_agents}

        """
        For v1 of the CustomCTF environment: Can add logic to have a jitter / randomness in the flag locations.
        """
        self.deceptive_env = False #Note: Deception refers to uncertainty in the red_flag_location.
        self.blue_flag_location = (1, 1)
        assert (red_flag_locs is not None) and isinstance(red_flag_locs, tuple)
        self.red_flag_location = red_flag_locs
        
        if self.defense_team == 'Red': self.flag_location = self.red_flag_location
        else: self.flag_location = self.blue_flag_location

        self.blue_init_spawn_y_lim = 2 # Measured from bottom of the grid.
        self.possible_init_headings_blue_team = [0, 1, 2, 3] # possible init headings = (0, 45, 90, 135)
        self.red_init_spawn_y_lim = 0 #1 #2 # Measured from top of the grid.
        self.possible_init_headings_red_team = [0, 7, 6, 5] # possible init headings = (0, -45, -90, -135)

        # Tag spawning area is a rectangle, away from the flag in a corner.
        self.blue_tag_spawn_area = {"x_lim":(0, self.grid_size - 1), "y_lim": (0, 1)} # {"x_lim":(0, 7), "y_lim": (0, 1)} # (x, y) coordinate tuple -- x_low y_low and x_high y_high
        self.red_tag_spawn_area = {"x_lim":(0, self.grid_size - 1), "y_lim": (6, 7)} # {"x_lim":(0, 2), "y_lim": (6, 7)} # (x, y) coordinate tuple -- x_low y_low and x_high y_high

        self.current_step = 0
        self.state = { agent: None for agent in self.possible_agents}
        self.respawn_info_state = {agent: 0 for agent in self.possible_agents}

        self.F = 100 #self.capture_flag_reward = 100
        self.tag_reward_shaping = tag_reward_shaping
        self.reward_shaping = reward_shaping
        if self.tag_reward_shaping:
            if 'tag_reward' in self.reward_shaping: self.T = self.reward_shaping['tag_reward']
            else: self.T = 10
        else: self.T = 0

        self.blue_flag_img_path = 'flag_imgs/blue_flag.png'
        self.red_flag_img_path = 'flag_imgs/red_flag.png'

        self.blue_flag_img = mpimg.imread('flag_imgs/blue_flag.png')
        self.red_flag_img = mpimg.imread('flag_imgs/red_flag.png')

        self.flag_capture_team = None

        self._init_args = (defense_team)
        self._init_kwargs = {"grid_size":grid_size, "ctf_player_config":ctf_player_config, "max_num_cycles":max_num_cycles, "red_flag_locs":red_flag_locs, "max_respawns":max_respawns, "tag_reward_shaping":tag_reward_shaping, "reward_shaping":reward_shaping, "verbose":verbose, "obs_mode":obs_mode, "seed":seed}
    
    def __call__(self):
        return DefenseCTF(*self._init_args, **self._init_kwargs)

    def seed(self, seed=None):
        self.rngs = make_seeded_rngs(seed)
        self._seed = self.rngs["actual_seed"]
        self.np_random = self.rngs["np_random"]
        self.torch_rng = self.rngs["torch_rng"]
        return [self._seed]

    def set_obs_mode(self, obs_mode):
        assert obs_mode in [ 'hetero', 'homo_blue', 'homo_red' ]
        self.obs_mode = obs_mode
        return

    def _sample_init_heading(self, agent_team, x, y):
        assert agent_team == "Blue" or agent_team == "Red"
        if agent_team == "Blue":
            possible_init_headings = copy.deepcopy(self.possible_init_headings_blue_team)
            if x == 0: possible_init_headings.remove(3)
            if x == self.grid_size - 1:
                possible_init_headings.remove(0)
                possible_init_headings.remove(1)
            heading = self.np_random.choice(possible_init_headings)
        elif agent_team == "Red":
            possible_init_headings = copy.deepcopy(self.possible_init_headings_red_team)
            if x == 0: possible_init_headings.remove(5)
            if x == 1:
                possible_init_headings.remove(0)
                possible_init_headings.remove(7)
            heading = self.np_random.choice(possible_init_headings)
        return heading

    def reset(self, seed = None, options = None):
      #### Debugging prints: print valid actions for each agent at the step. => a) actions available at the walls b) tagging action available or not.
        if seed is not None:
            self.seed(seed=seed)

        self.current_step = 0
        self.possible_agents = copy.deepcopy(self.blue_team_agents + self.red_team_agents)
        self.agents = copy.deepcopy(self.possible_agents) # !!! 04/25 3.56 pm: DON'T UNDERSTAND THIS BUG.
        self.agent_alive = {agent: True for agent in self.possible_agents}
        self.respawn_info_state = {agent: 0 for agent in self.possible_agents}

        """
        NOTE: In the current implementation, v0, flag locations are parameters of the Markov game and are hence not a part of the state, agent-wise or global.
        For v1: Can add logic to have a jitter / randomness in the flag locations.
        """
        #self.blue_flag_location = (1, 1)
        #self.red_flag_location = (self.grid_size - 2, self.grid_size - 2) # We assume the grid_size > 3 or something like that for the flag placements to be a non-trivial game scenario.

        self.blue_flag_location = self.blue_flag_location
        if self.deceptive_env:
            red_flag_idx = self.np_random.choice(len(self.red_flag_locs['locs']), p=self.red_flag_locs['p'])
            self.red_flag_location = self.red_flag_locs['locs'][red_flag_idx]
            print("RED FLAG LOCATION: {}".format(self.red_flag_location))

        blue_team_init_xys = []
        for agent_iter, blue_agent_num in enumerate(range(self.num_agents_blue_team)):
            if agent_iter == 0:
                rand_x = self.np_random.integers(0, self.grid_size)
                rand_y = self.np_random.integers(0, self.blue_init_spawn_y_lim + 1)
                rand_theta = self._sample_init_heading(agent_team="Blue", x=rand_x, y=rand_y)
                self.state["Blue_{}".format(blue_agent_num)] = np.array([rand_x, rand_y, rand_theta])
                blue_team_init_xys.append((rand_x, rand_y))
                continue
            # A while loop to break ties in the x-y location to prevent collision during agent initialization.
            while (rand_x, rand_y) in blue_team_init_xys:
                rand_x = self.np_random.integers(0, self.grid_size)
                rand_y = self.np_random.integers(0, self.blue_init_spawn_y_lim + 1)
            blue_team_init_xys.append((rand_x, rand_y))
            rand_theta = self._sample_init_heading(agent_team="Blue", x=rand_x, y=rand_y)
            self.state["Blue_{}".format(blue_agent_num)] = np.array([rand_x, rand_y, rand_theta])

        red_team_init_xys = []
        for agent_iter, red_agent_num in enumerate(range(self.num_agents_red_team)):
            if agent_iter == 0:
                rand_x = self.np_random.integers(0, self.grid_size)
                rand_y = self.np_random.integers(self.grid_size - self.red_init_spawn_y_lim - 1, self.grid_size)
                rand_theta = self._sample_init_heading(agent_team="Red", x=rand_x, y=rand_y)
                self.state["Red_{}".format(red_agent_num)] = np.array([rand_x, rand_y, rand_theta])
                red_team_init_xys.append((rand_x, rand_y))
                continue
            # A while loop to break ties in the x-y location to prevent collision during agent initialization.
            while (rand_x, rand_y) in red_team_init_xys:
                rand_x = self.np_random.integers(0, self.grid_size)
                rand_y = self.np_random.integers(self.grid_size - self.red_init_spawn_y_lim - 1, self.grid_size)
            red_team_init_xys.append((rand_x, rand_y))
            rand_theta = self._sample_init_heading(agent_team="Red", x=rand_x, y=rand_y)
            self.state["Red_{}".format(red_agent_num)] = np.array([rand_x, rand_y, rand_theta])

        #global_state = [self.state[agent] for agent in self.agents]
        #obs = {agent: global_state for agent in self.agents} # global_state dict. The prisoner guard example on ParallelEnv has this as a tuple, and you have this as a List currently.
        obs = self._observations()
        info = {agent: {} for agent in self.possible_agents}
        #print("self.agents: {}".format(self.agents))
        return obs, info

    def step(self, actions):
        """
        Comment: Reward sharing breaks between agents on the same team because of potential agent death.
        """
      #### Debugging prints: print valid actions for each agent at the step. => a) actions available at the walls b) tagging action available or not.
        ### v0: No collision logic.
        # TO-DO for v1: Resolve collisions and tagging and re-spawning. # Collisions within a team prevented via action_masking. Where to write actions_available_to_an_agent?

        # The input actions is a Dict from agent_name to actions. Add an assert statement for Type of actions.
        """
        Action keys:
        0 = same pos, same heading
        1 = same pos, minus heading
        2 = same pos, plus heading
        3 = advance pos, same heading
        4 = tagging action
        """
        state_before_step = copy.deepcopy(self.state)

        if self.verbose: print("Current Step: {}".format(self.current_step))
        if self.verbose: print("Before stepping self.agents state: {}".format(self.state))
        if self.verbose: print("Actions: {}".format(actions))

        infos = {agent: {} for agent in self.possible_agents}
        for agent in self.possible_agents:
            infos[agent]["Win"] = None # "Win" descriptor is for the last step of the game on which the winning team is announced if any!
        
        tag_bool = {agent: False for agent in self.possible_agents if self.agent_alive[agent]}
        respawn_bool = {agent: False for agent in self.possible_agents if self.agent_alive[agent]}
        death_bool = {agent: False for agent in self.possible_agents if self.agent_alive[agent]}
        reach_goal_bool = {agent: False for agent in self.possible_agents if self.agent_alive[agent]}
        new_state = {agent: None for agent in self.possible_agents}
        
        rewards = {agent: 0. for agent in self.possible_agents}
        terminations = {agent: False for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}
        
        alive_before_step = [agent for agent in self.possible_agents if self.agent_alive[agent]]
        # Gather new states after applying actions in the new_state dict from the following for-loop.
        for agent in self.possible_agents:
            agent_team = 'Red' if agent[0] == 'R' else 'Blue'
            if not self.agent_alive[agent]: continue
            # Agent alive.
            assert actions[agent] in self.action_space(agent=agent), "agent = {}, actions[agent] = {} in self.action_space(agent=agent) = {}".format(agent, actions[agent], self.action_space(agent=agent))
            assert actions[agent] in self.valid_actions(state_before_step, agent), "agent = {}, actions[agent] = {} in self.valid_actions(self.state, agent) = {}".format(agent, actions[agent], self.valid_actions(self.state, agent))
            
            tag_bool[agent], respawn_bool[agent], death_bool[agent], reach_goal_bool[agent], new_state[agent] = self.dynamics(agent=agent, actions=actions)
            #print("agent: {}, tag_bool_agent: {}".format(agent, tag_bool[agent]))

        new_dead_after_step = [agent for agent in alive_before_step if death_bool[agent]] # new_dead / dead_after_step.

        # Update states globally by copying information from new_state to self.state dict.
        for agent in self.possible_agents:
            # Nothing to do for dead agents...
            if not self.agent_alive[agent]: continue
            # If there is a new agent_death among alive agents...
            if death_bool[agent]: # New agent death.
                self.agent_alive[agent] = False
            # Alive agents which are still alive after this step...
            self.state[agent] = new_state[agent]

        blue_tag = np.sum([ tag_bool[agent] for agent in self.blue_team_agents if agent in alive_before_step ])
        red_tag = np.sum([ tag_bool[agent] for agent in self.red_team_agents if agent in alive_before_step ])
        
        if self.verbose:
            print("=============================")
            for agent in self.blue_team_agents:
                if agent in alive_before_step: print("agent: {}, blue tag check : {}".format(agent, tag_bool[agent]))
                else: continue
            for agent in self.red_team_agents:
                if agent in alive_before_step: print("agent: {}, red tag check : {}".format(agent, tag_bool[agent]))
                else: continue
            print("-----------------------------")
            for agent in self.blue_team_agents:
                if agent in alive_before_step: print("agent: {}, blue respawn check : {}".format(agent, respawn_bool[agent]))
                else: continue
            for agent in self.red_team_agents:
                if agent in alive_before_step: print("agent: {}, red respawn check : {}".format(agent, respawn_bool[agent]))
                else: continue
            print("-----------------------------")
            for agent in self.blue_team_agents:
                if agent in alive_before_step: print("agent: {}, blue death check : {}".format(agent, death_bool[agent]))
                else: continue
            for agent in self.red_team_agents:
                if agent in alive_before_step: print("agent: {}, red death check : {}".format(agent, death_bool[agent]))
                else: continue
            print("=============================")
            for agent in self.blue_team_agents:
                if agent in alive_before_step: print("agent: {}, blue reach_goal check : {}".format(agent, reach_goal_bool[agent]))
                else: continue
            for agent in self.red_team_agents:
                if agent in alive_before_step: print("agent: {}, red reach_goal check : {}".format(agent, reach_goal_bool[agent]))
                else: continue
            print("=============================")
            print("respawn_info_state after tagging: {}".format(self.respawn_info_state))
            print("=============================")

        blue_reach_goal = np.sum([ reach_goal_bool[agent] for agent in self.blue_team_agents if agent in alive_before_step ])
        red_reach_goal = np.sum([ reach_goal_bool[agent] for agent in self.red_team_agents if agent in alive_before_step ])

        if self.defense_team == 'Red': flag_capture_bool = blue_reach_goal
        elif self.defense_team == 'Blue': flag_capture_bool = red_reach_goal

        if not flag_capture_bool:
            # Flag not captured.
            for agent in self.possible_agents:
                # Set terminations dict.
                if not self.agent_alive[agent]:
                    # This sets the terminations dict for all agents not alive: including the prev dead agents and the newly dead agent in this step...
                    terminations[agent] = True
                    continue
                terminations[agent] = False
            # Assign tagging rewards if any.
            for agent in self.blue_team_agents:
                # Set tagging rewards.
                if not self.agent_alive[agent]: continue
                rewards[agent] +=  - blue_tag * (self.T)
                rewards[agent] += + red_tag * (self.T)
            for agent in self.red_team_agents:
                if not self.agent_alive[agent]: continue
                rewards[agent] += + blue_tag * (self.T)
                rewards[agent] += - red_tag * (self.T)
        else:
            # Flag captured.
            for agent in self.possible_agents:
                if not self.agent_alive[agent]:
                    terminations[agent] = True
                    continue
                terminations[agent] = True
            for agent in self.possible_agents:
                if agent[0] == self.defense_team[0]:
                    # Agent in defense team.
                    if self.agent_alive[agent]: rewards[agent] += - self.F
                    infos[agent]["Win"] = False
                    pass
                else:
                    # Agent in away team.
                    if self.agent_alive[agent]: rewards[agent] += self.F
                    infos[agent]["Win"] = True
                    pass
            self.flag_capture_team = self.away_team

        obs = self._observations()

        self.current_step += 1
        if self.current_step == self.max_num_cycles:
            truncations = {agent: True for agent in self.possible_agents if self.agent_alive[agent]}
            # Assign reward to defending team for defending the flag (since the episode terminated without flag capture so successful defense).
            for agent in self.possible_agents:
                if agent[0] == self.defense_team[0]:
                    # Agent in defense team.
                    if self.agent_alive[agent]: rewards[agent] += self.F
                    infos[agent]["Win"] = True
                    pass
                else:
                    # Agent in away team.
                    if self.agent_alive[agent]: rewards[agent] += - self.F
                    infos[agent]["Win"] = False
                    pass
            self.flag_capture_team = None
            pass

        agents_to_remove = []
        for agent in alive_before_step:
            if death_bool[agent]: agents_to_remove.append(agent)
        
        for agent in self.possible_agents:
            if not self.agent_alive[agent]: continue #This skips the newly dead agents too.
            if terminations[agent] or truncations[agent]:
                agents_to_remove.append(agent)

        # Remove done agents
        for agent in agents_to_remove:
            self.agents.remove(agent)
        if self.verbose:
            print("Updated self.agents state: {}".format(self.state))
            print("rewards: {}".format(rewards))
            print("terminations: {}".format(terminations))
            print("truncations: {}".format(truncations))
        return obs, rewards, terminations, truncations, infos

    def global_state_to_agent_local_obs(self, agent):
        """
        Hard-coded for a 2v2 scenario.
        For more agents == GNN.
        ### For the second iteration of the PSRO loop: CTDE? Fixing one team behaviour, have a CTDE scheme for the other team.
        """
        """
        Idea: Permutation invariance for better learning: ordering of adversaries shouldn't matter. Same strategy.
        """
        # coordinate of agent, heading, home_flag_rel_location, away_flag_rel_location, friend_rel_location, friend_rel_pose, adversary_1_rel_location, adversary_1_rel_pose, adversary_2_rel_location, adversary_2_rel_pose
        # 3 = self coordinate, 2 = home flag location, 2 = away flag location, 3 = friend rel pos, 3 = adversary 1 rel pos, 3 = adversary 2 rel pos
        assert self.agent_alive[agent] is True
        agent_type = agent[0]

        blue_team_agents = copy.deepcopy(self.blue_team_agents)
        red_team_agents = copy.deepcopy(self.red_team_agents)

        if agent_type == "B":
            blue_team_agents.remove(agent)
            ally_agent = blue_team_agents[0] #Hard-coded for 2v2 scenario.
            adv_agents = red_team_agents
        if agent_type == "R":
            red_team_agents.remove(agent)
            ally_agent = red_team_agents[0] #Hard-coded for 2v2 scenario.
            adv_agents = blue_team_agents

        obs = np.zeros(16, dtype=np.int8)
        #obs.append(self.state[agent])
        obs[:3] = self.state[agent]
        if agent[0] == "R":
            obs[3:5] = np.array(self.red_flag_location) - np.array(self.state[agent][:2]) # HOME FLAG.
            obs[5:7] = np.array(self.blue_flag_location) - np.array(self.state[agent][:2]) # AWAY FLAG.
        elif agent[0] == "B":
            obs[3:5] = np.array(self.blue_flag_location) - np.array(self.state[agent][:2]) # HOME FLAG.
            obs[5:7] = np.array(self.red_flag_location) - np.array(self.state[agent][:2]) # AWAY FLAG.

        if not self.agent_alive[ally_agent]:
            obs[7:9] = np.array([-100, -100]) #np.array([-np.inf, -np.inf])
            obs[9] = -5 #-np.inf
        else:
            obs[7:9] = np.array(self.state[ally_agent][:2]) - np.array(self.state[agent][:2]) # Friend rel xy position
            obs[9] = (self.state[ally_agent][2] - self.state[agent][2]) % (self.num_headings) # Friend rel heading

        if not self.agent_alive[adv_agents[0]]:
            obs[10:12] = np.array([-100, -100])
            obs[12] = -5
        else:
            obs[10:12] = np.array(self.state[adv_agents[0]][:2]) - np.array(self.state[agent][:2]) # Adversary 1 rel xy position
            obs[12] = (self.state[adv_agents[0]][2] - self.state[agent][2]) % (self.num_headings) # Adversary 1 rel heading

        if not self.agent_alive[adv_agents[1]]:
            obs[13:15] = np.array([-100, -100])
            obs[15] = -5
        else:
            obs[13:15] = np.array(self.state[adv_agents[1]][:2]) - np.array(self.state[agent][:2]) # Adversary 2 rel xy position
            obs[15] = (self.state[adv_agents[1]][2] - self.state[agent][2]) % (self.num_headings) # Adversary 2 rel heading
        
        """
        # Archived on Wimbledon Final day, July 13 @ 06.30 pm.
        obs_mask = np.ones(16, dtype=np.bool)
        if self.obs_mode == 'hetero':
            if agent[0] == 'R': pass
            elif agent[0] == 'B': obs_mask[5:7] = np.array([False, False])
        elif self.obs_mode == 'homo_blue':
            obs_mask[5:7] = np.array([False, False])
        elif self.obs_mode == 'homo_red':
            pass
        obs = obs[obs_mask]
        """
        if self.obs_mode == 'hetero' and agent[0] == 'B':
            obs[5:7] = np.array([-100, -100], dtype=np.int8)  # Mask away flag
        elif self.obs_mode == 'homo_blue':
            obs[5:7] = np.array([-100, -100], dtype=np.int8)  # Mask away flag
        return obs

    def _observations(self):
        # Implements the action_masks from the global state.
        obs = {}
        dummy_obs = np.zeros(self.observation_space(self.possible_agents[0])["observation"].shape, dtype=np.int8) #np.zeros(self.observation_space(self.possible_agents[0]).shape, dtype=np.int8) #np.zeros(self.observation_space(agent).shape, dtype=np.float32)
        for agent in self.possible_agents:
            ##agent_obs = global_state # Had to comment due to some API glue-ing error that Google Colab pointed out.
            #agent_obs = np.concatenate([s for s in global_state]).astype(np.int8)

            if self.agent_alive[agent]: agent_obs = self.global_state_to_agent_local_obs(agent=agent) #v2 observations for better policy sharing.
            else: agent_obs = dummy_obs

            valid = self.valid_actions(global_state_dict=self.state, agent=agent)
            action_mask = np.zeros(self.num_actions, dtype=np.int8)
            action_mask[valid] = 1
            obs[agent] = { "observation": agent_obs, "action_mask": action_mask }
        return obs

    def dynamics(self, agent, actions):
        # Single agent dynamics.
        ### Action keys:
        ## 0 = same pos, same heading
        ## 1 = same pos, minus heading
        ## 2 = same pos, plus heading
        ## 3 = advance pos, same heading
        ## 4 = tagging action

        assert type(agent) == str
        assert self.agent_alive[agent] is True, "agent: {}, self.agent_alive[agent]: {}".format(agent, self.agent_alive[agent])

        agent_team = "Red" if agent[0] == "R" else "Blue"
        #tag_bool, reach_goal_bool = False, False
        tag_bool, respawn_bool, death_bool, reach_goal_bool = False, False, False, False

        x, y, heading = self.state[agent]
        action = actions[agent]
        x_, y_, heading_ = None, None, None

        # Check if there are incoming tagging actions from nearby agents.
        potential_incoming_tagging_agents = []
        incoming_tagging_agents = []
        for neighbouring_agent in self.possible_agents:
            if not self.agent_alive[neighbouring_agent]: continue
            if neighbouring_agent == agent: continue
            else:
                x_neighbour, y_neighbour, heading_neighbour = self.state[neighbouring_agent]
                if abs(x - x_neighbour) <= 1 and abs(y - y_neighbour) <= 1:
                    neighbour_point_to_x, neighbour_point_to_y =  self.state[neighbouring_agent][:2] + self._heading_to_direction_vector(heading=heading_neighbour)
                    if (neighbour_point_to_x, neighbour_point_to_y) == (x, y): potential_incoming_tagging_agents.append(neighbouring_agent)
                    else: continue

        for agent_iter in potential_incoming_tagging_agents:
            if actions[agent_iter] == 4: incoming_tagging_agents.append(agent_iter)
            else: continue
        
        if len(incoming_tagging_agents) == 0:
            # No incoming tags.
            if action == 0:
                # same pos, same heading
                x_, y_, heading_ = x, y, heading
                pass
            elif action == 1:
                # same pos, minus heading
                x_, y_, heading_ = x, y, (heading - 1) % self.num_headings
                pass
            elif action == 2:
                # same pos, plus heading
                x_, y_, heading_ = x, y, (heading + 1) % self.num_headings
                pass
            elif action == 3:
                # advance pos, same heading
                x_, y_, heading_ = x, y, heading
                if heading == 0:
                    x_, y_ = x + 1, y
                elif heading == 1:
                    x_, y_ = x + 1, y + 1
                elif heading == 2:
                    x_, y_ = x, y + 1
                elif heading == 3:
                    x_, y_ = x - 1, y + 1
                elif heading == 4:
                    x_, y_ = x - 1, y
                elif heading == 5:
                    x_, y_ = x - 1, y - 1
                elif heading == 6:
                    x_, y_ = x, y - 1
                elif heading == 7:
                    x_, y_ = x + 1, y - 1
            elif action == 4: # Tagging action.
                x_, y_, heading_ = x, y, heading
        else:
            # Agent tagged.
            ### LOGIC: if respawn_allowed: respawn. Else: agent_death. have respawn_bool and death_bool as sub-categories of tag_bool.
            tag_bool = True
            if self.verbose: print(".............. TAG TAG TAG TAG TAG TAG TAG TAG TAG TAG TAG ..............")
            #respawn_allowed = self.respawn_info_state[agent_team] < self.max_respawns
            agents_on_agent_team = [agent for agent in self.possible_agents if agent[0] == agent_team[0]]
            team_tags = sum([self.respawn_info_state[agent] for agent in agents_on_agent_team])
            respawn_allowed = team_tags < self.max_respawns
            if self.verbose:
                print("agent: {}".format(agent))
                print("agent_team: {}".format(agent_team))
                print("agents on agent team: {}".format(agents_on_agent_team))
                print("respawn_info_state: {}".format([self.respawn_info_state[agent] for agent in agents_on_agent_team]))
                print("team_tags: {}".format(team_tags))
                print("self.max_respawns: {}".format(self.max_respawns))
                print("respawn_allowed if team_tags < self.max_respawns: {}".format(respawn_allowed))

            if respawn_allowed:
                # Incoming tags. Spawn in the tagging area.
                # Multiple tags = single penalty.
                self.respawn_info_state[agent] += 1
                respawn_bool = True
                """
                Spawning area is a rectangle, away from the flag in a corner.
                blue_tag_spawn_area = {"x_lim":(5, 7), "y_lim": (0, 1)} # (x, y) coordinate tuple -- x_low y_low and x_high y_high
                red_tag_spawn_area = {"x_lim":(0, 2), "y_lim": (6, 7)} # (x, y) coordinate tuple -- x_low y_low and x_high y_high
                """
                if agent_team == "Red":
                    spawn_area = self.red_tag_spawn_area
                    possible_spawn_headings = [0, 7, 6, 5] # [0, 7, 6, 5], possible init headings = (0, -45, -90, -135)
                elif agent_team == "Blue":
                    spawn_area = self.blue_tag_spawn_area
                    possible_spawn_headings = [0, 1, 2, 3] # possible init headings = (0, 45, 90, 135)
                x_prime = self.np_random.integers( spawn_area["x_lim"][0], spawn_area["x_lim"][1] + 1)
                y_prime = self.np_random.integers( spawn_area["y_lim"][0], spawn_area["y_lim"][1] + 1)

                if agent_team == "Red" and x_prime == 0:
                    try: possible_spawn_headings.remove(5)
                    except: pass
                elif agent_team == "Red" and x_prime == self.grid_size - 1:
                    try: possible_spawn_headings.remove(0)
                    except: pass
                    try: possible_spawn_headings.remove(7)
                    except: pass
                
                if agent_team == "Blue" and x_prime == 0:
                    try: possible_spawn_headings.remove(3)
                    except: pass
                elif agent_team == "Blue" and x_prime == self.grid_size - 1:
                    try: possible_spawn_headings.remove(0)
                    except: pass
                    try: possible_spawn_headings.remove(1)
                    except: pass

                heading_prime = self.np_random.choice(possible_spawn_headings)
                x_, y_, heading_ = x_prime, y_prime, heading_prime
            else:
                # Agent tagged but respawn not allowed hence agent_death.
                self.respawn_info_state[agent] += 1
                death_bool = True

            print("respawn_info_state after updating: {}".format([self.respawn_info_state[agent] for agent in agents_on_agent_team]))
        
        if agent_team is self.away_team and (x_, y_) == self.flag_location:
            reach_goal_bool = True

        new_state = (x_, y_, heading_)
        return tag_bool, respawn_bool, death_bool, reach_goal_bool, new_state

    def _heading_to_direction_vector(self, heading):
        assert heading in range(self.num_headings)
        vec = None
        if heading == 0:
            vec = np.array([ +1, 0 ])
        elif heading == 1:
            vec = np.array([ +1, +1 ])
        elif heading == 2:
            vec = np.array([ 0, +1 ])
        elif heading == 3:
            vec = np.array([ -1, +1 ])
        elif heading == 4:
            vec = np.array([ -1, 0 ])
        elif heading == 5:
            vec = np.array([ -1, -1 ])
        elif heading == 6:
            vec = np.array([ 0, -1 ])
        elif heading == 7:
            vec = np.array([ +1, -1 ])
        return vec

    def valid_actions(self, global_state_dict, agent):
        """
        ### Action keys:
        ## 0 = same pos, same heading
        ## 1 = same pos, minus heading
        ## 2 = same pos, plus heading
        ## 3 = advance pos, same heading
        ## 4 = tagging action
        """

        # NOTE: No collision logic in v0.
        # Takes in as input global_state of the world and the agent name and returns valid actions for the agent in the global_state of the world.
        # Utility: to decide action_masking for the agent in the global_state.
        assert agent in self.possible_agents
        if not self.agent_alive[agent]: return [0]

        agent_state = copy.deepcopy(global_state_dict[agent])
        # Action masking due to walls of the environment.
        x, y, heading = agent_state
        point_to_x, point_to_y =  agent_state[:2] + self._heading_to_direction_vector(heading=heading)

        valid_actions = [0, 1, 2, 3]
        invalid_action_x = None
        invalid_action_y = None

        if x == 0 and heading == 2:
            invalid_action_x = 2
        if x == 0 and heading == 6:
            invalid_action_x = 1
        if x == 0 and heading in [3, 4, 5]:
          invalid_action_x = 3
        if x == self.grid_size - 1 and heading == 2:
            invalid_action_x = 1
        if x == self.grid_size - 1 and heading == 6:
            invalid_action_x = 2
        if x == self.grid_size - 1 and heading in [0, 1, 7]:
          invalid_action_x = 3
        try:
            valid_actions.remove(invalid_action_x)
        except: pass

        if y == 0 and heading == 0:
            invalid_action_y = 1
        if y == 0 and heading == 4:
            invalid_action_y = 2
        if y == 0 and heading in [5, 6, 7]:
            invalid_action_y = 3
        if y == self.grid_size - 1 and heading == 0:
            invalid_action_y = 2
        if y == self.grid_size - 1 and heading == 4:
            invalid_action_y = 1
        if y == self.grid_size - 1 and heading in [1, 2, 3]:
            invalid_action_y = 3
        try:
            valid_actions.remove(invalid_action_y)
        except: pass

        # Asking masking on the Tagging space due to nearby agents.
        # Logic to see nearby agents position-wise: once a list of nearby agents is generated: check if heading is aligned with the neighbouring agent for potential tagging action.
        target_tagging_agents = []
        for potential_nearby_agent in self.possible_agents:
            if not self.agent_alive[potential_nearby_agent]: continue
            if potential_nearby_agent[0] == agent[0]: continue # Cannot tag same team member. HAD A BUG HERE previously. Corrected now.
            else:
                x_neighbour, y_neighbour, heading_neighbour = global_state_dict[potential_nearby_agent]
                if abs(x - x_neighbour) <= 1 and abs(y - y_neighbour) <= 1:
                    if (x_neighbour, y_neighbour) == (point_to_x, point_to_y): target_tagging_agents.append(potential_nearby_agent)
                    else: continue
                pass
            pass
        if len(target_tagging_agents) > 0: valid_actions.append(4) #Tagging action available.

        return valid_actions

    def state_space(self, agent):
        # The state of the agent on the grid: (x, y, theta).
        return MultiDiscrete([self.grid_size, self.grid_size, self.num_headings])

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Each agent can observe the state of all the agents.
        # obs_type = MultiDiscrete([self.grid_size, self.grid_size, self.num_headings, self.grid_size, self.grid_size, self.num_headings, self.grid_size, self.grid_size, self.num_headings, self.grid_size, self.grid_size, self.num_headings]) #Tuple([ self.state_space(agent) for agent in self.agents ])
        # coordinate of agent, heading, home_flag_rel_location, away_flag_rel_location, friend_rel_location, friend_rel_pose, adversary_1_rel_location, adversary_1_rel_pose, adversary_2_rel_location, adversary_2_rel_pose
        # 3 = self coordinate, 2 = home flag location, 2 = away flag location, 3 = friend rel pos, 3 = adversary 1 rel pos, 3 = adversary 2 rel pos
        assert agent in self.possible_agents
        obs_type_full = Box(low=0, high=self.grid_size, shape=(2 + 2 + len(self.possible_agents) * 3,), dtype=np.int8) # Away flag is visible.
        obs_type_partial = Box(low=0, high=self.grid_size, shape=(2 + 0 + len(self.possible_agents) * 3,), dtype=np.int8) # Away flag not visible.
        """
        # Archiving the following on July 13 @ 06.20 pm.
        if self.obs_mode == 'hetero':
            if agent[0] == 'R': obs_type = obs_type_full
            elif agent[0] == 'B': obs_type = obs_type_partial
        elif self.obs_mode == 'homo_blue':
            obs_type = obs_type_partial
        elif self.obs_mode == 'homo_red':
            obs_type = obs_type_full
        """
        obs_type = obs_type_full # Added on July 13 @ 06.20 pm. And add -1 observations to occlude away information.
        obs_space_type = Dict({"observation": obs_type, "action_mask": Box(low=0, high=1, shape=(self.num_actions,), dtype=np.int8)})
        return obs_space_type

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        assert agent in self.possible_agents
        #Simple dynamics. #same heading same place, heading plus, heading minus, move forward. And a tagging action.
        return Discrete(self.num_actions)

    def num_agents(self):
        return self.num_teams * self.num_agents_blue_team

    def render(self, history_traj=None):
        # Note: Once you have the state and the flag locations, this module can be written disjoint of any other logic in the whole program.
        blue_agent_pos = [self.state[agent] for agent in self.blue_team_agents if self.agent_alive[agent]]
        red_agent_pos = [self.state[agent] for agent in self.red_team_agents if self.agent_alive[agent]]
        if self.away_team == 'Red':
            fig, ax = draw_grid(grid_size=self.grid_size, blue_agent_pos=blue_agent_pos, red_agent_pos=red_agent_pos, blue_flag_loc=self.blue_flag_location, red_flag_loc=None, blue_flag_img_path=self.blue_flag_img_path, red_flag_img_path=self.red_flag_img_path)
        else:
            fig, ax = draw_grid(grid_size=self.grid_size, blue_agent_pos=blue_agent_pos, red_agent_pos=red_agent_pos, blue_flag_loc=None, red_flag_loc=self.red_flag_location, blue_flag_img_path=self.blue_flag_img_path, red_flag_img_path=self.red_flag_img_path)
        plt.show()
        return fig, ax

class DefenseCTF_v2(ParallelEnv):
    """
    Created on July 14, 2025 (Monday). Description: handle agent_deaths without editing self.agents mid-episode.
    #1 Start with homogenous observation spaces for all agents.
    #2 Extend to heterogenous observation spaces.

    Idea: if a team is tagged by more than max_respawns, end episode and the other team loses == easier to implement.
    Implement: re-create heterogenous observation space. Episode ends if a team reaches max_respawns limit and the opponent team wins ala successful defense.
    """

    """
    Summary of DefenseCTF_v2 environment, added on July 14 2025: All agents are terminated when all of one team is tagged / episode is over or the flag is captured. Truncated when length of episode goes over max_allowed.
    """
    """
    Summary of observation space in DefenseCTF_v2, added on Aug 2 2025: Defense_team agents have obs_type = obs_type_partial (only home flag visible), and away_team agents have obs_type = obs_type_full (both home and away flags visible).
    """
    meta_data = { "name": "DefenseCTF"}
    def __init__(self, defense_team, grid_size=8, ctf_player_config="2v2", max_num_cycles=300, red_flag_locs=(6,6), max_respawns=2, tag_reward_shaping=True, reward_shaping={}, verbose=False, obs_mode=None, seed=None, **kwargs):
        assert defense_team in ['Blue', 'Red']
        assert isinstance(reward_shaping, dict)

        self.metadata = { "name": "DefenseCTF"}
        self.render_mode = "human"
        self.verbose = verbose

        if seed is not None:
            self.seed(seed=seed)
        else:
            self._seed = None
            self.np_random = None
            self.torch_rng = None

        self.defense_team = defense_team
        if obs_mode is None:
            """
            if self.defense_team == 'Blue': obs_mode = 'hetero'
            elif self.defense_team == 'Red': obs_mode = 'homo_red'
            self.set_obs_mode(obs_mode=obs_mode)
            """
            self.obs_mode = None # In the observation_space, defense_team is obs_type_partial and away_team is obs_type_full.
        else: self.set_obs_mode(obs_mode=obs_mode)
        if self.defense_team == 'Red': self.away_team = 'Blue'
        else: self.away_team = 'Red'

        """
        Note: Defense_team influences obs_mode. For eg, if blue is defending, then the red team has to have the blue_flag i.e. the away flag as a part of the policy. This means Blue team has obs_mode = homo_blue but Red team has obs_mode = hetero.
        Note: If Red is defending, it has to be trained against a Blue that knows the Red flag location.
        If defense_mode = 'Blue':
            obs_mode of Blue agents = obs_partial
            obs_mode of Red agents = obs_full

        If defense_mode = 'Red'
            obs_mode of Blue agents = obs_full
            obs_mode of Red agents = obs_full
        """
        self.grid_size = grid_size #this defines the state space of the Markov Game and hence the observation_space.
        self.num_headings = 8  # Possible headings in degrees:(0, 45, 90, 135, 180, 225, 270, 315)
        self.num_actions = 5
        self.max_respawns = max_respawns
        self.max_num_cycles = max_num_cycles

        self.num_teams = 2
        if ctf_player_config == "1v1":
            self.num_agents_blue_team = 1
        elif ctf_player_config == "2v2":
            self.num_agents_blue_team = 2
        self.num_agents_red_team = self.num_agents_blue_team
        self.num_agents = self.num_agents()

        self.blue_team_agents = ["Blue_" + "{}".format(i) for i in range(self.num_agents_blue_team)]
        self.red_team_agents = ["Red_" + "{}".format(i) for i in range(self.num_agents_red_team)]

        self.possible_agents = copy.deepcopy(self.blue_team_agents + self.red_team_agents)
        self.agents = copy.deepcopy(self.possible_agents)
        if self.verbose: print("self.agents: {}".format(self.agents))
        if self.verbose: print("self.possible_agents: {}".format(self.possible_agents))
        self.agent_alive = {agent: None for agent in self.possible_agents} # set to True for each agent on resetting.

        self.observation_spaces = {agent: self.observation_space(agent=agent)["observation"] for agent in self.possible_agents}
        self.action_spaces = {agent: self.action_space(agent=agent) for agent in self.possible_agents}

        """
        For v1 of the CustomCTF environment: Can add logic to have a jitter / randomness in the flag locations.
        """
        self.deceptive_env = True #Note: Deception refers to uncertainty in the red_flag_location.
        self.blue_flag_location = kwargs.get('blue_flag_location', (1, 1))
        assert (red_flag_locs is not None) and isinstance(red_flag_locs, tuple)
        self.red_flag_location = red_flag_locs
        self.red_flag_locs = {'locs': [red_flag_locs], 'p': [1.]}
        
        if self.defense_team == 'Red': self.flag_location = self.red_flag_location
        else: self.flag_location = self.blue_flag_location

        self.blue_init_spawn_y_lim = 2 # Measured from bottom of the grid.
        self.possible_init_headings_blue_team = [0, 1, 2, 3] # possible init headings = (0, 45, 90, 135)
        self.red_init_spawn_y_lim = 0 #1 #2 # Measured from top of the grid.
        self.possible_init_headings_red_team = [0, 7, 6, 5] # possible init headings = (0, -45, -90, -135)

        # Tag spawning area is a rectangle, away from the flag in a corner.
        self.blue_tag_spawn_area = {"x_lim":(0, self.grid_size - 1), "y_lim": (0, 1)} # {"x_lim":(0, 7), "y_lim": (0, 1)} # (x, y) coordinate tuple -- x_low y_low and x_high y_high
        self.red_tag_spawn_area = {"x_lim":(0, self.grid_size - 1), "y_lim": (6, 7)} # {"x_lim":(0, 2), "y_lim": (6, 7)} # (x, y) coordinate tuple -- x_low y_low and x_high y_high

        self.current_step = 0
        self.state = { agent: None for agent in self.possible_agents}
        self.respawn_info_state = {agent: 0 for agent in self.possible_agents}

        self.F = 100 #self.capture_flag_reward = 100
        self.tag_reward_shaping = tag_reward_shaping
        self.reward_shaping = reward_shaping
        if self.tag_reward_shaping:
            if 'tag_reward' in self.reward_shaping: self.T = self.reward_shaping['tag_reward']
            else: self.T = 10
        else: self.T = 0

        self.blue_flag_img_path = 'flag_imgs/blue_flag.png'
        self.red_flag_img_path = 'flag_imgs/red_flag.png'

        self.blue_flag_img = mpimg.imread('flag_imgs/blue_flag.png')
        self.red_flag_img = mpimg.imread('flag_imgs/red_flag.png')

        self.flag_capture_team = None

        self._init_args = (defense_team)
        self._init_kwargs = {"grid_size":grid_size, "ctf_player_config":ctf_player_config, "max_num_cycles":max_num_cycles, "red_flag_locs":red_flag_locs, "max_respawns":max_respawns, "tag_reward_shaping":tag_reward_shaping, "reward_shaping":reward_shaping, "verbose":verbose, "obs_mode":obs_mode, "seed":seed}

    def __call__(self):
        return DefenseCTF_v2(*self._init_args, **self._init_kwargs)

    def seed(self, seed=None):
        self.rngs = make_seeded_rngs(seed)
        self._seed = self.rngs["actual_seed"]
        self.np_random = self.rngs["np_random"]
        self.torch_rng = self.rngs["torch_rng"]
        return [self._seed]

    def set_obs_mode(self, obs_mode):
        assert obs_mode in [ 'hetero', 'homo_blue', 'homo_red' ]
        self.obs_mode = obs_mode
        return

    def _sample_init_heading(self, agent_team, x, y):
        assert agent_team == "Blue" or agent_team == "Red"
        if agent_team == "Blue":
            possible_init_headings = copy.deepcopy(self.possible_init_headings_blue_team)
            if x == 0: possible_init_headings.remove(3)
            if x == self.grid_size - 1:
                possible_init_headings.remove(0)
                possible_init_headings.remove(1)
            heading = self.np_random.choice(possible_init_headings)
        elif agent_team == "Red":
            possible_init_headings = copy.deepcopy(self.possible_init_headings_red_team)
            if x == 0: possible_init_headings.remove(5)
            if x == 1:
                possible_init_headings.remove(0)
                possible_init_headings.remove(7)
            heading = self.np_random.choice(possible_init_headings)
        return heading

    def reset(self, seed = None, options = None):
      #### Debugging prints: print valid actions for each agent at the step. => a) actions available at the walls b) tagging action available or not.
        if seed is not None:
            self.seed(seed=seed)

        self.current_step = 0
        self.possible_agents = copy.deepcopy(self.blue_team_agents + self.red_team_agents)
        self.agents = copy.deepcopy(self.possible_agents) # !!! 04/25 3.56 pm: DON'T UNDERSTAND THIS BUG.
        self.agent_alive = {agent: True for agent in self.possible_agents}
        self.respawn_info_state = {agent: 0 for agent in self.possible_agents}

        """
        NOTE: In the current implementation, v0, flag locations are parameters of the Markov game and are hence not a part of the state, agent-wise or global.
        For v1: Can add logic to have a jitter / randomness in the flag locations.
        """
        #self.blue_flag_location = (1, 1)
        #self.red_flag_location = (self.grid_size - 2, self.grid_size - 2) # We assume the grid_size > 3 or something like that for the flag placements to be a non-trivial game scenario.

        self.blue_flag_location = self.blue_flag_location
        if self.deceptive_env:
            red_flag_idx = self.np_random.choice(len(self.red_flag_locs['locs']), p=self.red_flag_locs['p'])
            self.red_flag_location = self.red_flag_locs['locs'][red_flag_idx]
            print("RED FLAG LOCATION: {}".format(self.red_flag_location))

        blue_team_init_xys = []
        for agent_iter, blue_agent_num in enumerate(range(self.num_agents_blue_team)):
            if agent_iter == 0:
                rand_x = self.np_random.integers(0, self.grid_size)
                rand_y = self.np_random.integers(0, self.blue_init_spawn_y_lim + 1)
                rand_theta = self._sample_init_heading(agent_team="Blue", x=rand_x, y=rand_y)
                self.state["Blue_{}".format(blue_agent_num)] = np.array([rand_x, rand_y, rand_theta])
                blue_team_init_xys.append((rand_x, rand_y))
                continue
            # A while loop to break ties in the x-y location to prevent collision during agent initialization.
            while (rand_x, rand_y) in blue_team_init_xys:
                rand_x = self.np_random.integers(0, self.grid_size)
                rand_y = self.np_random.integers(0, self.blue_init_spawn_y_lim + 1)
            blue_team_init_xys.append((rand_x, rand_y))
            rand_theta = self._sample_init_heading(agent_team="Blue", x=rand_x, y=rand_y)
            self.state["Blue_{}".format(blue_agent_num)] = np.array([rand_x, rand_y, rand_theta])

        red_team_init_xys = []
        for agent_iter, red_agent_num in enumerate(range(self.num_agents_red_team)):
            if agent_iter == 0:
                rand_x = self.np_random.integers(0, self.grid_size)
                rand_y = self.np_random.integers(self.grid_size - self.red_init_spawn_y_lim - 1, self.grid_size)
                rand_theta = self._sample_init_heading(agent_team="Red", x=rand_x, y=rand_y)
                self.state["Red_{}".format(red_agent_num)] = np.array([rand_x, rand_y, rand_theta])
                red_team_init_xys.append((rand_x, rand_y))
                continue
            # A while loop to break ties in the x-y location to prevent collision during agent initialization.
            while (rand_x, rand_y) in red_team_init_xys:
                rand_x = self.np_random.integers(0, self.grid_size)
                rand_y = self.np_random.integers(self.grid_size - self.red_init_spawn_y_lim - 1, self.grid_size)
            red_team_init_xys.append((rand_x, rand_y))
            rand_theta = self._sample_init_heading(agent_team="Red", x=rand_x, y=rand_y)
            self.state["Red_{}".format(red_agent_num)] = np.array([rand_x, rand_y, rand_theta])

        #global_state = [self.state[agent] for agent in self.agents]
        #obs = {agent: global_state for agent in self.agents} # global_state dict. The prisoner guard example on ParallelEnv has this as a tuple, and you have this as a List currently.
        obs = self._observations()
        info = {agent: {} for agent in self.possible_agents}
        #print("self.agents: {}".format(self.agents))
        return obs, info

    def step(self, actions):
        """
        Comment added on July 14 2025: episode terminates as soon as any agent dies on any team.
        Termination condition: if the flag is captured, or all agents dead on a team (home or away team). Task: do this without removing agents from self.agents.
        Truncation condition: if time exceeds max time-steps.
        """
        """
        Comment: Reward sharing breaks between agents on the same team because of potential agent death.
        """
      #### Debugging prints: print valid actions for each agent at the step. => a) actions available at the walls b) tagging action available or not.
        ### v0: No collision logic.
        # TO-DO for v1: Resolve collisions and tagging and re-spawning. # Collisions within a team prevented via action_masking. Where to write actions_available_to_an_agent?

        # The input actions is a Dict from agent_name to actions. Add an assert statement for Type of actions.
        """
        Action keys:
        0 = same pos, same heading
        1 = same pos, minus heading
        2 = same pos, plus heading
        3 = advance pos, same heading
        4 = tagging action
        """
        state_before_step = copy.deepcopy(self.state)

        if self.verbose: print("Current Step: {}".format(self.current_step))
        if self.verbose: print("Before stepping self.agents state: {}".format(self.state))
        if self.verbose: print("Actions: {}".format(actions))

        infos = {agent: {} for agent in self.possible_agents}
        for agent in self.possible_agents:
            infos[agent]["Win"] = None # "Win" descriptor is for the last step of the game on which the winning team is announced if any!
        
        tag_bool = {agent: False for agent in self.possible_agents if self.agent_alive[agent]}
        respawn_bool = {agent: False for agent in self.possible_agents if self.agent_alive[agent]}
        death_bool = {agent: False for agent in self.possible_agents if self.agent_alive[agent]}
        reach_goal_bool = {agent: False for agent in self.possible_agents if self.agent_alive[agent]}
        new_state = {agent: None for agent in self.possible_agents}
        
        rewards = {agent: 0. for agent in self.possible_agents}
        terminations = {agent: False for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}
        
        alive_before_step = [agent for agent in self.possible_agents if self.agent_alive[agent]]
        # Gather new states after applying actions in the new_state dict from the following for-loop.
        for agent in self.possible_agents:
            agent_team = 'Red' if agent[0] == 'R' else 'Blue'
            if not self.agent_alive[agent]: continue
            # Agent alive.
            assert actions[agent] in self.action_space(agent=agent), "agent = {}, actions[agent] = {} in self.action_space(agent=agent) = {}".format(agent, actions[agent], self.action_space(agent=agent))
            assert actions[agent] in self.valid_actions(state_before_step, agent), "agent = {}, actions[agent] = {} in self.valid_actions(self.state, agent) = {}".format(agent, actions[agent], self.valid_actions(self.state, agent))
            
            tag_bool[agent], respawn_bool[agent], death_bool[agent], reach_goal_bool[agent], new_state[agent] = self.dynamics(agent=agent, actions=actions)
            #print("agent: {}, tag_bool_agent: {}".format(agent, tag_bool[agent]))

        new_dead_after_step = [agent for agent in alive_before_step if death_bool[agent]] # new_dead / dead_after_step.

        # Update states globally by copying information from new_state to self.state dict.
        for agent in self.possible_agents:
            # Nothing to do for dead agents...
            if not self.agent_alive[agent]: continue
            # If there is a new agent_death among alive agents...
            if death_bool[agent]: # New agent death.
                self.agent_alive[agent] = False
            # Alive agents which are still alive after this step...
            self.state[agent] = new_state[agent]

        blue_tag = np.sum([ tag_bool[agent] for agent in self.blue_team_agents if agent in alive_before_step ])
        red_tag = np.sum([ tag_bool[agent] for agent in self.red_team_agents if agent in alive_before_step ])
        
        if self.verbose:
            print("=============================")
            for agent in self.blue_team_agents:
                if agent in alive_before_step: print("agent: {}, blue tag check : {}".format(agent, tag_bool[agent]))
                else: continue
            for agent in self.red_team_agents:
                if agent in alive_before_step: print("agent: {}, red tag check : {}".format(agent, tag_bool[agent]))
                else: continue
            print("-----------------------------")
            for agent in self.blue_team_agents:
                if agent in alive_before_step: print("agent: {}, blue respawn check : {}".format(agent, respawn_bool[agent]))
                else: continue
            for agent in self.red_team_agents:
                if agent in alive_before_step: print("agent: {}, red respawn check : {}".format(agent, respawn_bool[agent]))
                else: continue
            print("-----------------------------")
            for agent in self.blue_team_agents:
                if agent in alive_before_step: print("agent: {}, blue death check : {}".format(agent, death_bool[agent]))
                else: continue
            for agent in self.red_team_agents:
                if agent in alive_before_step: print("agent: {}, red death check : {}".format(agent, death_bool[agent]))
                else: continue
            print("=============================")
            for agent in self.blue_team_agents:
                if agent in alive_before_step: print("agent: {}, blue reach_goal check : {}".format(agent, reach_goal_bool[agent]))
                else: continue
            for agent in self.red_team_agents:
                if agent in alive_before_step: print("agent: {}, red reach_goal check : {}".format(agent, reach_goal_bool[agent]))
                else: continue
            print("=============================")
            print("respawn_info_state after tagging: {}".format(self.respawn_info_state))
            print("=============================")

        blue_reach_goal = np.sum([ reach_goal_bool[agent] for agent in self.blue_team_agents if agent in alive_before_step ])
        red_reach_goal = np.sum([ reach_goal_bool[agent] for agent in self.red_team_agents if agent in alive_before_step ])

        if self.defense_team == 'Red': flag_capture_bool = blue_reach_goal
        elif self.defense_team == 'Blue': flag_capture_bool = red_reach_goal

        if not flag_capture_bool:
            # Flag not captured. Termination only when all the agents in a team (either of the teams) are dead.
            """
            Pending: break down into cases. If an agent is dead, just mark that as dead in terminations. If all agents on a team dead: terminate episode by marking all agents as terminated. Otherwise truncation via max_num_timesteps.
            Pending: Re-think observation spaces. First get the training running on a homogenous behaviour here. Then run it for heterogenous observation spaces.
            """
            """
            Latest: termination only when all agents dead in a team: enforce by saying tag_num > max_respawns + something.
            """
            alive_away_agents = [agent for agent in self.possible_agents if agent[0] == self.away_team[0] and self.agent_alive[agent]]
            alive_defense_agents = [agent for agent in self.possible_agents if agent[0] == self.defense_team[0] and self.agent_alive[agent]]
            # Case-1: Flag is not captured, but all agents on either team are dead. So episode terminates.
            if alive_away_agents == [] or alive_defense_agents == []:
                # Set terminations dict.
                for agent in self.possible_agents:
                    terminations[agent] = True
                    """
                    # Commented on July 14 2025 @ 10.57 pm.
                    if not self.agent_alive[agent]:
                        # This sets the terminations dict for all agents not alive: including the prev dead agents and the newly dead agent in this step...
                        terminations[agent] = True
                        continue
                    """
                # Case-1A: Add reward and info_tag information. If all of away_team agents are tagged, the defense_team gets the mission completion positive reward and away_team gets negative mission completion reward.
                if alive_away_agents == []:
                    for agent in self.possible_agents:
                        if agent[0] == self.away_team[0]:
                            rewards[agent] += -self.F
                            infos[agent]["Win"] = False
                        else:
                            rewards[agent] += self.F
                            infos[agent]["Win"] = True
                # Case-1B: Add reward and info_tag information. If all of defense_team agents are tagged, the away_team gets the mission completion positive reward and defense_team gets negative mission completion reward.
                else:
                    # Case when the set alive_defense_agents is null / empty.
                    for agent in self.possible_agents:
                        if agent[0] == self.away_team[0]:
                            rewards[agent] += self.F
                            infos[agent]["Win"] = True
                        else:
                            rewards[agent] += -self.F
                            infos[agent]["Win"] = False
            # Case-2: Flag is not captured, and none of the team is eliminated, therefore episode goes on.
            else:
                # Assign tagging rewards if any.
                for agent in self.blue_team_agents:
                    # Set tagging rewards.
                    if not self.agent_alive[agent]: continue
                    rewards[agent] +=  - blue_tag * (self.T)
                    rewards[agent] += + red_tag * (self.T)
                for agent in self.red_team_agents:
                    if not self.agent_alive[agent]: continue
                    rewards[agent] += + blue_tag * (self.T)
                    rewards[agent] += - red_tag * (self.T)
        else:
            # Flag captured.
            for agent in self.possible_agents:
                if not self.agent_alive[agent]:
                    terminations[agent] = True
                    continue
                terminations[agent] = True
            for agent in self.possible_agents:
                if agent[0] == self.defense_team[0]:
                    # Agent in defense team.
                    if self.agent_alive[agent]: rewards[agent] += - self.F
                    infos[agent]["Win"] = False
                    pass
                else:
                    # Agent in away team.
                    if self.agent_alive[agent]: rewards[agent] += self.F
                    infos[agent]["Win"] = True
                    pass
            self.flag_capture_team = self.away_team

        obs = self._observations()

        self.current_step += 1
        if self.current_step == self.max_num_cycles:
            for agent in self.possible_agents: truncations[agent] = True
            # Assign reward to defending team for defending the flag (since the episode terminated without flag capture so successful defense).
            for agent in self.possible_agents:
                if agent[0] == self.defense_team[0]:
                    # Agent in defense team.
                    if self.agent_alive[agent]: rewards[agent] += self.F
                    infos[agent]["Win"] = True
                    pass
                else:
                    # Agent in away team.
                    if self.agent_alive[agent]: rewards[agent] += - self.F
                    infos[agent]["Win"] = False
                    pass
            self.flag_capture_team = None
            pass

        agents_to_remove = []
        """
        # Commented on Tuesday, July 15 2025 @ 11.31 am. No agent_deaths mid-episode.
        for agent in alive_before_step:
            if death_bool[agent]: agents_to_remove.append(agent)
        """

        for agent in self.possible_agents:
            #if not self.agent_alive[agent]: continue #This skips the newly dead agents too. # Commented on Tuesday July 15 2025 @ 11.31 am. All agents terminate / truncate together. For compatibility with SB3 PPO training.
            if terminations[agent] or truncations[agent]:
                agents_to_remove.append(agent)

        # Remove done agents
        for agent in agents_to_remove:
            self.agents.remove(agent)
        if self.verbose:
            print("Updated self.agents state: {}".format(self.state))
            print("rewards: {}".format(rewards))
            print("terminations: {}".format(terminations))
            print("truncations: {}".format(truncations))
        return obs, rewards, terminations, truncations, infos

    def global_state_to_agent_local_obs(self, agent):
        """
        Hard-coded for a 2v2 scenario.
        For more agents == GNN.
        ### For the second iteration of the PSRO loop: CTDE? Fixing one team behaviour, have a CTDE scheme for the other team.
        """
        """
        Idea: Permutation invariance for better learning: ordering of adversaries shouldn't matter. Same strategy.
        """
        # coordinate of agent, heading, home_flag_rel_location, away_flag_rel_location, friend_rel_location, friend_rel_pose, adversary_1_rel_location, adversary_1_rel_pose, adversary_2_rel_location, adversary_2_rel_pose
        # 3 = self coordinate, 2 = home flag location, 2 = away flag location, 3 = friend rel pos, 3 = adversary 1 rel pos, 3 = adversary 2 rel pos
        assert self.agent_alive[agent] is True
        agent_type = agent[0]

        blue_team_agents = copy.deepcopy(self.blue_team_agents)
        red_team_agents = copy.deepcopy(self.red_team_agents)

        if agent_type == "B":
            blue_team_agents.remove(agent)
            ally_agent = blue_team_agents[0] #Hard-coded for 2v2 scenario.
            adv_agents = red_team_agents
        if agent_type == "R":
            red_team_agents.remove(agent)
            ally_agent = red_team_agents[0] #Hard-coded for 2v2 scenario.
            adv_agents = blue_team_agents

        obs = np.zeros(16, dtype=np.int8)
        #obs.append(self.state[agent])
        obs[:3] = self.state[agent]
        if agent[0] == "R":
            obs[3:5] = np.array(self.red_flag_location) - np.array(self.state[agent][:2]) # HOME FLAG.
            obs[5:7] = np.array(self.blue_flag_location) - np.array(self.state[agent][:2]) # AWAY FLAG.
        elif agent[0] == "B":
            obs[3:5] = np.array(self.blue_flag_location) - np.array(self.state[agent][:2]) # HOME FLAG.
            obs[5:7] = np.array(self.red_flag_location) - np.array(self.state[agent][:2]) # AWAY FLAG.

        if not self.agent_alive[ally_agent]:
            obs[7:9] = np.array([-100, -100]) #np.array([-np.inf, -np.inf])
            obs[9] = -100 #-5 #-np.inf
        else:
            obs[7:9] = np.array(self.state[ally_agent][:2]) - np.array(self.state[agent][:2]) # Friend rel xy position
            obs[9] = (self.state[ally_agent][2] - self.state[agent][2]) % (self.num_headings) # Friend rel heading

        if not self.agent_alive[adv_agents[0]]:
            obs[10:12] = np.array([-100, -100])
            obs[12] = -100 #-5
        else:
            obs[10:12] = np.array(self.state[adv_agents[0]][:2]) - np.array(self.state[agent][:2]) # Adversary 1 rel xy position
            obs[12] = (self.state[adv_agents[0]][2] - self.state[agent][2]) % (self.num_headings) # Adversary 1 rel heading

        if not self.agent_alive[adv_agents[1]]:
            obs[13:15] = np.array([-100, -100])
            obs[15] = -100 #-5
        else:
            obs[13:15] = np.array(self.state[adv_agents[1]][:2]) - np.array(self.state[agent][:2]) # Adversary 2 rel xy position
            obs[15] = (self.state[adv_agents[1]][2] - self.state[agent][2]) % (self.num_headings) # Adversary 2 rel heading
        
        # Archived on Wimbledon Final day, July 13 @ 06.30 pm.
        obs_mask = np.ones(16, dtype=np.bool)
        if self.obs_mode == 'hetero':
            if agent[0] == 'R': pass
            elif agent[0] == 'B': obs_mask[5:7] = np.array([False, False])
        elif self.obs_mode == 'homo_blue':
            obs_mask[5:7] = np.array([False, False])
        elif self.obs_mode == 'homo_red':
            pass
        obs = obs[obs_mask]
        """
        # Commented out on Tuesday, July 15 2025.
        if self.obs_mode == 'hetero' and agent[0] == 'B':
            obs[5:7] = np.array([-100, -100], dtype=np.int8)  # Mask away flag
        elif self.obs_mode == 'homo_blue':
            obs[5:7] = np.array([-100, -100], dtype=np.int8)  # Mask away flag
        """
        return obs

    def _observations(self):
        # Implements the action_masks from the global state.
        obs = {}
        dummy_obs = np.zeros(self.observation_space(self.possible_agents[0])["observation"].shape, dtype=np.int8) #np.zeros(self.observation_space(self.possible_agents[0]).shape, dtype=np.int8) #np.zeros(self.observation_space(agent).shape, dtype=np.float32)
        for agent_idx, agent in enumerate(self.possible_agents):
            ##agent_obs = global_state # Had to comment due to some API glue-ing error that Google Colab pointed out.
            #agent_obs = np.concatenate([s for s in global_state]).astype(np.int8)

            if self.agent_alive[agent]: agent_obs = self.global_state_to_agent_local_obs(agent=agent) #v2 observations for better policy sharing.
            else:
                #dummy_obs = np.zeros(self.observation_space(self.possible_agents[agent_idx])["observation"].shape, dtype=np.int8) #np.zeros(self.observation_space(self.possible_agents[0]).shape, dtype=np.int8) #np.zeros(self.observation_space(agent).shape, dtype=np.float32)
                dummy_obs = -100*np.ones(self.observation_space(self.possible_agents[agent_idx])["observation"].shape, dtype=np.int8) #np.zeros(self.observation_space(self.possible_agents[0]).shape, dtype=np.int8) #np.zeros(self.observation_space(agent).shape, dtype=np.float32)
                agent_obs = dummy_obs

            valid = self.valid_actions(global_state_dict=self.state, agent=agent)
            action_mask = np.zeros(self.num_actions, dtype=np.int8)
            action_mask[valid] = 1
            obs[agent] = { "observation": agent_obs, "action_mask": action_mask }
        return obs

    def dynamics(self, agent, actions):
        # Single agent dynamics.
        ### Action keys:
        ## 0 = same pos, same heading
        ## 1 = same pos, minus heading
        ## 2 = same pos, plus heading
        ## 3 = advance pos, same heading
        ## 4 = tagging action

        assert type(agent) == str
        assert self.agent_alive[agent] is True, "agent: {}, self.agent_alive[agent]: {}".format(agent, self.agent_alive[agent])

        agent_team = "Red" if agent[0] == "R" else "Blue"
        #tag_bool, reach_goal_bool = False, False
        tag_bool, respawn_bool, death_bool, reach_goal_bool = False, False, False, False

        x, y, heading = self.state[agent]
        action = actions[agent]
        x_, y_, heading_ = None, None, None

        # Check if there are incoming tagging actions from nearby agents.
        potential_incoming_tagging_agents = []
        incoming_tagging_agents = []
        for neighbouring_agent in self.possible_agents:
            if not self.agent_alive[neighbouring_agent]: continue
            if neighbouring_agent == agent: continue
            else:
                x_neighbour, y_neighbour, heading_neighbour = self.state[neighbouring_agent]
                if abs(x - x_neighbour) <= 1 and abs(y - y_neighbour) <= 1:
                    neighbour_point_to_x, neighbour_point_to_y =  self.state[neighbouring_agent][:2] + self._heading_to_direction_vector(heading=heading_neighbour)
                    if (neighbour_point_to_x, neighbour_point_to_y) == (x, y): potential_incoming_tagging_agents.append(neighbouring_agent)
                    else: continue

        for agent_iter in potential_incoming_tagging_agents:
            if actions[agent_iter] == 4: incoming_tagging_agents.append(agent_iter)
            else: continue
        
        if len(incoming_tagging_agents) == 0:
            # No incoming tags.
            if action == 0:
                # same pos, same heading
                x_, y_, heading_ = x, y, heading
                pass
            elif action == 1:
                # same pos, minus heading
                x_, y_, heading_ = x, y, (heading - 1) % self.num_headings
                pass
            elif action == 2:
                # same pos, plus heading
                x_, y_, heading_ = x, y, (heading + 1) % self.num_headings
                pass
            elif action == 3:
                # advance pos, same heading
                x_, y_, heading_ = x, y, heading
                if heading == 0:
                    x_, y_ = x + 1, y
                elif heading == 1:
                    x_, y_ = x + 1, y + 1
                elif heading == 2:
                    x_, y_ = x, y + 1
                elif heading == 3:
                    x_, y_ = x - 1, y + 1
                elif heading == 4:
                    x_, y_ = x - 1, y
                elif heading == 5:
                    x_, y_ = x - 1, y - 1
                elif heading == 6:
                    x_, y_ = x, y - 1
                elif heading == 7:
                    x_, y_ = x + 1, y - 1
            elif action == 4: # Tagging action.
                x_, y_, heading_ = x, y, heading
        else:
            # Agent tagged.
            ### LOGIC: if respawn_allowed: respawn. Else: agent_death. have respawn_bool and death_bool as sub-categories of tag_bool.
            tag_bool = True
            if self.verbose: print(".............. TAG TAG TAG TAG TAG TAG TAG TAG TAG TAG TAG ..............")
            #respawn_allowed = self.respawn_info_state[agent_team] < self.max_respawns
            agents_on_agent_team = [agent for agent in self.possible_agents if agent[0] == agent_team[0]]
            team_tags = sum([self.respawn_info_state[agent] for agent in agents_on_agent_team])
            respawn_allowed = team_tags < self.max_respawns
            if self.verbose:
                print("agent: {}".format(agent))
                print("agent_team: {}".format(agent_team))
                print("agents on agent team: {}".format(agents_on_agent_team))
                print("respawn_info_state: {}".format([self.respawn_info_state[agent] for agent in agents_on_agent_team]))
                print("team_tags: {}".format(team_tags))
                print("self.max_respawns: {}".format(self.max_respawns))
                print("respawn_allowed if team_tags < self.max_respawns: {}".format(respawn_allowed))

            if respawn_allowed:
                # Incoming tags. Spawn in the tagging area.
                # Multiple tags = single penalty.
                self.respawn_info_state[agent] += 1
                respawn_bool = True
                """
                Spawning area is a rectangle, away from the flag in a corner.
                blue_tag_spawn_area = {"x_lim":(5, 7), "y_lim": (0, 1)} # (x, y) coordinate tuple -- x_low y_low and x_high y_high
                red_tag_spawn_area = {"x_lim":(0, 2), "y_lim": (6, 7)} # (x, y) coordinate tuple -- x_low y_low and x_high y_high
                """
                if agent_team == "Red":
                    spawn_area = self.red_tag_spawn_area
                    possible_spawn_headings = [0, 7, 6, 5] # [0, 7, 6, 5], possible init headings = (0, -45, -90, -135)
                elif agent_team == "Blue":
                    spawn_area = self.blue_tag_spawn_area
                    possible_spawn_headings = [0, 1, 2, 3] # possible init headings = (0, 45, 90, 135)
                x_prime = self.np_random.integers( spawn_area["x_lim"][0], spawn_area["x_lim"][1] + 1)
                y_prime = self.np_random.integers( spawn_area["y_lim"][0], spawn_area["y_lim"][1] + 1)

                if agent_team == "Red" and x_prime == 0:
                    try: possible_spawn_headings.remove(5)
                    except: pass
                elif agent_team == "Red" and x_prime == self.grid_size - 1:
                    try: possible_spawn_headings.remove(0)
                    except: pass
                    try: possible_spawn_headings.remove(7)
                    except: pass
                
                if agent_team == "Blue" and x_prime == 0:
                    try: possible_spawn_headings.remove(3)
                    except: pass
                elif agent_team == "Blue" and x_prime == self.grid_size - 1:
                    try: possible_spawn_headings.remove(0)
                    except: pass
                    try: possible_spawn_headings.remove(1)
                    except: pass

                heading_prime = self.np_random.choice(possible_spawn_headings)
                x_, y_, heading_ = x_prime, y_prime, heading_prime
            else:
                # Agent tagged but respawn not allowed hence agent_death.
                self.respawn_info_state[agent] += 1
                death_bool = True

            if self.verbose: print("respawn_info_state after updating: {}".format([self.respawn_info_state[agent] for agent in agents_on_agent_team]))
        
        if agent_team is self.away_team and (x_, y_) == self.flag_location:
            reach_goal_bool = True

        new_state = (x_, y_, heading_)
        return tag_bool, respawn_bool, death_bool, reach_goal_bool, new_state

    def _heading_to_direction_vector(self, heading):
        assert heading in range(self.num_headings)
        vec = None
        if heading == 0:
            vec = np.array([ +1, 0 ])
        elif heading == 1:
            vec = np.array([ +1, +1 ])
        elif heading == 2:
            vec = np.array([ 0, +1 ])
        elif heading == 3:
            vec = np.array([ -1, +1 ])
        elif heading == 4:
            vec = np.array([ -1, 0 ])
        elif heading == 5:
            vec = np.array([ -1, -1 ])
        elif heading == 6:
            vec = np.array([ 0, -1 ])
        elif heading == 7:
            vec = np.array([ +1, -1 ])
        return vec

    def valid_actions(self, global_state_dict, agent):
        """
        ### Action keys:
        ## 0 = same pos, same heading
        ## 1 = same pos, minus heading
        ## 2 = same pos, plus heading
        ## 3 = advance pos, same heading
        ## 4 = tagging action
        """

        # NOTE: No collision logic in v0.
        # Takes in as input global_state of the world and the agent name and returns valid actions for the agent in the global_state of the world.
        # Utility: to decide action_masking for the agent in the global_state.
        assert agent in self.possible_agents
        #if not self.agent_alive[agent]: return [0] # Commented on July 14, 2025 @ 03.30 pm.
        ##if not self.agent_alive[agent]: return [] # Added on July 14, 2025 @ 03.30 pm.
        if not self.agent_alive[agent]: return [0] # Added on July 15, 2025 @ 09.04 pm (weird action sampling error when applying action_mask during feed-forward via the neural network).

        agent_state = copy.deepcopy(global_state_dict[agent])
        # Action masking due to walls of the environment.
        x, y, heading = agent_state
        point_to_x, point_to_y =  agent_state[:2] + self._heading_to_direction_vector(heading=heading)

        valid_actions = [0, 1, 2, 3]
        invalid_action_x = None
        invalid_action_y = None

        if x == 0 and heading == 2:
            invalid_action_x = 2
        if x == 0 and heading == 6:
            invalid_action_x = 1
        if x == 0 and heading in [3, 4, 5]:
          invalid_action_x = 3
        if x == self.grid_size - 1 and heading == 2:
            invalid_action_x = 1
        if x == self.grid_size - 1 and heading == 6:
            invalid_action_x = 2
        if x == self.grid_size - 1 and heading in [0, 1, 7]:
          invalid_action_x = 3
        try:
            valid_actions.remove(invalid_action_x)
        except: pass

        if y == 0 and heading == 0:
            invalid_action_y = 1
        if y == 0 and heading == 4:
            invalid_action_y = 2
        if y == 0 and heading in [5, 6, 7]:
            invalid_action_y = 3
        if y == self.grid_size - 1 and heading == 0:
            invalid_action_y = 2
        if y == self.grid_size - 1 and heading == 4:
            invalid_action_y = 1
        if y == self.grid_size - 1 and heading in [1, 2, 3]:
            invalid_action_y = 3
        try:
            valid_actions.remove(invalid_action_y)
        except: pass

        # Asking masking on the Tagging space due to nearby agents.
        # Logic to see nearby agents position-wise: once a list of nearby agents is generated: check if heading is aligned with the neighbouring agent for potential tagging action.
        target_tagging_agents = []
        for potential_nearby_agent in self.possible_agents:
            if not self.agent_alive[potential_nearby_agent]: continue
            if potential_nearby_agent[0] == agent[0]: continue # Cannot tag same team member. HAD A BUG HERE previously. Corrected now.
            else:
                x_neighbour, y_neighbour, heading_neighbour = global_state_dict[potential_nearby_agent]
                if abs(x - x_neighbour) <= 1 and abs(y - y_neighbour) <= 1:
                    if (x_neighbour, y_neighbour) == (point_to_x, point_to_y): target_tagging_agents.append(potential_nearby_agent)
                    else: continue
                pass
            pass
        if len(target_tagging_agents) > 0: valid_actions.append(4) #Tagging action available.

        return valid_actions

    def state_space(self, agent):
        # The state of the agent on the grid: (x, y, theta).
        return MultiDiscrete([self.grid_size, self.grid_size, self.num_headings])

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Each agent can observe the state of all the agents.
        # obs_type = MultiDiscrete([self.grid_size, self.grid_size, self.num_headings, self.grid_size, self.grid_size, self.num_headings, self.grid_size, self.grid_size, self.num_headings, self.grid_size, self.grid_size, self.num_headings]) #Tuple([ self.state_space(agent) for agent in self.agents ])
        # coordinate of agent, heading, home_flag_rel_location, away_flag_rel_location, friend_rel_location, friend_rel_pose, adversary_1_rel_location, adversary_1_rel_pose, adversary_2_rel_location, adversary_2_rel_pose
        # 3 = self coordinate, 2 = home flag location, 2 = away flag location, 3 = friend rel pos, 3 = adversary 1 rel pos, 3 = adversary 2 rel pos
        assert agent in self.possible_agents
        obs_type_full = Box(low=0, high=self.grid_size, shape=(2 + 2 + len(self.possible_agents) * 3,), dtype=np.int8) # Away flag is visible.
        obs_type_partial = Box(low=0, high=self.grid_size, shape=(2 + 0 + len(self.possible_agents) * 3,), dtype=np.int8) # Away flag not visible.
        
        if self.obs_mode is None:
            if agent[0] == self.defense_team[0]: obs_type = obs_type_partial
            elif agent[0] == self.away_team[0]: obs_type = obs_type_full
        elif self.obs_mode == 'hetero':
            if agent[0] == 'R': obs_type = obs_type_full
            elif agent[0] == 'B': obs_type = obs_type_partial
        elif self.obs_mode == 'homo_blue':
            obs_type = obs_type_partial
        elif self.obs_mode == 'homo_red':
            obs_type = obs_type_full

        #obs_type = obs_type_full # Added on July 13 @ 06.20 pm. And add -1 observations to occlude away information.
        obs_space_type = Dict({"observation": obs_type, "action_mask": Box(low=0, high=1, shape=(self.num_actions,), dtype=np.int8)})
        return obs_space_type

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        assert agent in self.possible_agents
        #Simple dynamics. #same heading same place, heading plus, heading minus, move forward. And a tagging action.
        return Discrete(self.num_actions)

    def num_agents(self):
        return self.num_teams * self.num_agents_blue_team

    def render(self, history_traj=None):
        # Note: Once you have the state and the flag locations, this module can be written disjoint of any other logic in the whole program.
        blue_agent_pos = [self.state[agent] for agent in self.blue_team_agents if self.agent_alive[agent]]
        red_agent_pos = [self.state[agent] for agent in self.red_team_agents if self.agent_alive[agent]]
        if self.away_team == 'Red':
            fig, ax = draw_grid(grid_size=self.grid_size, blue_agent_pos=blue_agent_pos, red_agent_pos=red_agent_pos, blue_flag_loc=self.blue_flag_location, red_flag_loc=None, blue_flag_img_path=self.blue_flag_img_path, red_flag_img_path=self.red_flag_img_path)
        else:
            fig, ax = draw_grid(grid_size=self.grid_size, blue_agent_pos=blue_agent_pos, red_agent_pos=red_agent_pos, blue_flag_loc=None, red_flag_loc=self.red_flag_location, blue_flag_img_path=self.blue_flag_img_path, red_flag_img_path=self.red_flag_img_path)
        plt.show()
        return fig, ax

class Policy:
    def __init__(self, policy_network):
        from stable_baselines3 import PPO
        pass

def check_valid_policy(policy):
    assert isinstance(policy, dict)
    assert list(policy.keys()) == ['networks', 'weights']
    assert isinstance(policy['networks'][0], str)
    assert policy['weights'].shape == (len(policy['networks']), )
    return

class CoopEnv(ParallelEnv): # Comment: Not compatible with agent_deaths. Added on July 16, 2025.
    """
    NOTE: 09 May 2025 12.04 pm: There is some error here in training this code for 5M steps. Some valid_actions Assertion error at "assert CoopActionsDict[agent] in self.valid_actions(agent=agent)" in the step() function.
    """
    ## Do a heterogenous example too after the Best Response on the homogenous (/ symmetric) example. Red flag is somewhere messy in the grid and not (7,7).
    """
    This class inherits from ParallelEnv and provides a Cooperative environment with the opponent team policy fixed (as given by the input PolicySet) in the MixedCompCoop setting (the Capture-the-Flag environment).
    GOAL: This class should pass the ParallelEnv API test and should be a valid ParallelEnv class.
    """
    def __init__(self, MixedCompCoop: CustomCTF_v0, Policy, verbose=False, seed=None):
        """
        PENDING: Load policies using addresses not Policies in memory. And all of them should be on the same device.
        """
        from stable_baselines3 import PPO
        import numpy as np

        """
        CoopEnv both inherits and takes in as argument the MixedCompCoop environment i.e. CustomCTF_v0.
        ^^^ Rethinking the above. CoopEnv should just inherit ParallelEnv and take in as argument the MixedCompCoop CustomCTF_v0.
        Pending: Define the Policy class properly.
        """
        
        """
        Policy = { "metadata": {"team": "Red", "PolicyType": pt}, "policy": p}
        if pt == "single", p is a PPO model.
        if pt == "mixture", p is a dict. p = {"networks": [], "weights": np.ndarray([])}
        """
        self.verbose = verbose
        assert isinstance(MixedCompCoop, CustomCTF_v0) or isinstance(MixedCompCoop, CustomCTF_v1) or isinstance(MixedCompCoop, DefenseCTF)
        
        assert "metadata" in Policy and "policy" in Policy
        assert "team" in Policy["metadata"]
        #assert isinstance(Policy["policy"], PPO) # Commented on 05/27/2025 3.20 pm.
        
        
        assert "PolicyType" in Policy["metadata"]
        if Policy["metadata"]["PolicyType"] == "single": assert isinstance(Policy["policy"], PPO)
        elif Policy["metadata"]["PolicyType"] == "mixture":
            assert isinstance(Policy["policy"], dict)
            assert list(Policy["policy"].keys()) == ["networks", "weights"] # Policy["policy"] is a dict with keywords: networks and the other keyword weights.
            assert isinstance(Policy["policy"]["networks"], list)
            assert isinstance(Policy["policy"]["weights"], np.ndarray)
            assert Policy["policy"]["weights"].shape[0] == len(Policy["policy"]["networks"])
            assert abs(np.sum(Policy["policy"]["weights"]) - 1.) <= 1e-8

        if isinstance(MixedCompCoop, CustomCTF_v0) or isinstance(MixedCompCoop, CustomCTF_v1): self.metadata = { "name": "custom_ctf_v0_CoopEnv"}
        else: self.metadata = { "name": "defense_ctf_CoopEnv"}
        
        self.render_mode = "human"
        self.num_teams = 1
        self.MixedCompCoop = MixedCompCoop

        if seed is not None:
            self.seed(seed=seed)
        else:
            self._seed = None
            self.np_random = None
            self.torch_rng = None
        
        self.OppPolicyType = Policy["metadata"]["PolicyType"]
        if self.OppPolicyType == "single":
            self.OppPolicy = Policy["policy"]
        elif self.OppPolicyType == "mixture":
            self.OppPolicyMixture = Policy["policy"]["networks"]
            self.OppPolicyMixtureSize = len(self.OppPolicyMixture)
            self.OppPolicyWeights = Policy["policy"]["weights"]
            self.OppPolicy = None # This is sampled from the mixture at each reset of the environment.
        """
        Idea: Write more portable code such that this class only takes a mixture net with networks and weights even if it's a single network with a weight of one.
        """
        #self.OppPolicy = Policy["policy"]

        self.grid_size = self.MixedCompCoop.grid_size #this defines the state space of the Markov Game and hence the observation_space.
        self.num_headings = self.MixedCompCoop.num_headings  # Possible headings in degrees:(0, 45, 90, 135, 180, 225, 270, 315)
        self.num_actions = self.MixedCompCoop.num_actions
        self.max_num_cycles = self.MixedCompCoop.max_num_cycles

        if Policy["metadata"]["team"] == "Blue":
            self.passive_agents = copy.deepcopy(self.MixedCompCoop.blue_team_agents)
            self.team = "Red"
            self.num_agents = self.MixedCompCoop.num_agents_red_team
            self.agents = copy.deepcopy(self.MixedCompCoop.red_team_agents) #[ "Red_{}".format(agent_idx) for agent_idx in range(self.MixedCompCoop.num_agents_red_team) ]
        elif Policy["metadata"]["team"] == "Red":
            self.passive_agents = copy.deepcopy(self.MixedCompCoop.red_team_agents)
            self.team = "Blue"
            self.num_agents = self.MixedCompCoop.num_agents_blue_team
            self.agents = copy.deepcopy(self.MixedCompCoop.blue_team_agents) #[ "Blue_{}".format(agent_idx) for agent_idx in range(self.MixedCompCoop.num_agents_blue_team) ]
        
        if self.verbose: print("self.agents INITTTTT: {}".format(self.agents))
        self.possible_agents = copy.deepcopy(self.agents)
        if self.verbose: print(self.possible_agents)

        self.actors_in_the_env = copy.deepcopy(self.agents + self.passive_agents)
        
        self.current_step = 0
        self.state = { actor: None for actor in self.actors_in_the_env }

        self.observation_spaces = {agent: self.observation_space(agent=agent)["observation"] for agent in self.agents}
        self.action_spaces = {agent: self.action_space(agent=agent) for agent in self.agents}

        self._init_args = (MixedCompCoop, Policy)
        self._init_kwargs = {"verbose":verbose, "seed":seed}

    def __call__(self):
        return CoopEnv(*self._init_args, **self._init_kwargs)

    def seed(self, seed=None):
        self.rngs = make_seeded_rngs(seed)
        self._seed = self.rngs["actual_seed"]
        self.np_random = self.rngs["np_random"]
        self.torch_rng = self.rngs["torch_rng"]
        return [self._seed]

    def num_agents(self):
        return len(self.agents)

    def reset(self, seed = None, options = None):
        """
        Pending: if policy is a mixture, just sample a policy out of the mixture at every reset and use it for the rest of the episode.
        Just sample a policy index from the mixture and load that policy.
        Pending: For an implementation with just the addresses of the stored networks, load all the networks in the ensemble and store in a list.
        """
        if seed is not None:
            self.seed(seed=seed)

        if self.OppPolicyType == "mixture":
            # Sample a network idx from the mixture and set the self.OppPolicy global variable (None before setting).
            policy_network_idx = self.np_random.choice(self.OppPolicyMixtureSize, p=self.OppPolicyWeights)
            self.OppPolicy = self.OppPolicyMixture[policy_network_idx]
        
        self.current_step = 0
        self.agents = copy.deepcopy(self.possible_agents) ########### !!!!!!!!!!!! WHY NEED THIS LINE WHEN self.agents is initialized in __init__(). Answer: to avoid no agents when the environment is reset after being terminated / truncated previously when the agents array is emptied.
        obs, info = self.MixedCompCoop.reset()
        
        # Copy the MixedCompCoop state to the current environment state.
        for actor in self.actors_in_the_env:
            self.state[actor] = self.MixedCompCoop.state[actor]
        
        coop_obs = {agent: obs[agent] for agent in self.agents}
        coop_info = {agent: info[agent] for agent in self.agents}
        if self.verbose: print("self.agents: {}".format(self.agents))
        return coop_obs, coop_info
    
    def evalPolicy(self, policy, obs, agent):
        assert agent in self.passive_agents
        assert isinstance(obs, dict)
        assert "observation" in obs and "action_mask" in obs
        assert isinstance(policy, PPO)

        obs_tensor = th.as_tensor(obs["observation"]).unsqueeze(0)
        action_mask_tensor = th.as_tensor(obs["action_mask"]).unsqueeze(0)
        valid_actions = self.MixedCompCoop.valid_actions(self.MixedCompCoop.state, agent)
        
        with th.no_grad():
            features = policy.policy.extract_features({"observation": obs_tensor, "action_mask": action_mask_tensor})
            latent_pi, _ = policy.policy.mlp_extractor(features)
            distribution = policy.policy._get_action_dist_from_latent(latent_pi)  # get the action distribution

            # Apply action mask to the distribution
            logits = distribution.distribution.logits
            # Following commented on July 15 2025, Tuesday @ 8.55 pm.
            """
            # Avoid log(0) by clamping values
            mask = (action_mask_tensor + 1e-8).log()
            masked_logits = logits + mask
            """
            masked_logits = logits.masked_fill(action_mask_tensor == 0, -1e10)
            distribution.distribution.logits = masked_logits

            # Get the predicted action (from the distribution now)
            agent_action = distribution.distribution.sample(generator=self.torch_rng).cpu().numpy()[0] #distribution.get_actions(deterministic=False).cpu().numpy()[0] # Assuming batch_size = 1
            if self.verbose: print(agent_action)

        if agent_action not in valid_actions:
            if self.verbose: print("Policy predicted invalid action for agent !!!!!")
            agent_action = self.np_random.choice(valid_actions)

        return agent_action
    
    def valid_actions(self, agent):
        assert agent in self.agents
        return self.MixedCompCoop.valid_actions(self.MixedCompCoop.state, agent)

    def step(self, CoopActionsDict):
        if self.verbose:
            print("self.agents: {}".format(self.agents))
            print("len(CoopActionsDict.keys()) = {}".format(len(CoopActionsDict.keys())))
            print("self.num_agents = {}".format(self.num_agents))
        #assert len(CoopActionsDict.keys()) == self.num_agents

        for agent in self.agents:
            assert CoopActionsDict[agent] in self.valid_actions(agent=agent), "CoopActionsDict[agent]: {}, self.valid_actions(agent=agent): {}".format(CoopActionsDict[agent], self.valid_actions(agent=agent))
            assert CoopActionsDict[agent] in self.action_space(agent=agent), "CoopActionsDict[agent]: {}, self.action_space(agent=agent): {}".format(CoopActionsDict[agent], self.action_space(agent=agent))
        assert self.OppPolicy is not None
        
        """
        Take in actions dict just for the Coop team, augments with actions for the opponent team from the input Policy and steps through the MixedCompCoop environment.
        """
        ActionsDict = {}
        for actor in self.actors_in_the_env:
            if actor in self.agents: ActionsDict[actor] = CoopActionsDict[actor]
            elif actor in self.passive_agents:
                actor_obs = self.MixedCompCoop.global_state_to_agent_local_obs(agent=actor)
                valid_actions = self.MixedCompCoop.valid_actions(global_state_dict=self.MixedCompCoop.state, agent=actor)
                action_mask = np.zeros(self.MixedCompCoop.num_actions, dtype=np.int8)
                action_mask[valid_actions] = 1
                actor_obs = { "observation": actor_obs, "action_mask": action_mask }
                ActionsDict[actor] = self.evalPolicy(self.OppPolicy, obs=actor_obs, agent=actor)
                assert ActionsDict[actor] in valid_actions
        
        obs, rewards, terminations_, truncations_, infos_ = self.MixedCompCoop.step(ActionsDict)

        obs_coop = {agent: obs[agent] for agent in self.agents}
        rewards_coop = {agent: rewards[agent] for agent in self.agents}
        terminations = {agent: terminations_[agent] for agent in self.agents}
        truncations = {agent: truncations_[agent] for agent in self.agents}
        infos = {agent: infos_[agent] for agent in self.agents}

        for actor in self.actors_in_the_env:
            self.state[actor] = self.MixedCompCoop.state[actor]
        self.current_step += 1

        agents_to_remove = []
        for agent in self.agents:
            if terminations[agent] or truncations[agent]:
                agents_to_remove.append(agent)

        # Remove done agents
        for agent in agents_to_remove:
            self.agents.remove(agent)
        
        if self.verbose:
            print(obs_coop)
            print(rewards_coop)
            print(terminations)
            print(truncations)
            print(infos)

        return obs_coop, rewards_coop, terminations, truncations, infos
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        assert agent in self.agents
        #return super().action_space(agent)
        return self.MixedCompCoop.action_space(agent)
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        assert agent in self.agents
        return self.MixedCompCoop.observation_space(agent)

class CoopEnv_v1(ParallelEnv): #CoopEnv_v0 with policy_paths instead of policies.
    """
    This class inherits from ParallelEnv and provides a Cooperative environment with the opponent team policy fixed (as given by the input PolicySet) in the MixedCompCoop setting (the Capture-the-Flag environment).
    GOAL: This class should pass the ParallelEnv API test and should be a valid ParallelEnv class.
    """
    """
    IMPORTANT: v1 implementation just makes the change to loading OppPolicies from paths on a 'CPU' and still assumes homogenous observation spaces amongst the two teams.

    """
    def __init__(self, MixedCompCoop: CustomCTF_v0, Policy, verbose=False, seed=None):
        """
        PENDING: Load policies using addresses not Policies in memory. And all of them should be on the same device.
        """
        from stable_baselines3 import PPO
        import numpy as np

        """
        CoopEnv both inherits and takes in as argument the MixedCompCoop environment i.e. CustomCTF_v0.
        ^^^ Rethinking the above. CoopEnv should just inherit ParallelEnv and take in as argument the MixedCompCoop CustomCTF_v0.
        Pending: Define the Policy class properly.
        """
        
        """
        Policy = { "metadata": {"team": "Red", "PolicyType": pt}, "policy": p}
        if pt == "single", p is a PPO model path.
        if pt == "mixture", p is a dict. p = {"networks": [], "weights": np.ndarray([])}
        """
        self.verbose = verbose
        assert isinstance(MixedCompCoop, CustomCTF_v0) or isinstance(MixedCompCoop, CustomCTF_v1) or isinstance(MixedCompCoop, DefenseCTF_v2)
        
        assert "metadata" in Policy and "policy" in Policy
        assert "team" in Policy["metadata"]
        #assert isinstance(Policy["policy"], PPO) # Commented on 05/27/2025 3.20 pm.
        
        assert "PolicyType" in Policy["metadata"]
        if Policy["metadata"]["PolicyType"] == "single":
            assert isinstance(Policy["policy"], str) #isinstance(Policy["policy"], PPO)
        elif Policy["metadata"]["PolicyType"] == "mixture":
            assert isinstance(Policy["policy"], dict)
            assert list(Policy["policy"].keys()) == ["networks", "weights"] # Policy["policy"] is a dict with keywords: networks and the other keyword weights.
            assert isinstance(Policy["policy"]["networks"], list)
            assert isinstance(Policy["policy"]["weights"], np.ndarray)
            assert Policy["policy"]["weights"].shape[0] == len(Policy["policy"]["networks"])
            assert abs(np.sum(Policy["policy"]["weights"]) - 1.) <= 1e-8
        
        if isinstance(MixedCompCoop, CustomCTF_v0) or isinstance(MixedCompCoop, CustomCTF_v1): self.metadata = { "name": "custom_ctf_v0_CoopEnv"}
        else: self.metadata = { "name": "defense_ctf_CoopEnv"}
        
        self.render_mode = "human"
        self.num_teams = 1
        self.MixedCompCoop = MixedCompCoop
        self.MixedCompCoopType = 'defense' if isinstance(self.MixedCompCoop, DefenseCTF_v2) else 'ctf'
        self.agent_deaths = True if (hasattr(self.MixedCompCoop, 'max_respawns') and self.MixedCompCoop.max_respawns > 0) else False
        #self.MixedCompCoop_vec_env = ss.pettingzoo_env_to_vec_env_v1(self.MixedCompCoop)
        #self.MixedCompCoop_vec_env = ss.concat_vec_envs_v1(self.MixedCompCoop_vec_env, num_vec_envs=1, base_class="stable_baselines3")
        
        if seed is not None:
            self.seed(seed=seed)
        else:
            self._seed = None
            self.np_random = None
            self.torch_rng = None

        self.OppPolicyType = Policy["metadata"]["PolicyType"]
        if self.OppPolicyType == "single":
            #self.OppPolicy = PPO.load(Policy["policy"], env=self.MixedCompCoop_vec_env, device='cpu')
            self.OppPolicy = PPO.load(Policy["policy"], env=None, device='cpu')
        elif self.OppPolicyType == "mixture":
            #self.OppPolicyMixture = [PPO.load(path, env=self.MixedCompCoop_vec_env, device='cpu') for path in Policy["policy"]["networks"]]
            self.OppPolicyMixture = [PPO.load(path, env=None, device='cpu') for path in Policy["policy"]["networks"]]
            self.OppPolicyMixtureSize = len(self.OppPolicyMixture)
            self.OppPolicyWeights = Policy["policy"]["weights"]
            self.OppPolicy = None # This is sampled from the mixture at each reset of the environment.
        """
        Idea: Write more portable code such that this class only takes a mixture net with networks and weights even if it's a single network with a weight of one.
        """
        #self.OppPolicy = Policy["policy"]

        self.grid_size = self.MixedCompCoop.grid_size #this defines the state space of the Markov Game and hence the observation_space.
        self.num_headings = self.MixedCompCoop.num_headings  # Possible headings in degrees:(0, 45, 90, 135, 180, 225, 270, 315)
        self.num_actions = self.MixedCompCoop.num_actions
        self.max_num_cycles = self.MixedCompCoop.max_num_cycles

        if Policy["metadata"]["team"] == "Blue":
            self.passive_agents = copy.deepcopy(self.MixedCompCoop.blue_team_agents)
            self.team = "Red"
            self.num_agents = self.MixedCompCoop.num_agents_red_team
            self.agents = copy.deepcopy(self.MixedCompCoop.red_team_agents) #[ "Red_{}".format(agent_idx) for agent_idx in range(self.MixedCompCoop.num_agents_red_team) ]
        elif Policy["metadata"]["team"] == "Red":
            self.passive_agents = copy.deepcopy(self.MixedCompCoop.red_team_agents)
            self.team = "Blue"
            self.num_agents = self.MixedCompCoop.num_agents_blue_team
            self.agents = copy.deepcopy(self.MixedCompCoop.blue_team_agents) #[ "Blue_{}".format(agent_idx) for agent_idx in range(self.MixedCompCoop.num_agents_blue_team) ]
        
        if self.verbose: print("self.agents INITTTTT: {}".format(self.agents))
        self.possible_agents = copy.deepcopy(self.agents)
        if self.verbose: print(self.possible_agents)
        self.actors_in_the_env = copy.deepcopy(self.agents + self.passive_agents)

        if self.agent_deaths:
            if self.verbose: print("Agent deaths is True in the parent MixedCompCoop environment...")
            self.agent_alive = {agent: None for agent in self.possible_agents} # set to True for each agent on resetting.
        
        self.current_step = 0
        self.state = { actor: None for actor in self.actors_in_the_env }

        self.observation_spaces = {agent: self.observation_space(agent=agent)["observation"] for agent in self.agents}
        self.action_spaces = {agent: self.action_space(agent=agent) for agent in self.agents}

        self._init_args = (MixedCompCoop, Policy)
        self._init_kwargs = {"verbose": verbose, "seed": seed}

    def __call__(self):
        return CoopEnv_v1(*self._init_args, **self._init_kwargs)

    def seed(self, seed=None):
        self.rngs = make_seeded_rngs(seed)
        self._seed = self.rngs["actual_seed"]
        self.np_random = self.rngs["np_random"]
        self.torch_rng = self.rngs["torch_rng"]
        return [self._seed]

    def num_agents(self):
        return len(self.agents)

    def reset(self, seed = None, options = None):
        """
        Pending: if policy is a mixture, just sample a policy out of the mixture at every reset and use it for the rest of the episode.
        Just sample a policy index from the mixture and load that policy.
        Pending: For an implementation with just the addresses of the stored networks, load all the networks in the ensemble and store in a list.
        """
        if seed is not None:
            self.seed(seed=seed)

        if self.OppPolicyType == "mixture":
            # Sample a network idx from the mixture and set the self.OppPolicy global variable (None before setting).
            policy_network_idx = self.np_random.choice(self.OppPolicyMixtureSize, p=self.OppPolicyWeights)
            self.OppPolicy = self.OppPolicyMixture[policy_network_idx]
        
        self.current_step = 0
        self.agents = copy.deepcopy(self.possible_agents) ########### !!!!!!!!!!!! WHY NEED THIS LINE WHEN self.agents is initialized in __init__(). Answer: to avoid no agents when the environment is reset after being terminated / truncated previously when the agents array is emptied.
        obs, info = self.MixedCompCoop.reset()
        
        # Copy the MixedCompCoop state to the current environment state.
        for actor in self.actors_in_the_env:
            self.state[actor] = self.MixedCompCoop.state[actor]
            if self.agent_deaths: self.agent_alive[actor] = self.MixedCompCoop.agent_alive[actor]
        
        coop_obs = {agent: obs[agent] for agent in self.agents}
        coop_info = {agent: info[agent] for agent in self.agents}
        if self.verbose: print("self.agents: {}".format(self.agents))
        return coop_obs, coop_info
    
    def evalPolicy(self, policy, obs, agent):
        assert agent in self.passive_agents
        assert isinstance(obs, dict)
        assert "observation" in obs and "action_mask" in obs
        assert isinstance(policy, PPO)

        obs_tensor = th.as_tensor(obs["observation"]).unsqueeze(0)
        action_mask_tensor = th.as_tensor(obs["action_mask"]).unsqueeze(0)
        valid_actions = self.MixedCompCoop.valid_actions(self.MixedCompCoop.state, agent)
        
        with th.no_grad():
            features = policy.policy.extract_features({"observation": obs_tensor, "action_mask": action_mask_tensor})
            latent_pi, _ = policy.policy.mlp_extractor(features)
            distribution = policy.policy._get_action_dist_from_latent(latent_pi)  # get the action distribution

            # Apply action mask to the distribution
            logits = distribution.distribution.logits
            # Following commented on July 15 2025 @ 8.56 pm.
            """
            # Avoid log(0) by clamping values
            mask = (action_mask_tensor + 1e-8).log()
            masked_logits = logits + mask
            """
            masked_logits = logits.masked_fill(action_mask_tensor == 0, -1e10)
            distribution.distribution.logits = masked_logits

            # Get the predicted action (from the distribution now)
            dist = distribution.distribution  # For clarity

            if hasattr(dist, 'sample') and 'generator' in dist.sample.__code__.co_varnames:
                sample = dist.sample(generator=self.torch_rng)
            else:
                # fallback if generator is not accepted
                torch.manual_seed(self.torch_rng.initial_seed())
                sample = dist.sample()

            agent_action = sample.cpu().numpy()[0]

            #agent_action = distribution.distribution.sample(generator=self.torch_rng).cpu().numpy()[0] #distribution.get_actions(deterministic=False).cpu().numpy()[0] # Assuming batch_size = 1
            if self.verbose: print(agent_action)

        if agent_action not in valid_actions:
            if self.verbose: print("Policy predicted invalid action for agent !!!!!")
            agent_action = self.np_random.choice(valid_actions)

        return agent_action
    
    def valid_actions(self, agent):
        assert agent in self.agents
        return self.MixedCompCoop.valid_actions(self.MixedCompCoop.state, agent)

    def step(self, CoopActionsDict):
        if self.verbose:
            print("self.agents: {}".format(self.agents))
            print("len(CoopActionsDict.keys()) = {}".format(len(CoopActionsDict.keys())))
            print("self.num_agents = {}".format(self.num_agents))
        #assert len(CoopActionsDict.keys()) == self.num_agents

        for agent in self.agents:
            assert CoopActionsDict[agent] in self.valid_actions(agent=agent), "CoopActionsDict[agent]: {}, self.valid_actions(agent=agent): {}".format(CoopActionsDict[agent], self.valid_actions(agent=agent))
            assert CoopActionsDict[agent] in self.action_space(agent=agent), "CoopActionsDict[agent]: {}, self.action_space(agent=agent): {}".format(CoopActionsDict[agent], self.action_space(agent=agent))
        assert self.OppPolicy is not None
        
        """
        Take in actions dict just for the Coop team, augments with actions for the opponent team from the input Policy and steps through the MixedCompCoop environment.
        """
        ActionsDict = {}
        for actor in self.actors_in_the_env:
            if self.agent_deaths:
                if not self.agent_alive[actor]: continue
            if actor in self.agents: ActionsDict[actor] = CoopActionsDict[actor]
            elif actor in self.passive_agents:
                actor_obs = self.MixedCompCoop.global_state_to_agent_local_obs(agent=actor)
                valid_actions = self.MixedCompCoop.valid_actions(global_state_dict=self.MixedCompCoop.state, agent=actor)
                action_mask = np.zeros(self.MixedCompCoop.num_actions, dtype=np.int8)
                action_mask[valid_actions] = 1
                actor_obs = { "observation": actor_obs, "action_mask": action_mask }
                ActionsDict[actor] = self.evalPolicy(self.OppPolicy, obs=actor_obs, agent=actor)
                assert ActionsDict[actor] in valid_actions
        
        obs, rewards, terminations_, truncations_, infos_ = self.MixedCompCoop.step(ActionsDict)

        obs_coop = {agent: obs[agent] for agent in self.agents}
        rewards_coop = {agent: rewards[agent] for agent in self.agents}
        terminations = {agent: terminations_[agent] for agent in self.agents}
        truncations = {agent: truncations_[agent] for agent in self.agents}
        infos = {agent: infos_[agent] for agent in self.agents}

        for actor in self.actors_in_the_env:
            self.state[actor] = self.MixedCompCoop.state[actor]
            if self.agent_deaths: self.agent_alive[actor] = self.MixedCompCoop.agent_alive[actor]
        self.current_step += 1

        agents_to_remove = []
        for agent in self.agents:
            if terminations[agent] or truncations[agent]:
                agents_to_remove.append(agent)

        # Remove done agents
        for agent in agents_to_remove:
            self.agents.remove(agent)
        
        if self.verbose:
            print(obs_coop)
            print(rewards_coop)
            print(terminations)
            print(truncations)
            print(infos)

        return obs_coop, rewards_coop, terminations, truncations, infos
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        assert agent in self.agents
        #return super().action_space(agent)
        return self.MixedCompCoop.action_space(agent)
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        assert agent in self.agents
        return self.MixedCompCoop.observation_space(agent)

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def make_env(env_constructor, seed=0, rank=0, env_args=(), env_kwargs=None):
    import inspect
    assert (
        inspect.isclass(env_constructor) or inspect.isfunction(env_constructor) or callable(env_constructor)
    ), "env_constructor_type: {}. env_constructor must be a class, function, or callable object that returns an environment instance.".format(type(env_constructor))

    if env_kwargs is None: env_kwargs = {}
    def _init():
        env = env_constructor(*env_args, **env_kwargs) #env = env_constructor(**env_kwargs)
        seed_ = seed + rank
        env.seed(seed_)
        return env
    return _init

def make_vec_env(env_constructor, seed=0, num_vec_envs=8, env_args=(), env_kwargs=None):
    """
    This is a monkey patch! Handle with care. Rank based seed-offsetting is handled by Supersuit's ConcatVecEnv internally! Monkey patch works with the current version of Supersuit i.e. supersuit 3.10.0 as of July 25, 2025.
    """
    from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper
    env = make_env(env_constructor, seed=seed, rank=0, env_args=env_args, env_kwargs=env_kwargs)()
    assert isinstance(env, ParallelEnv)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
    vec_env = ss.concat_vec_envs_v1(vec_env, num_vec_envs=num_vec_envs, base_class="stable_baselines3")
    """ The following piece of code applies seed offsets to all the num_vecs VecEns due to Supersuit internals. Refer to Supersuit code for details. """
    def patch_seed_for_concat_vec_env(sb3_wrapped_env):
        
        assert isinstance(sb3_wrapped_env, SB3VecEnvWrapper)
        
        if not hasattr(sb3_wrapped_env.venv, "seed"):
            def seed(self, seed=None):
                return self.reset(seed=seed)
            from types import MethodType
            sb3_wrapped_env.venv.seed = MethodType(seed, sb3_wrapped_env.venv) #sb3_wrapped_env.venv accesses the underlying VecEnv class by Stable_Baselines3.
        return sb3_wrapped_env
    
    vec_env = patch_seed_for_concat_vec_env(vec_env)
    vec_env.seed(seed=seed)
    assert isinstance(vec_env, SB3VecEnvWrapper)
    return vec_env
"""
The following utility is a copy of the path_seed_for_concat_vec_env() method in the make_vec_env() method above.
"""
def patch_seed_for_concat_vec_env(sb3_wrapped_env):
    from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper
    assert isinstance(sb3_wrapped_env, SB3VecEnvWrapper)

    if not hasattr(sb3_wrapped_env.venv, "seed"):
        def seed(self, seed=None):
            return self.reset(seed=seed)
        from types import MethodType
        sb3_wrapped_env.venv.seed = MethodType(seed, sb3_wrapped_env.venv) #sb3_wrapped_env.venv accesses the underlying VecEnv class by Stable_Baselines3.
    return sb3_wrapped_env

def BestResponse(env: CustomCTF_v0, Policy, device='cpu', num_million=2, learn=True, load=False, load_path=None, model_save_path=None, seed=None):
    """
    Policy basic implementation is to serve as a placeholder for a single policy for an agent. Here, "str" is the agent / team ID for the CustomCTF_v0 / MixedCompCoop environment.
    """
    """
    Pending debugging: 3M steps some Assertion error. Also, cuda device mismatch debug.
    """
    assert "metadata" in Policy and "policy" in Policy
    assert "team" in Policy["metadata"]        
    assert "PolicyType" in Policy["metadata"]
    if Policy["metadata"]["PolicyType"] == "single": assert isinstance(Policy["policy"], PPO)
    elif Policy["metadata"]["PolicyType"] == "mixture":
        assert isinstance(Policy["policy"], dict)
        assert list(Policy["policy"].keys()) == ["networks", "weights"] # Policy["policy"] is a dict with keywords: networks and the other keyword weights.
        assert isinstance(Policy["policy"]["networks"], list)
        assert isinstance(Policy["policy"]["weights"], np.ndarray)
        assert Policy["policy"]["weights"].shape[0] == len(Policy["policy"]["networks"])
        assert abs(np.sum(Policy["policy"]["weights"]) - 1.) <= 1e-8 #np.sum(Policy["policy"]["weights"]) == 1.

    if load: assert load_path is not None
    coop_env = CoopEnv(MixedCompCoop=env, Policy=Policy)

    # Learn an IPPO policy (shared amongst the team / policy of the population looks like a same policy running on both the agents). Steps:
    ## First convert the CoopEnv which is an instance of a PettingZoo ParallelEnv to a MarkovVecEnv via the wrapper provided by Supersuit.
    ## Second, run PPO on the vectorized environment.
    vec_env = ss.pettingzoo_env_to_vec_env_v1(coop_env)
    vec_env = ss.concat_vec_envs_v1(vec_env, num_vec_envs=4, base_class="stable_baselines3")

    if device is None: device = 'cuda' if th.cuda.is_available() else 'cpu'

    model_best_response = PPO(
        policy=MaskedMultiInputPolicy,   # your custom policy class
        env=vec_env,            # your PettingZoo env, possibly converted to gym
        verbose=1,
        device=device
    )

    if learn:
        if load: pass # Add code to load_model and resume training.
        import time
        t1 = time.time()
        model_best_response.learn(total_timesteps=100000*2.5*(2*num_million), progress_bar=True)
        ### Add a try and except statement in the above model_best_response.learn() to catch errors and better error handling.
        
        if model_save_path is None: model_save_path = "ppo_CTF_{}M_steps_BestResponse".format(num_million)
        model_best_response.save(model_save_path)
        
        t2 = time.time()
        print("Time to train {}M steps of the BestResponse via CoopEnv wrapper: {}s".format(num_million, t2-t1))
        print("=="*80)

    """
    Pending work: Learn a CTDE BestResponse to the Cooperative environment defined by CoopEnv(MixedCompCoopEnv, Policy).
    # Define Policy to be IPPO policies? And have a separate class for CTDE BestResponses?
    """
    return model_best_response, model_save_path

def BestResponse_v1(env: CustomCTF_v0, Policy, device=None, num_million=2, learn=True, load=False, load_path=None, model_save_path=None, seed=None):
    """
    Policy basic implementation is to serve as a placeholder for a single policy for an agent. Here, "str" is the agent / team ID for the CustomCTF_v0 / MixedCompCoop environment.
    """
    assert "metadata" in Policy and "policy" in Policy
    assert "team" in Policy["metadata"]        
    assert "PolicyType" in Policy["metadata"]
    if Policy["metadata"]["PolicyType"] == "single": assert isinstance(Policy["policy"], str) #assert isinstance(Policy["policy"], PPO)
    elif Policy["metadata"]["PolicyType"] == "mixture":
        assert isinstance(Policy["policy"], dict)
        assert list(Policy["policy"].keys()) == ["networks", "weights"] # Policy["policy"] is a dict with keywords: networks and the other keyword weights.
        assert isinstance(Policy["policy"]["networks"], list)
        assert isinstance(Policy["policy"]["weights"], np.ndarray)
        assert Policy["policy"]["weights"].shape[0] == len(Policy["policy"]["networks"])
        assert abs(np.sum(Policy["policy"]["weights"]) - 1.) <= 1e-8 #np.sum(Policy["policy"]["weights"]) == 1.
    if load: assert load_path is not None

    coop_env = CoopEnv_v1(MixedCompCoop=env, Policy=Policy)

    # Learn an IPPO policy (shared amongst the team / policy of the population looks like a same policy running on both the agents). Steps:
    ## First convert the CoopEnv which is an instance of a PettingZoo ParallelEnv to a MarkovVecEnv via the wrapper provided by Supersuit.
    ## Second, run PPO on the vectorized environment.
    vec_env = ss.pettingzoo_env_to_vec_env_v1(coop_env)
    vec_env = ss.concat_vec_envs_v1(vec_env, num_vec_envs=4, base_class="stable_baselines3")

    if device is None: device = 'cuda' if th.cuda.is_available() else 'cpu'

    model_best_response = PPO(
            policy=MaskedMultiInputPolicy,   # your custom policy class
            env=vec_env,            # your PettingZoo env, possibly converted to gym
            verbose=1,
            device=device
        )
    
    if load:
        load_model_best_response = PPO.load(load_path, env=vec_env, device=device)
        model_best_response.policy.load_state_dict(load_model_best_response.policy.state_dict())

    import time
    t1 = time.time()
    model_best_response.learn(total_timesteps=100000*2.5*(2*num_million), progress_bar=True) ### Add a try and except statement in the model_best_response.learn() to catch errors and better error handling.
    if model_save_path is None: model_save_path = "ppo_CTF_{}M_steps_BestResponse".format(num_million)
    model_best_response.save(model_save_path)
    t2 = time.time()
    print("Time to train {}M steps of the BestResponse via CoopEnv wrapper: {}s".format(num_million, t2-t1))
    print("=="*80)

    """
    Pending work: Learn a CTDE BestResponse to the Cooperative environment defined by CoopEnv(MixedCompCoopEnv, Policy).
    # Define Policy to be IPPO policies? And have a separate class for CTDE BestResponses?
    """
    return model_best_response, model_save_path

def check_valid_Policy(Policy):
    assert "metadata" in Policy and "policy" in Policy
    assert "team" in Policy["metadata"]        
    assert "PolicyType" in Policy["metadata"]
    if Policy["metadata"]["PolicyType"] == "single": assert isinstance(Policy["policy"], str) #assert isinstance(Policy["policy"], PPO)
    elif Policy["metadata"]["PolicyType"] == "mixture":
        assert isinstance(Policy["policy"], dict)
        assert list(Policy["policy"].keys()) == ["networks", "weights"] # Policy["policy"] is a dict with keywords: networks and the other keyword weights.
        assert isinstance(Policy["policy"]["networks"], list)
        assert isinstance(Policy["policy"]["weights"], np.ndarray)
        assert Policy["policy"]["weights"].shape[0] == len(Policy["policy"]["networks"]), "policy weights shape: {} \n len policy network: {}".format(Policy["policy"]["weights"].shape[0], len(Policy["policy"]["networks"]))
        assert abs(np.sum(Policy["policy"]["weights"]) - 1.) <= 1e-8 #np.sum(Policy["policy"]["weights"]) == 1.
    return

from train import *
def BestResponse_v2(env_constructor, env_seed, env_args, env_kwargs, num_vec_envs, Policy, device=None, num_million=2, load_path=None, model_save_path=None, br_seed=None, eval_env_bool=True, hyperparams={}, **kwargs):
    """
    Policy basic implementation is to serve as a placeholder for a single policy for an agent. Here, "str" is the agent / team ID for the CustomCTF_v0 / MixedCompCoop environment.
    v2: Calls train to train the BestResponse.
    """
    set_global_seed(seed=br_seed)
    check_valid_Policy(Policy=Policy)
    mixedcompcoopenv = make_env(env_constructor=env_constructor, seed=env_seed, env_args=env_args, env_kwargs=env_kwargs)()
    coop_env = CoopEnv_v1(MixedCompCoop=mixedcompcoopenv, Policy=Policy) # =====> this is a ParallelEnv, input this into the train function with the seed.
    
    if eval_env_bool:
        eval_env = ss.pettingzoo_env_to_vec_env_v1(coop_env)
        eval_env = ss.concat_vec_envs_v1(eval_env, num_vec_envs=1, base_class="stable_baselines3")
        eval_env = patch_seed_for_concat_vec_env(eval_env)
        eval_env.seed(seed=env_seed)

    _, model_save_path = train_v1(train_env=coop_env,
                    train_env_seed=env_seed,
                    num_train_vec_envs=num_vec_envs,
                    train_policy_seed=br_seed,
                    device=device,
                    num_million=num_million,
                    load_path=load_path,
                    load_perturb=False,
                    model_save_path=model_save_path,
                    hyperparams=hyperparams,
                    eval_env=eval_env,
                    eval_freq=100_000)
    
    return model_save_path

def computeExploitability(env: CustomCTF_v0, Policy):
    # NOTE: Input: Policy


    ### Step 1: Compute BestResponse.
    ### Step 2: Play the BestResponse against the input Policy via episodeSim and return score.
    ### Step 3: Return score for the input Policy pair and compute difference with Step 2.
    blue_model, red_model = Policy["Blue"], Policy["Red"]

    # Compute a BestResponse policy to the blue team.
    bluemodel_best_response, _ = BestResponse(env=env, Policy={"metadata": {"team": "Blue"}, "policy": blue_model}, num_million=2.5)
    redmodel_best_response, _ = BestResponse(env=env, Policy={"metadata": {"team": "Red"}, "policy": red_model}, num_million=2.5)
    return

ENV_REGISTORY = {"CustomCTF_v0": CustomCTF_v0,
                 "CustomCTF_v1": CustomCTF_v1,
                 "CustomCTF_v2": CustomCTF_v2,
                 "DefenseCTF": DefenseCTF,
                 "DefenseCTF_v2": DefenseCTF_v2,
                 "CoopEnv": CoopEnv,
                 "CoopEnv_v1": CoopEnv_v1
}

if __name__ == "__main__":
    """
    env = CustomCTF_v0(ctf_player_config="2v2")
    ###### TRY 3V3 OR 2V3. ==== ????????? HOW TO DEAL WITH DIFFERENT NUMBER OF NEIGHBOURING AGENTS FOR A SHARED POLICY BETWEEN THE TWO TEAMS? MAYBE NEAREST NEIGHBOURS - A CONSTANT NUMBER -- OR A GRAPH NEURAL NETWORK.
    ###### TRY different spaces instead of multidiscrete == use just boxes or discrete.

    #parallel_api_test(env, num_cycles=1000000)
    parallel_api_test(env, num_cycles=1000000)
    """
    device = 'cuda' if th.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(f"Using device: {device}")

    import supersuit as ss
    max_num_cycles = 100
    red_flag_location = (3,7)
    env = CustomCTF_v0(grid_size=8, ctf_player_config="2v2", max_num_cycles=max_num_cycles, red_flag_location=red_flag_location)
    parallel_env = env
    # Convert ParallelEnv to a Stable-Baselines3-compatible vectorized environment
    vec_env = ss.pettingzoo_env_to_vec_env_v1(parallel_env) # Independent PPO.
    vec_env = ss.concat_vec_envs_v1(vec_env, num_vec_envs=4, base_class="stable_baselines3") # This line is an important and necessary / required wrapper for the code to work.

    from stable_baselines3 import PPO
    model = PPO(
    policy=MaskedMultiInputPolicy,   # your custom policy class
    env=vec_env,            # your PettingZoo env, possibly converted to gym
    verbose=1,
    device=device
)
    model_path = 'ppo_CTF_2M_steps.zip'
    vec_env = ss.pettingzoo_env_to_vec_env_v1(parallel_env) # Independent PPO.
    vec_env = ss.concat_vec_envs_v1(vec_env, num_vec_envs=1, base_class="stable_baselines3")
    model = PPO.load(model_path, env=vec_env, device=device)
    print("model on device: {}".format(model.policy.parameters().__next__().device))
    
    model_path2 = 'ppo_CTF_20M_steps_BestResponse.zip'
    model2 = PPO.load(model_path2, env=vec_env, device=device)
    print("model2 on device: {}".format(model2.policy.parameters().__next__().device))

    model_path3 = 'ppo_CTF_10M_steps_BestResponse.zip'
    model3 = PPO.load(model_path3, env=vec_env, device=device)
    print("model3 on device: {}".format(model3.policy.parameters().__next__().device))

    model_best_response, model_best_response_save_path = BestResponse(env=env, Policy={"metadata": {"team": "Blue", "PolicyType":"mixture"}, "policy": {"networks": [model, model2, model3], "weights": np.array([0.3, 0.6, 0.1]) }}, num_million=0.9)
    print("===*40")
    print("BEST RESPONSE LEARNED...")
    print("===*40")

    """
    # MULTIPROCESSING FOR MULTIPLE SIMULTANEOUS BEST RESPONSE CALCULATION.
    from multiprocessing import Pool
    from functools import partial
    blue_policy = Policy={"metadata": {"team": "Blue"}, "policy": model}
    red_policy = Policy={"metadata": {"team": "Red"}, "policy": model}
    best_response_env = BestResponse(env=env)
    #args_ = [(env=env, Policy={"metadata": {"team": "Blue"}, "policy": model}, num_million=2), ]
    args_ = [(env, blue_policy, 2), (env, red_policy, 2)]

    import time
    t1 = time.time()
    with Pool(processes=2) as pool:
        BRs = pool.map(BestResponse, args_)
        pass
    t2 = time.time()
    print(t2-t1)
    print("Time taken for 2M parallel processes num: 2 = {} s".format(t2-t1))
    print("Time taken for {}M parallel processes num: {} = {}s".format(2, 2, t2-t1))
    """

    # Meeting tomorrow: 
    # Solve this Markov Game.
    # Complexity: imperfect knowledge of the game parameters such as the flag location: learn a diverse set of policies that is performant against the uncertainty in the game.

    ##### ORDER OF PROGRAMMING STUFF:
    # GET THE RESET AND STEP MODULES WORKING.
    # 

    ## RESEARCH:
    #1. CTF sim with IPPO / MAPPO: PettingZoo and RLlib
    #2. PSRO implementation
    #3. Eps,delta- PSRO: supposed to be more diverse
    #4. Quantifiers: diversity of the population; exploitbaility of the computed ensemble: so have an exploitability computation module.

    ### Immediate Todos:
    ## Get the sim working then integrate with rlllib then render then the sim loop.
    ### Once done then do the exploitability computation.

    #### OLD: PATH TO COMPLETION: IMPLEMENT A BASIC REWARD TO REACH GOAL: NO TAGGING AND NO RESPAWNING ETC. AND INTEGRATE WITH RLLIB AND THEN START IMPLEMENTING RENDER AND THEN ITERATE.

    ###### LATEST:
    # Software bug versus Algorithm bug.
    """
    ###### LATEST: April 28, 2025
    Large scale, perfect flag information
    Small scale, imperfect flag information
    Scaling axes: number of agents - large scale, small scale.. Partial observability full observability, large grid small grid (macro-actions)
    """
    import pettingzoo.mpe.simple_tag_v3 as simple_tag
    from pettingzoo.utils.conversions import aec_to_parallel
    import supersuit as ss
    from stable_baselines3 import PPO

    import functools
    import random
    import copy

    import numpy as np
    from gymnasium.spaces import Discrete, MultiDiscrete, Tuple
    from gymnasium.spaces import Dict, Box

    from pettingzoo import ParallelEnv
    from pettingzoo.test import parallel_api_test

    from matplotlib import pyplot as plt

    from stable_baselines3.ppo.policies import MultiInputPolicy
    import torch as th
    from torch import nn
    """
    class MaskedMultiInputPolicy(MultiInputPolicy):
        def forward(self, obs, deterministic=False):
            # 'obs' is a dict with 'obs' and 'action_mask'
            features = self.extract_features(obs)  # handles dict
            latent_pi, latent_vf = self.mlp_extractor(features)

            dist = self._get_action_dist_from_latent(latent_pi)
            # Extract the original logits from the distribution
            logits = dist.distribution.logits

            # Apply mask: set invalid action logits to a very negative value
            action_mask = obs["action_mask"]
            # Avoid log(0) by clamping values
            #mask = (action_mask + 1e-8).log()
            mask = (action_mask + np.exp(-np.inf)).log()
            masked_logits = logits + mask

            dist.distribution.logits = masked_logits
            actions = dist.get_actions(deterministic=deterministic)

            log_prob = dist.log_prob(actions)
            #return actions, None, log_prob

            values = self.value_net(latent_vf)
            return actions, values, log_prob
    """
    max_num_cycles = 100
    env = CustomCTF_v0(grid_size=8, ctf_player_config="2v2", max_num_cycles=max_num_cycles)
    parallel_env = env

    """
    # Pad observations & actions to ensure consistency across agents
    #parallel_env = ss.pad_observations_v0(env)
    #parallel_env = ss.pad_action_space_v0(env)
    """

    # Convert ParallelEnv to a Stable-Baselines3-compatible vectorized environment
    vec_env = ss.pettingzoo_env_to_vec_env_v1(parallel_env)
    vec_env = ss.concat_vec_envs_v1(vec_env, num_vec_envs=4, base_class="stable_baselines3")

    # Train a PPO model using Stable-Baselines3
    #model = PPO("MlpPolicy", vec_env, verbose=1)

    #model = PPO("MultiInputPolicy", vec_env, verbose=1)
    device = 'cuda' if th.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = PPO(
        policy=MaskedMultiInputPolicy,   # your custom policy class
        env=vec_env,            # your PettingZoo env, possibly converted to gym
        verbose=1,
        device=device
    )
    print(model.policy.parameters().__next__().device)
    """
    from stable_baselines3.common.callbacks import CheckpointCallback
    save_path = '/content/drive/My Drive/CTF_Checkpoints/'
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=save_path,
                                            name_prefix="ctf_model")
    """

    model_path = '/content/drive/My Drive/ppo_CTF_{}M_steps.pth'.format(2)
    model_path_current_session = "ppo_CTF_{}M_steps".format(2)
    load = False
    if load:
        model = PPO.load(model_path, env=vec_env)
        #model = PPO.load(model_path)
    
    import time
    learn = False
    num_million = 0.002 #5.5
    if learn:
        t1 = time.time()
        """
        # Pass the callback to the model's learn() method
        model.learn(total_timesteps=100000*2.5*num_million, callback=checkpoint_callback)
        """
        model.learn(total_timesteps=100000*2.5*num_million)
        t2 = time.time()
        # Save the model
        model.save("ppo_CTF_{}M_steps".format(num_million))
        print("="*40)
        print("Time taken to train {}M steps: {}s".format(num_million, t2-t1))
        print("="*40)
        pass