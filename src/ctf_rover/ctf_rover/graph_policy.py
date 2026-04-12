import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from stable_baselines3.common.policies import ActorCriticPolicy, BaseFeaturesExtractor
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, softmax
import networkx as nx

from stable_baselines3 import PPO
from customCTF import GraphCTF, GraphCoopEnv, GraphParallelEnvToSB3VecEnv_v1, make_env
from pettingzoo.test import parallel_api_test

from gymnasium.spaces import Box, Dict # if using Gymnasium
# ----------------------------
# 1. MPNN Layer with edge_attr
# ----------------------------
class MPNNLayer(MessagePassing):
    def __init__(self, node_in_dim, edge_in_dim, out_dim):
        super().__init__(aggr='add')  # sum aggregation
        self.mlp = nn.Sequential(
            nn.Linear(2*node_in_dim + edge_in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        """
        x: [N, F_node]
        edge_index: [2, E]
        edge_attr: [E, F_edge]
        """
        x = x.float()
        edge_attr = edge_attr.float()

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # pad edge_attr for self-loops if needed
        if edge_attr is not None:
            loop_attr = edge_attr.new_zeros((x.size(0), edge_attr.size(1)))
            edge_attr = torch.cat([edge_attr, loop_attr], dim=0)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i = target node, x_j = source node
        m = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp(m)

# ----------------------------
# 2. Graph Feature Extractor (batched)
# ----------------------------
class GraphCTFMPNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, embedding_dim=64):
        super().__init__(observation_space, features_dim=embedding_dim)
        node_feat_dim = observation_space['x'].shape[-1]
        edge_feat_dim = observation_space['edge_attr'].shape[-1]

        self.mpnn1 = MPNNLayer(node_feat_dim, edge_feat_dim, 64)
        #self.mpnn2 = MPNNLayer(64, edge_feat_dim, embedding_dim)
        self.mpnn2 = MPNNLayer(64, edge_feat_dim, 64)
        self.mpnn3 = MPNNLayer(64, edge_feat_dim, embedding_dim)

    def forward(self, obs):
        """
        obs['x']: [B, N, F_node]
        obs['edge_index']: [B, 2, E]
        obs['edge_attr']: [B, E, F_edge]
        obs['node_visibility_mask']: [B, N]
        """
        x_all = obs['x'].float()
        
        edge_attr_all = obs['edge_attr'].float()
        node_mask_all = obs['node_visibility_mask'].float()

        B, N, F_node = obs['x'].shape
        device = obs['x'].device
        all_embeddings = []

        for b in range(B):
            x = x_all[b] # [N, F_node]
            edge_index = obs['edge_index'][b].to(torch.long).to(device)  # [2, E]
            edge_attr = edge_attr_all[b] # [E, F_edge]
            node_mask = node_mask_all[b] # [N]

            h = F.relu(self.mpnn1(x, edge_index, edge_attr))
            h = F.relu(self.mpnn2(h, edge_index, edge_attr))
            h = F.relu(self.mpnn3(h, edge_index, edge_attr))

            # Zero-out invalid nodes
            h = h * node_mask.unsqueeze(-1)
            all_embeddings.append(h)

        return torch.stack(all_embeddings, dim=0)  # [B, N, hidden_dim]

# ----------------------------
# Neighbor-based Actor-Critic Policy (corrected)
# ----------------------------
class GraphCTFNeighborBatchPolicy(ActorCriticPolicy):
    """
    Actor-Critic policy for GraphCTF.
    Works with full-graph observations.
    Returns logits and values for **all agents in the batch**.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=GraphCTFMPNNExtractor,
            features_extractor_kwargs=dict(embedding_dim=64)
        )

        hidden_dim = 64
        self.extra_action_dim = 2  # stay + tag

        # Node logits
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Extra actions
        self.extra_action_head = nn.Linear(hidden_dim, self.extra_action_dim)

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

    # ----------------------------
    # Forward (used for action selection)
    # ----------------------------
    def forward(self, obs, deterministic=False):
        B, N, F_node = obs['x'].shape
        device = obs['x'].device

        # Node embeddings
        node_embeddings = self.features_extractor(obs)  # [B, N, D]

        # Agent node embedding
        agent_node_mask = obs['agent_node_mask']  # [B, N], one-hot
        agent_node_idx = agent_node_mask.argmax(dim=1)
        h_agent = node_embeddings[torch.arange(B, device=device), agent_node_idx]  # [B, D]

        # Node logits
        node_logits = self.node_mlp(node_embeddings).squeeze(-1)  # [B, N]

        # Extra actions
        extra_logits = self.extra_action_head(h_agent)  # [B, 2]

        # Full logits
        full_logits = torch.cat([node_logits, extra_logits], dim=-1)  # [B, N+2]

        # Apply action mask if given
        if 'action_mask' in obs and obs['action_mask'] is not None:
            mask = obs['action_mask'].to(torch.bool)
            if mask.shape[1] != full_logits.shape[1]:
                if mask.shape[1] < full_logits.shape[1]:
                    full_logits = full_logits[:, :mask.shape[1]]
                else:
                    pad = full_logits.new_full((B, mask.shape[1] - full_logits.shape[1]), float("-inf"))
                    full_logits = torch.cat([full_logits, pad], dim=-1)
            full_logits = full_logits.masked_fill(~mask, float("-inf"))

        # Distribution
        dist = Categorical(logits=full_logits)

        # Actions
        actions = torch.argmax(full_logits, dim=-1) if deterministic else dist.sample()
        log_probs = dist.log_prob(actions)

        # Values (from agent node)
        values = self.value_head(h_agent).squeeze(-1)  # [B]

        return actions, values, log_probs

    # ----------------------------
    # Predict values (used in rollout buffer)
    # ----------------------------
    def predict_values(self, obs):
        node_embeddings = self.features_extractor(obs)  # [B, N, D]

        agent_node_mask = obs['agent_node_mask']  # [B, N]
        agent_node_idx = agent_node_mask.argmax(dim=1)

        agent_emb = node_embeddings[torch.arange(node_embeddings.size(0), device=node_embeddings.device),
                                    agent_node_idx]  # [B, D]

        values = self.value_head(agent_emb).squeeze(-1)  # [B]
        return values

    # ----------------------------
    # Evaluate actions (used during PPO training)
    # ----------------------------
    def evaluate_actions(self, obs, actions):
        B, N, F_node = obs['x'].shape
        device = obs['x'].device

        # Node embeddings
        node_embeddings = self.features_extractor(obs)  # [B, N, D]

        # Agent node embedding
        agent_node_mask = obs['agent_node_mask']
        agent_node_idx = agent_node_mask.argmax(dim=1)
        h_agent = node_embeddings[torch.arange(B, device=device), agent_node_idx]  # [B, D]

        # Node logits
        node_logits = self.node_mlp(node_embeddings).squeeze(-1)  # [B, N]

        # Extra actions
        extra_logits = self.extra_action_head(h_agent)  # [B, 2]

        # Full logits
        full_logits = torch.cat([node_logits, extra_logits], dim=-1)  # [B, N+2]

        # Apply action mask
        if 'action_mask' in obs and obs['action_mask'] is not None:
            mask = obs['action_mask'].to(torch.bool)
            if mask.shape[1] != full_logits.shape[1]:
                if mask.shape[1] < full_logits.shape[1]:
                    full_logits = full_logits[:, :mask.shape[1]]
                else:
                    pad = full_logits.new_full((B, mask.shape[1] - full_logits.shape[1]), float("-inf"))
                    full_logits = torch.cat([full_logits, pad], dim=-1)
            full_logits = full_logits.masked_fill(~mask, float("-inf"))

        # Distribution
        dist = Categorical(logits=full_logits)

        # Values
        values = self.value_head(h_agent).squeeze(-1)  # [B]

        # Log-probabilities
        log_prob = dist.log_prob(actions)

        # Entropy
        entropy = dist.entropy()

        return values, log_prob, entropy

# ----------------------------
# 1. MPNN Layer (supports edge visibility mask)
# ----------------------------
class MPNNLayer_SubgraphCompatible(MessagePassing):
    def __init__(self, node_in_dim, edge_in_dim, out_dim, use_attention=False):
        super().__init__(aggr='add')  # sum aggregation
        self.use_attention = use_attention
        """
        self.mlp = nn.Sequential(
            nn.Linear(2*node_in_dim + edge_in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        """
        self.mlp = nn.Sequential(
            nn.Linear(2*node_in_dim + 0, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        if use_attention:
            # GAT-style: learned scalar attention weight per edge from cat[h_i, h_j]
            # Weights are then softmax-normalised over each target node's neighborhood.
            self.attn = nn.Linear(2 * node_in_dim, 1, bias=False)

    def forward(self, x, edge_index, edge_attr, edge_mask=None):
        """
        x: [N, F_node]
        edge_index: [2, E]
        edge_attr: [E, F_edge]
        edge_mask: [E], 0 for padded edges
        """
        x = x.float()
        if edge_attr is not None:
            edge_attr = edge_attr.float()
        else:
            edge_attr = torch.zeros((edge_index.size(1), 0), device=x.device)

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        if edge_attr.size(1) == 0:
        # Pad edge_attr for self-loops
            loop_attr = edge_attr.new_zeros((x.size(0), 0))
            edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

        if edge_mask is not None:
            loop_mask = edge_mask.new_ones(x.size(0))
            edge_mask = torch.cat([edge_mask, loop_mask], dim=0)
        else:
            edge_mask = None

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_mask=None)

    def message(self, x_i, x_j, edge_attr, edge_index_i, edge_mask=None):
        #m = torch.cat([x_i, x_j, edge_attr], dim=-1)
        m = torch.cat([x_i, x_j], dim=-1)
        msg = self.mlp(m)
        if self.use_attention:
            alpha = self.attn(m)                  # [E, 1]: unnormalised attention score
            alpha = softmax(alpha, edge_index_i)  # [E, 1]: normalised per target node
            msg = msg * alpha
        if edge_mask is not None:
            msg = msg * edge_mask.unsqueeze(-1)
        return msg

from torch_geometric.data import Data, Batch
def build_pyg_batch(obs):
    """
    obs: dict of torch tensors with leading batch dim
    """
    B = obs["x"].shape[0]
    data_list = []

    for b in range(B):
        node_mask = obs["node_visibility_mask"][b].bool()
        edge_mask = obs["edge_visibility_mask"][b].bool()

        x = obs["x"][b][node_mask]                         # [N_b, F]
        edge_index = obs["edge_index"][b][:, edge_mask]    # [2, E_b]
        edge_attr = obs["edge_attr"][b][edge_mask]         # [E_b, D_e]

        data = Data(
            x=torch.as_tensor(x, dtype=torch.float32),
            edge_index=torch.as_tensor(edge_index, dtype=torch.long),
            edge_attr=torch.as_tensor(edge_attr, dtype=torch.float32) if edge_attr is not None else None
        )

        data_list.append(data)

    return Batch.from_data_list(data_list)

# ----------------------------
# 2. Graph Feature Extractor (batched, subgraph compatible)
# ----------------------------
class GraphCTFMPNNExtractor_SubgraphCompatible(BaseFeaturesExtractor):
    def __init__(self, observation_space, embedding_dim=64, use_attention=False):
        super().__init__(observation_space, features_dim=embedding_dim)
        node_feat_dim = observation_space['x'].shape[-1]
        edge_feat_dim = observation_space.get('edge_attr', Box(low=-1, high=1, shape=(1,1))).shape[-1]

        self.mpnn1 = MPNNLayer_SubgraphCompatible(node_feat_dim, edge_feat_dim, 64, use_attention=use_attention)
        self.mpnn2 = MPNNLayer_SubgraphCompatible(64, edge_feat_dim, 64, use_attention=use_attention)
        self.mpnn3 = MPNNLayer_SubgraphCompatible(64, edge_feat_dim, embedding_dim, use_attention=use_attention)

    """
    def forward(self, obs):
        #obs keys:
         #   x: [B, N, F_node]
          #  edge_index: [B, 2, E]
           # edge_attr: [B, E, F_edge] (optional)
            #node_visibility_mask: [B, N]
            #edge_visibility_mask: [B, E]

        x_all = obs['x'].float()
        edge_attr_all = obs.get('edge_attr', None)
        node_mask_all = obs['node_visibility_mask'].float()
        edge_mask_all = obs.get('edge_visibility_mask', None)

        B, N, F_node = x_all.shape
        device = x_all.device
        embeddings = []
        masks = []

                for b in range(B):
            x = x_all[b]
            edge_index = obs['edge_index'][b].to(torch.long).to(device)
            edge_attr = edge_attr_all[b] if edge_attr_all is not None else None
            node_mask = node_mask_all[b]
            edge_mask = edge_mask_all[b] if edge_mask_all is not None else None

            num_nodes = int(node_mask.sum())
            x = x[:num_nodes]

            edge_mask = edge_mask.bool()
            edge_index = edge_index[:, edge_mask]
            #edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

            h = F.relu(self.mpnn1(x, edge_index, edge_attr))
            h = F.relu(self.mpnn2(h, edge_index, edge_attr))
            h = F.relu(self.mpnn3(h, edge_index, edge_attr))

            # Zero-out invalid nodes
            #h = h * node_mask.unsqueeze(-1)
            node_idx = node_mask.bool().nonzero(as_tuple=False).squeeze(-1)
            node_mask_sub = node_mask[node_idx] # optional: all ones now
            h = h * node_mask_sub.unsqueeze(-1) # safe

            embeddings.append(h)
            masks.append(node_mask[node_idx])

        max_nodes = max(h.shape[0] for h in embeddings)
        padded = []
        mask_tensor = []
        for h, m in zip(embeddings, masks):
            pad_len = max_nodes - h.shape[0]
            if pad_len > 0:
                h = torch.cat([h, torch.zeros(pad_len, h.shape[1], device=h.device)], dim=0)
                m = torch.cat([m, torch.zeros(pad_len, device=m.device)], dim=0)
        padded.append(h)
        mask_tensor.append(m)
        return torch.stack(padded, dim=0), torch.stack(mask_tensor, dim=0) #torch.stack(embeddings, dim=0)  # [B, N, embedding_dim]
    """
    def forward(self, obs):
    # 1. Build PyG batch (strips padding)
        pyg_batch = build_pyg_batch(obs)
        # pyg_batch.x          : [sum_b N_b, F]
        # pyg_batch.edge_index : [2, sum_b E_b]
        # pyg_batch.edge_attr  : [sum_b E_b, D_e]

        if pyg_batch.edge_index.dtype != torch.long:
            pyg_batch.edge_index = pyg_batch.edge_index.long()
        # 2. Stacked MPNN (fully parallel)
        h = pyg_batch.x

        h = self.mpnn1(h, pyg_batch.edge_index, pyg_batch.edge_attr)
        h = F.relu(h)

        h = self.mpnn2(h, pyg_batch.edge_index, pyg_batch.edge_attr)
        h = F.relu(h)

        h = self.mpnn3(h, pyg_batch.edge_index, pyg_batch.edge_attr)
        # h: [sum_b N_b, embedding_dim]

        return h, pyg_batch

# ----------------------------
# 3. Actor-Critic Policy (subgraph compatible)
# ----------------------------
class GraphCTFNeighborBatchPolicy_SubgraphCompatible(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=GraphCTFMPNNExtractor_SubgraphCompatible,
            features_extractor_kwargs=dict(embedding_dim=64)
        )

        hidden_dim = 64
        self.extra_action_dim = 2  # stay + tag

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.extra_action_head = nn.Linear(hidden_dim, self.extra_action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs, deterministic=False):
        B, N, _ = obs['x'].shape
        device = obs['x'].device

        node_embeddings = self.features_extractor(obs)  # [B, N, D]

        agent_node_mask = obs['agent_node_mask']  # [B, N]
        agent_node_idx = agent_node_mask.argmax(dim=1)
        h_agent = node_embeddings[torch.arange(B, device=device), agent_node_idx]  # [B, D]

        # Node logits
        node_logits = self.node_mlp(node_embeddings).squeeze(-1)  # [B, N]

        # Extra actions
        extra_logits = self.extra_action_head(h_agent)  # [B, 2]

        full_logits = torch.cat([node_logits, extra_logits], dim=-1)  # [B, N+2]

        # Apply action mask
        if 'action_mask' in obs and obs['action_mask'] is not None:
            mask = obs['action_mask'].to(torch.bool)
            if mask.shape[1] != full_logits.shape[1]:
                if mask.shape[1] < full_logits.shape[1]:
                    full_logits = full_logits[:, :mask.shape[1]]
                else:
                    pad = full_logits.new_full((B, mask.shape[1]-full_logits.shape[1]), float("-inf"))
                    full_logits = torch.cat([full_logits, pad], dim=-1)
            full_logits = full_logits.masked_fill(~mask, float("-inf"))

        dist = Categorical(logits=full_logits)
        actions = torch.argmax(full_logits, dim=-1) if deterministic else dist.sample()
        log_probs = dist.log_prob(actions)
        values = self.value_head(h_agent).squeeze(-1)  # [B]

        return actions, values, log_probs

    def predict_values(self, obs):
        node_embeddings = self.features_extractor(obs)
        agent_node_mask = obs['agent_node_mask']
        agent_node_idx = agent_node_mask.argmax(dim=1)
        agent_emb = node_embeddings[torch.arange(node_embeddings.size(0), device=node_embeddings.device),
                                    agent_node_idx]
        values = self.value_head(agent_emb).squeeze(-1)
        return values

    def evaluate_actions(self, obs, actions):
        B, N, _ = obs['x'].shape
        device = obs['x'].device

        node_embeddings = self.features_extractor(obs)
        agent_node_mask = obs['agent_node_mask']
        agent_node_idx = agent_node_mask.argmax(dim=1)
        h_agent = node_embeddings[torch.arange(B, device=device), agent_node_idx]

        node_logits = self.node_mlp(node_embeddings).squeeze(-1)
        extra_logits = self.extra_action_head(h_agent)
        full_logits = torch.cat([node_logits, extra_logits], dim=-1)

        # Apply action mask
        if 'action_mask' in obs and obs['action_mask'] is not None:
            mask = obs['action_mask'].to(torch.bool)
            if mask.shape[1] != full_logits.shape[1]:
                if mask.shape[1] < full_logits.shape[1]:
                    full_logits = full_logits[:, :mask.shape[1]]
                else:
                    pad = full_logits.new_full((B, mask.shape[1]-full_logits.shape[1]), float("-inf"))
                    full_logits = torch.cat([full_logits, pad], dim=-1)
            full_logits = full_logits.masked_fill(~mask, float("-inf"))

        dist = Categorical(logits=full_logits)
        values = self.value_head(h_agent).squeeze(-1)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return values, log_prob, entropy
    
class GraphCTFNeighborBatchPolicy_SubgraphCompatible_v1(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=GraphCTFMPNNExtractor_SubgraphCompatible,
            features_extractor_kwargs=dict(embedding_dim=64)
        )

        hidden_dim = 64
        self.extra_action_dim = 2  # stay + tag

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.extra_action_head = nn.Linear(hidden_dim, self.extra_action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs, deterministic=False):
        B = obs['x'].shape[0]
        device = obs['x'].device

        node_embeddings, mask_tensor = self.features_extractor(obs)  # [B, max_nodes, D], [B, max_nodes]

        # Get agent embedding
        agent_node_mask = obs['agent_node_mask']  # [B, N]
        # map agent_node_mask to padded embeddings
        max_nodes = node_embeddings.size(1)
        if agent_node_mask.size(1) < max_nodes:
            pad = agent_node_mask.new_zeros(B, max_nodes - agent_node_mask.size(1))
            agent_node_mask = torch.cat([agent_node_mask, pad], dim=1)

        agent_node_idx = agent_node_mask.argmax(dim=1)
        h_agent = node_embeddings[torch.arange(B, device=device), agent_node_idx]  # [B, D]

        # Node logits
        node_logits = self.node_mlp(node_embeddings).squeeze(-1)  # [B, max_nodes]
        # Mask padded nodes
        node_logits = node_logits.masked_fill(mask_tensor == 0, float("-inf"))

        # Extra actions
        extra_logits = self.extra_action_head(h_agent)  # [B, 2]

        full_logits = torch.cat([node_logits, extra_logits], dim=-1)  # [B, max_nodes + 2]

        # Apply action mask from environment
        if 'action_mask' in obs and obs['action_mask'] is not None:
            mask = obs['action_mask'].to(torch.bool)
            if mask.shape[1] != full_logits.shape[1]:
                if mask.shape[1] < full_logits.shape[1]:
                    full_logits = full_logits[:, :mask.shape[1]]
                else:
                    pad = full_logits.new_full((B, mask.shape[1]-full_logits.shape[1]), float("-inf"))
                    full_logits = torch.cat([full_logits, pad], dim=-1)
            full_logits = full_logits.masked_fill(~mask, float("-inf"))

        dist = Categorical(logits=full_logits)
        actions = torch.argmax(full_logits, dim=-1) if deterministic else dist.sample()
        log_probs = dist.log_prob(actions)
        values = self.value_head(h_agent).squeeze(-1)  # [B]

        return actions, values, log_probs

    def predict_values(self, obs):
        node_embeddings, mask_tensor = self.features_extractor(obs)
        B = node_embeddings.size(0)
        device = node_embeddings.device

        agent_node_mask = obs['agent_node_mask']
        max_nodes = node_embeddings.size(1)
        if agent_node_mask.size(1) < max_nodes:
            pad = agent_node_mask.new_zeros(B, max_nodes - agent_node_mask.size(1))
            agent_node_mask = torch.cat([agent_node_mask, pad], dim=1)

        agent_node_idx = agent_node_mask.argmax(dim=1)
        agent_emb = node_embeddings[torch.arange(B, device=device), agent_node_idx]
        values = self.value_head(agent_emb).squeeze(-1)
        return values

    def evaluate_actions(self, obs, actions):
        B = obs['x'].shape[0]
        device = obs['x'].device

        node_embeddings, mask_tensor = self.features_extractor(obs)

        agent_node_mask = obs['agent_node_mask']
        max_nodes = node_embeddings.size(1)
        if agent_node_mask.size(1) < max_nodes:
            pad = agent_node_mask.new_zeros(B, max_nodes - agent_node_mask.size(1))
            agent_node_mask = torch.cat([agent_node_mask, pad], dim=1)

        agent_node_idx = agent_node_mask.argmax(dim=1)
        h_agent = node_embeddings[torch.arange(B, device=device), agent_node_idx]

        node_logits = self.node_mlp(node_embeddings).squeeze(-1)
        node_logits = node_logits.masked_fill(mask_tensor == 0, float("-inf"))
        extra_logits = self.extra_action_head(h_agent)
        full_logits = torch.cat([node_logits, extra_logits], dim=-1)

        # Apply action mask
        if 'action_mask' in obs and obs['action_mask'] is not None:
            mask = obs['action_mask'].to(torch.bool)
            if mask.shape[1] != full_logits.shape[1]:
                if mask.shape[1] < full_logits.shape[1]:
                    full_logits = full_logits[:, :mask.shape[1]]
                else:
                    pad = full_logits.new_full((B, mask.shape[1]-full_logits.shape[1]), float("-inf"))
                    full_logits = torch.cat([full_logits, pad], dim=-1)
            full_logits = full_logits.masked_fill(~mask, float("-inf"))

        dist = Categorical(logits=full_logits)
        values = self.value_head(h_agent).squeeze(-1)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return values, log_prob, entropy


import torch
import torch.nn as nn
from torch.distributions import Categorical
from stable_baselines3.common.policies import ActorCriticPolicy

class GraphCTFNeighborBatchPolicy_SubgraphCompatible_v2(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=GraphCTFMPNNExtractor_SubgraphCompatible,
            features_extractor_kwargs=dict(embedding_dim=64)
        )

        hidden_dim = 64
        self.extra_action_dim = 2  # stay + tag

        # Node MLP for agent node logits
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Extra action logits
        self.extra_action_head = nn.Linear(hidden_dim, self.extra_action_dim)

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs, deterministic=False):
        device = obs['x'].device

        # --- 1. Node embeddings via MPNN ---
        node_h, pyg_batch = self.features_extractor(obs)  # node_h: [total_nodes, D]
        ptr = pyg_batch.ptr                                  # [B+1]
        B = ptr.size(0) - 1
        D = node_h.size(-1)

        # --- 2. Agent node indices in flattened batch ---
        agent_local_idx = obs['agent_node_local_idx'].long().to(device)
        if agent_local_idx.dim() > 1:
            agent_local_idx = agent_local_idx.squeeze(-1)  # [B, 1] -> [B]
        agent_global_idx = ptr[:-1] + agent_local_idx  # [B]

        # --- 3. Agent node embeddings ---
        h_agent = node_h[agent_global_idx]  # [B, D]

        # --- 4. Get neighbor local indices and convert to global indices ---
        neighbor_local_idx = obs['neighbor_local_idx'].long().to(device)  # [B, max_degree]
        neighbor_mask = obs['neighbor_mask'].to(device)                    # [B, max_degree]
        max_degree = neighbor_local_idx.size(1)

        # Convert local neighbor indices to global indices in flattened node_h
        # neighbor_global_idx[b, i] = ptr[b] + neighbor_local_idx[b, i]
        neighbor_global_idx = ptr[:-1].unsqueeze(1) + neighbor_local_idx  # [B, max_degree]

        # Clamp to valid range to avoid indexing errors (masked positions will be ignored anyway)
        neighbor_global_idx = neighbor_global_idx.clamp(0, node_h.size(0) - 1)

        # --- 5. Extract neighbor embeddings ---
        neighbor_embeddings = node_h[neighbor_global_idx]  # [B, max_degree, D]

        # --- 6. Compute neighbor logits ---
        neighbor_logits = self.node_mlp(neighbor_embeddings).squeeze(-1)  # [B, max_degree]

        # Mask out invalid neighbors
        neighbor_logits = neighbor_logits.masked_fill(neighbor_mask == 0, float('-inf'))

        # --- 7. Extra actions (stay + tag) ---
        extra_logits = self.extra_action_head(h_agent)  # [B, extra_action_dim]

        # --- 8. Concatenate neighbor logits + extra actions ---
        full_logits = torch.cat([neighbor_logits, extra_logits], dim=-1)  # [B, max_degree + extra_action_dim]

        # --- 9. Apply action mask from environment ---
        if 'action_mask' in obs and obs['action_mask'] is not None:
            mask = obs['action_mask'].to(torch.bool).to(device)
            A = full_logits.shape[1]

            if mask.shape[1] > A:
                mask = mask[:, :A]
            elif mask.shape[1] < A:
                pad = torch.zeros(B, A - mask.shape[1], dtype=torch.bool, device=device)
                mask = torch.cat([mask, pad], dim=-1)

            full_logits = full_logits.masked_fill(~mask, float('-inf'))

        # --- 10. Sample / argmax ---
        dist = Categorical(logits=full_logits)
        actions = dist.sample() if not deterministic else torch.argmax(full_logits, dim=-1)
        log_probs = dist.log_prob(actions)

        # --- 11. Values ---
        values = self.value_head(h_agent).squeeze(-1)  # [B]

        return actions, values, log_probs

    def predict_values(self, obs):
        device = obs['x'].device
        node_h, pyg_batch = self.features_extractor(obs)
        ptr = pyg_batch.ptr
        agent_local_idx = obs['agent_node_local_idx'].long().to(device)
        if agent_local_idx.dim() > 1:
            agent_local_idx = agent_local_idx.squeeze(-1)
        agent_global_idx = ptr[:-1] + agent_local_idx
        h_agent = node_h[agent_global_idx]
        values = self.value_head(h_agent).squeeze(-1)
        return values

    def evaluate_actions(self, obs, actions):
        device = obs['x'].device
        node_h, pyg_batch = self.features_extractor(obs)
        ptr = pyg_batch.ptr
        B = ptr.size(0) - 1

        agent_local_idx = obs['agent_node_local_idx'].long().to(device)
        if agent_local_idx.dim() > 1:
            agent_local_idx = agent_local_idx.squeeze(-1)
        agent_global_idx = ptr[:-1] + agent_local_idx
        h_agent = node_h[agent_global_idx]

        # Get neighbor local indices and convert to global indices
        neighbor_local_idx = obs['neighbor_local_idx'].long().to(device)  # [B, max_degree]
        neighbor_mask = obs['neighbor_mask'].to(device)                    # [B, max_degree]
        max_degree = neighbor_local_idx.size(1)

        neighbor_global_idx = ptr[:-1].unsqueeze(1) + neighbor_local_idx  # [B, max_degree]
        neighbor_global_idx = neighbor_global_idx.clamp(0, node_h.size(0) - 1)

        # Extract neighbor embeddings and compute logits
        neighbor_embeddings = node_h[neighbor_global_idx]  # [B, max_degree, D]
        neighbor_logits = self.node_mlp(neighbor_embeddings).squeeze(-1)  # [B, max_degree]
        neighbor_logits = neighbor_logits.masked_fill(neighbor_mask == 0, float('-inf'))

        extra_logits = self.extra_action_head(h_agent)  # [B, extra_action_dim]
        full_logits = torch.cat([neighbor_logits, extra_logits], dim=-1)

        # Masking
        if 'action_mask' in obs and obs['action_mask'] is not None:
            mask = obs['action_mask'].to(torch.bool).to(device)
            A = full_logits.shape[1]
            if mask.shape[1] > A:
                mask = mask[:, :A]
            elif mask.shape[1] < A:
                pad = torch.zeros(B, A - mask.shape[1], dtype=torch.bool, device=device)
                mask = torch.cat([mask, pad], dim=-1)
            full_logits = full_logits.masked_fill(~mask, float('-inf'))

        dist = Categorical(logits=full_logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value_head(h_agent).squeeze(-1)
        return values, log_prob, entropy


import torch
import torch.nn as nn
from torch.distributions import Categorical
from stable_baselines3.common.policies import ActorCriticPolicy


class GraphCTFNeighborBatchPolicy_SubgraphCompatible_v4(ActorCriticPolicy):
    """
    GNN Actor-Critic policy with two architectural improvements over v2:

    1. Relational actor scoring: neighbor logits are computed from
       cat[h_agent, h_neighbor] via a shared scoring_mlp, instead of
       scoring each neighbor independently with node_mlp(h_neighbor).
       h_agent now receives gradient from the policy loss, not just the
       value loss — making the GNN backbone train on both signals.

    2. Global pooling critic: the value head receives
       cat[h_agent, mean_pool(all_nodes), max_pool(all_nodes)] instead
       of h_agent alone. mean_pool captures average map state;
       max_pool captures worst-case threat signal. Together they give
       the critic a full-graph view regardless of local neighborhood size.
    """

    def __init__(self, *args, use_attention: bool = False, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=GraphCTFMPNNExtractor_SubgraphCompatible,
            features_extractor_kwargs=dict(embedding_dim=64, use_attention=use_attention)
        )

        hidden_dim = 64
        self.hidden_dim = hidden_dim
        self.extra_action_dim = 2  # stay + tag

        # Relational scoring: cat[h_agent, h_neighbor] -> scalar logit
        self.scoring_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Extra action logits (stay + tag) conditioned on agent state only
        self.extra_action_head = nn.Linear(hidden_dim, self.extra_action_dim)

        # Value head: cat[h_agent, mean_pool, max_pool] -> scalar
        # Input dim = 3 * hidden_dim
        self.value_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _get_embeddings(self, obs):
        """Run MPNN and extract all per-sample tensors needed by all three forward methods."""
        device = obs['x'].device
        node_h, pyg_batch = self.features_extractor(obs)   # node_h: [sumN, D]
        ptr = pyg_batch.ptr                                  # [B+1]
        B   = ptr.size(0) - 1

        agent_local_idx = obs['agent_node_local_idx'].long().to(device)
        if agent_local_idx.dim() > 1:
            agent_local_idx = agent_local_idx.squeeze(-1)
        agent_global_idx = ptr[:-1] + agent_local_idx       # [B]
        h_agent          = node_h[agent_global_idx]         # [B, D]

        neighbor_local_idx  = obs['neighbor_local_idx'].long().to(device)  # [B, max_deg]
        neighbor_mask       = obs['neighbor_mask'].to(device)              # [B, max_deg]
        neighbor_global_idx = ptr[:-1].unsqueeze(1) + neighbor_local_idx  # [B, max_deg]
        neighbor_global_idx = neighbor_global_idx.clamp(0, node_h.size(0) - 1)
        neighbor_embeddings = node_h[neighbor_global_idx]                  # [B, max_deg, D]

        return node_h, pyg_batch, B, device, h_agent, neighbor_embeddings, neighbor_mask

    def _compute_neighbor_logits(self, h_agent, neighbor_embeddings, neighbor_mask):
        """Relational scoring: logit_k = scoring_mlp(cat[h_agent, h_neighbor_k])."""
        max_degree  = neighbor_embeddings.size(1)
        h_agent_exp = h_agent.unsqueeze(1).expand(-1, max_degree, -1)   # [B, max_deg, D]
        pair        = torch.cat([h_agent_exp, neighbor_embeddings], dim=-1)  # [B, max_deg, 2D]
        logits      = self.scoring_mlp(pair).squeeze(-1)                 # [B, max_deg]
        return logits.masked_fill(neighbor_mask == 0, float('-inf'))

    def _compute_values(self, node_h, pyg_batch, h_agent):
        """Critic with global mean+max pooling over the full observed subgraph."""
        mean_g = global_mean_pool(node_h, pyg_batch.batch)          # [B, D]
        max_g  = global_max_pool(node_h, pyg_batch.batch)           # [B, D]
        v_in   = torch.cat([h_agent, mean_g, max_g], dim=-1)        # [B, 3*D]
        return self.value_head(v_in).squeeze(-1)                     # [B]

    def _apply_mask(self, full_logits, obs, B, device):
        if 'action_mask' not in obs or obs['action_mask'] is None:
            return full_logits
        mask = obs['action_mask'].to(torch.bool).to(device)
        A = full_logits.shape[1]
        if mask.shape[1] > A:
            mask = mask[:, :A]
        elif mask.shape[1] < A:
            pad = torch.zeros(B, A - mask.shape[1], dtype=torch.bool, device=device)
            mask = torch.cat([mask, pad], dim=-1)
        return full_logits.masked_fill(~mask, float('-inf'))

    # ------------------------------------------------------------------
    # ActorCriticPolicy interface
    # ------------------------------------------------------------------

    def forward(self, obs, deterministic=False):
        node_h, pyg_batch, B, device, h_agent, neighbor_embeddings, neighbor_mask = \
            self._get_embeddings(obs)

        neighbor_logits = self._compute_neighbor_logits(h_agent, neighbor_embeddings, neighbor_mask)
        extra_logits    = self.extra_action_head(h_agent)
        full_logits     = torch.cat([neighbor_logits, extra_logits], dim=-1)
        full_logits     = self._apply_mask(full_logits, obs, B, device)

        dist      = Categorical(logits=full_logits)
        actions   = dist.sample() if not deterministic else torch.argmax(full_logits, dim=-1)
        log_probs = dist.log_prob(actions)
        values    = self._compute_values(node_h, pyg_batch, h_agent)

        return actions, values, log_probs

    def predict_values(self, obs):
        node_h, pyg_batch, B, device, h_agent, _, _ = self._get_embeddings(obs)
        return self._compute_values(node_h, pyg_batch, h_agent)

    def evaluate_actions(self, obs, actions):
        node_h, pyg_batch, B, device, h_agent, neighbor_embeddings, neighbor_mask = \
            self._get_embeddings(obs)

        neighbor_logits = self._compute_neighbor_logits(h_agent, neighbor_embeddings, neighbor_mask)
        extra_logits    = self.extra_action_head(h_agent)
        full_logits     = torch.cat([neighbor_logits, extra_logits], dim=-1)
        full_logits     = self._apply_mask(full_logits, obs, B, device)

        dist     = Categorical(logits=full_logits)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        values   = self._compute_values(node_h, pyg_batch, h_agent)

        return values, log_prob, entropy


class GraphCTFNeighborBatchPolicy_SubgraphCompatible_v4_hybrid(ActorCriticPolicy):
    """
    Hybrid policy: v2 actor + v4 critic.

    Motivation
    ----------
    v4's relational actor scoring — scoring_mlp(cat[h_agent, h_neighbor]) — routes
    h_agent through every movement logit.  This couples the actor tightly to the
    critic via h_agent: critic gradient noise (especially high during the weaning
    stage when the value function must re-learn from sparse rewards) flows directly
    into all action logits, causing chronic training instability across the full
    curriculum (confirmed empirically: v4 collapses from Stage 1, Stage 4 worse
    than Stage 5, all-timeout even vs PassiveRed).

    Fix
    ---
    Actor : node_mlp(h_neighbor)   — v2 design, h_agent NOT used in neighbor logits.
            h_agent gradient comes only from extra_action_head (stay/tag), keeping
            the backbone shielded from policy-loss interference on movement.
    Critic: value_head(cat[h_agent, mean_pool, max_pool])  — v4 design.
            Global graph context improves credit assignment without touching actor.

    MPNN backbone: GraphCTFMPNNExtractor_SubgraphCompatible with optional GAT-style
    edge-level attention (use_attention=True).  Attention lives inside the message-
    passing layers — it is completely orthogonal to the relational actor scoring that
    was removed, so it is retained here.
    """

    def __init__(self, *args, use_attention: bool = False, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=GraphCTFMPNNExtractor_SubgraphCompatible,
            features_extractor_kwargs=dict(embedding_dim=64, use_attention=use_attention),
        )

        hidden_dim = 64
        self.hidden_dim = hidden_dim
        self.extra_action_dim = 2  # stay + tag

        # Actor — independent neighbor scoring (v2 style; no h_agent dependency)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Extra action logits (stay + tag) conditioned on agent state
        self.extra_action_head = nn.Linear(hidden_dim, self.extra_action_dim)

        # Critic — global pooling (v4 style): cat[h_agent, mean_pool, max_pool]
        self.value_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------
    # Shared helper
    # ------------------------------------------------------------------

    def _get_embeddings(self, obs):
        """Run MPNN and return tensors needed by all three forward methods."""
        device = obs['x'].device
        node_h, pyg_batch = self.features_extractor(obs)   # [sumN, D]
        ptr = pyg_batch.ptr                                  # [B+1]
        B   = ptr.size(0) - 1

        agent_local_idx = obs['agent_node_local_idx'].long().to(device)
        if agent_local_idx.dim() > 1:
            agent_local_idx = agent_local_idx.squeeze(-1)
        agent_global_idx = ptr[:-1] + agent_local_idx       # [B]
        h_agent          = node_h[agent_global_idx]          # [B, D]

        neighbor_local_idx  = obs['neighbor_local_idx'].long().to(device)   # [B, max_deg]
        neighbor_mask       = obs['neighbor_mask'].to(device)               # [B, max_deg]
        neighbor_global_idx = (ptr[:-1].unsqueeze(1) + neighbor_local_idx)  # [B, max_deg]
        neighbor_global_idx = neighbor_global_idx.clamp(0, node_h.size(0) - 1)
        neighbor_embeddings = node_h[neighbor_global_idx]                   # [B, max_deg, D]

        return node_h, pyg_batch, B, device, h_agent, neighbor_embeddings, neighbor_mask

    def _compute_logits(self, neighbor_embeddings, neighbor_mask, h_agent, obs, B, device):
        """v2-style independent neighbor scoring; h_agent used only for extra actions."""
        neighbor_logits = self.node_mlp(neighbor_embeddings).squeeze(-1)    # [B, max_deg]
        neighbor_logits = neighbor_logits.masked_fill(neighbor_mask == 0, float('-inf'))

        extra_logits = self.extra_action_head(h_agent)                      # [B, 2]
        full_logits  = torch.cat([neighbor_logits, extra_logits], dim=-1)

        if 'action_mask' in obs and obs['action_mask'] is not None:
            mask = obs['action_mask'].to(torch.bool).to(device)
            A = full_logits.shape[1]
            if mask.shape[1] > A:
                mask = mask[:, :A]
            elif mask.shape[1] < A:
                pad = torch.zeros(B, A - mask.shape[1], dtype=torch.bool, device=device)
                mask = torch.cat([mask, pad], dim=-1)
            full_logits = full_logits.masked_fill(~mask, float('-inf'))

        return full_logits

    def _compute_values(self, node_h, pyg_batch, h_agent):
        """v4-style global pooling critic."""
        mean_g = global_mean_pool(node_h, pyg_batch.batch)     # [B, D]
        max_g  = global_max_pool(node_h, pyg_batch.batch)      # [B, D]
        v_in   = torch.cat([h_agent, mean_g, max_g], dim=-1)   # [B, 3D]
        return self.value_head(v_in).squeeze(-1)                # [B]

    # ------------------------------------------------------------------
    # ActorCriticPolicy interface
    # ------------------------------------------------------------------

    def forward(self, obs, deterministic=False):
        node_h, pyg_batch, B, device, h_agent, neighbor_embeddings, neighbor_mask = \
            self._get_embeddings(obs)

        full_logits = self._compute_logits(
            neighbor_embeddings, neighbor_mask, h_agent, obs, B, device)

        dist      = Categorical(logits=full_logits)
        actions   = dist.sample() if not deterministic else torch.argmax(full_logits, dim=-1)
        log_probs = dist.log_prob(actions)
        values    = self._compute_values(node_h, pyg_batch, h_agent)

        return actions, values, log_probs

    def predict_values(self, obs):
        node_h, pyg_batch, B, device, h_agent, _, _ = self._get_embeddings(obs)
        return self._compute_values(node_h, pyg_batch, h_agent)

    def evaluate_actions(self, obs, actions):
        node_h, pyg_batch, B, device, h_agent, neighbor_embeddings, neighbor_mask = \
            self._get_embeddings(obs)

        full_logits = self._compute_logits(
            neighbor_embeddings, neighbor_mask, h_agent, obs, B, device)

        dist     = Categorical(logits=full_logits)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        values   = self._compute_values(node_h, pyg_batch, h_agent)

        return values, log_prob, entropy


import torch
import torch.nn as nn
from torch.distributions import Categorical
from stable_baselines3.common.policies import ActorCriticPolicy

class GraphCTFNeighborBatchPolicy_SubgraphCompatible_v3(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=GraphCTFMPNNExtractor_SubgraphCompatible,
            features_extractor_kwargs=dict(embedding_dim=64)
        )

        hidden_dim = 64
        self.extra_action_dim = 2  # stay + tag

        # MLP for agent node logits
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Extra action logits
        self.extra_action_head = nn.Linear(hidden_dim, self.extra_action_dim)

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs, deterministic=False):
        device = obs['x'].device

        # --- 1. Node embeddings via MPNN ---
        node_h, pyg_batch = self.features_extractor(obs)  # node_h: [total_nodes, D]
        ptr = pyg_batch.ptr                                  # [B+1]
        B = ptr.size(0) - 1

        # --- 2. Compute agent global indices in flattened batch ---
        agent_local_idx = obs['agent_node_local_idx'].long().to(device)  # [B]
        agent_global_idx = ptr[:-1] + agent_local_idx                     # [B]

        # --- 3. Select agent node embeddings ---
        h_agent = node_h[agent_global_idx]  # [B, D]

        # --- 4. Agent node logits ---
        agent_node_logits = self.node_mlp(h_agent).squeeze(-1)  # [B]

        # --- 5. Extra actions ---
        extra_logits = self.extra_action_head(h_agent)          # [B, extra_action_dim]

        # --- 6. Concatenate agent node + extra actions ---
        full_logits = torch.cat([agent_node_logits.unsqueeze(-1), extra_logits], dim=-1)  # [B, 1 + extra_action_dim]

        # --- 7. Apply action mask safely ---
        if 'action_mask' in obs and obs['action_mask'] is not None:
            mask = obs['action_mask'].to(torch.bool).to(device)  # [B, 1 + extra_action_dim]

            # Ensure mask matches full_logits shape
            if mask.shape[1] > full_logits.shape[1]:
                mask = mask[:, :full_logits.shape[1]]
            elif mask.shape[1] < full_logits.shape[1]:
                pad = torch.zeros(B, full_logits.shape[1]-mask.shape[1], dtype=torch.bool, device=device)
                mask = torch.cat([mask, pad], dim=-1)

            full_logits = full_logits.masked_fill(~mask, float('-inf'))

        # --- 8. Sample / argmax ---
        dist = Categorical(logits=full_logits)
        actions = dist.sample() if not deterministic else torch.argmax(full_logits, dim=-1)
        log_probs = dist.log_prob(actions)

        # --- 9. Values ---
        values = self.value_head(h_agent).squeeze(-1)  # [B]

        return actions, values, log_probs

    def predict_values(self, obs):
        device = obs['x'].device
        node_h, pyg_batch = self.features_extractor(obs)
        ptr = pyg_batch.ptr
        agent_local_idx = obs['agent_node_local_idx'].long().to(device)
        agent_global_idx = ptr[:-1] + agent_local_idx
        h_agent = node_h[agent_global_idx]
        values = self.value_head(h_agent).squeeze(-1)
        return values

    def evaluate_actions(self, obs, actions):
        device = obs['x'].device
        node_h, pyg_batch = self.features_extractor(obs)
        ptr = pyg_batch.ptr
        B = ptr.size(0) - 1

        agent_local_idx = obs['agent_node_local_idx'].long().to(device)
        agent_global_idx = ptr[:-1] + agent_local_idx
        h_agent = node_h[agent_global_idx]

        agent_node_logits = self.node_mlp(h_agent).squeeze(-1)  # [B]
        extra_logits = self.extra_action_head(h_agent)          # [B, extra_action_dim]
        full_logits = torch.cat([agent_node_logits.unsqueeze(-1), extra_logits], dim=-1)

        # Masking
        if 'action_mask' in obs and obs['action_mask'] is not None:
            mask = obs['action_mask'].to(torch.bool).to(device)
            if mask.shape[1] > full_logits.shape[1]:
                mask = mask[:, :full_logits.shape[1]]
            elif mask.shape[1] < full_logits.shape[1]:
                pad = torch.zeros(B, full_logits.shape[1]-mask.shape[1], dtype=torch.bool, device=device)
                mask = torch.cat([mask, pad], dim=-1)
            full_logits = full_logits.masked_fill(~mask, float('-inf'))

        dist = Categorical(logits=full_logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value_head(h_agent).squeeze(-1)
        return values, log_prob, entropy


import torch
import torch.nn as nn
from torch.distributions import Categorical
from stable_baselines3.common.policies import ActorCriticPolicy

class GraphCTFNeighborBatchPolicy_SubgraphCompatible_PyG(ActorCriticPolicy):
    """
    Action space: move-to-node (up to max_nodes per ego graph) + extra actions.
    Expects env to provide action_mask of shape [B, max_nodes + extra_action_dim] (or [max_nodes + extra_action_dim]).
    Expects extractor to return: node_h [sumN, D], pyg_batch with ptr [B+1].
    Expects obs to provide agent_node_local_idx [B] (local index of ego/agent node in each subgraph).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=GraphCTFMPNNExtractor_SubgraphCompatible,
            features_extractor_kwargs=dict(embedding_dim=64),
        )

        hidden_dim = 64
        self.extra_action_dim = 2  # stay + tag

        # score each node embedding -> scalar logit
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # extra actions from agent embedding
        self.extra_action_head = nn.Linear(hidden_dim, self.extra_action_dim)

        # value from agent embedding
        self.value_head = nn.Linear(hidden_dim, 1)

        # If your env uses a fixed padded node-action space (e.g., 8), store it here:
        # We infer at runtime from action_mask or observation_space if possible.
        self._cached_max_nodes = None

    def _infer_max_nodes(self, obs, ptr, B, device):
        """
        Determine max_nodes used in the padded node-action head.
        Priority:
          1) from action_mask width (A) minus extra_action_dim
          2) from max nodes in the current batch (max(ptr diff))
          3) cached value
        """
        # 1) from action_mask
        if "action_mask" in obs and obs["action_mask"] is not None:
            m = obs["action_mask"]
            if m.dim() == 1:
                A = int(m.shape[0])
            else:
                A = int(m.shape[-1])
            max_nodes = A - self.extra_action_dim
            if max_nodes > 0:
                self._cached_max_nodes = max_nodes
                return max_nodes

        # 2) from current batch max nodes (variable, but still ok)
        max_nodes = int((ptr[1:] - ptr[:-1]).max().item())
        if self._cached_max_nodes is None:
            self._cached_max_nodes = max_nodes
        return self._cached_max_nodes

    def _make_2d_mask(self, obs, B, A, device):
        """
        Returns boolean mask of shape [B, A] (or None).
        Pads/slices safely.
        """
        if "action_mask" not in obs or obs["action_mask"] is None:
            return None

        mask = obs["action_mask"].to(device)

        # Convert to bool robustly
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)

        # Ensure 2D [B, A]
        if mask.dim() == 1:
            # [A] -> [B, A]
            mask = mask.unsqueeze(0).expand(B, -1)
        elif mask.dim() == 2:
            pass
        else:
            # If something accidentally becomes 3D, flatten last dim only if it makes sense
            # but in your case it should never be 3D.
            raise RuntimeError(f"action_mask must be 1D or 2D, got shape={tuple(mask.shape)}")

        # Fix width
        if mask.shape[1] > A:
            mask = mask[:, :A]
        elif mask.shape[1] < A:
            pad = torch.zeros((B, A - mask.shape[1]), dtype=torch.bool, device=device)
            mask = torch.cat([mask, pad], dim=1)

        return mask

    def forward(self, obs, deterministic=False):
        device = obs["x"].device

        # Extract PyG-batched node embeddings
        node_h, pyg_batch = self.features_extractor(obs)  # node_h: [sumN, D]
        ptr = pyg_batch.ptr  # [B+1]
        B = ptr.numel() - 1

        # agent local indices (fallback to 0 if not provided)
        if "agent_node_local_idx" in obs:
            agent_local = obs["agent_node_local_idx"].to(device).long()
            if agent_local.dim() == 0:
                agent_local = agent_local.unsqueeze(0).expand(B)
        else:
            agent_local = torch.zeros((B,), dtype=torch.long, device=device)

        # sanity: agent_local within each graph size
        num_nodes_per_graph = ptr[1:] - ptr[:-1]
        if torch.any(agent_local < 0) or torch.any(agent_local >= num_nodes_per_graph):
            raise RuntimeError(
                f"agent_node_local_idx out of bounds. "
                f"agent_local={agent_local.detach().cpu().tolist()} "
                f"num_nodes_per_graph={num_nodes_per_graph.detach().cpu().tolist()}"
            )

        agent_global = ptr[:-1] + agent_local  # [B]
        h_agent = node_h[agent_global]         # [B, D]

        # Infer padded node-action width
        max_nodes = self._infer_max_nodes(obs, ptr, B, device)  # e.g., 8
        A = max_nodes + self.extra_action_dim                  # e.g., 10

        # ---- Node logits (padded to [B, max_nodes]) ----
        flat_node_logits = self.node_mlp(node_h).squeeze(-1)   # [sumN]
        node_logits = node_h.new_full((B, max_nodes), float("-inf"))  # [B, max_nodes]

        # Fill per-graph logits into padded tensor
        # (This loop is cheap; you cannot avoid ragged->rect conversion for SB3.)
        for b in range(B):
            start = int(ptr[b].item())
            end = int(ptr[b + 1].item())
            k = min(end - start, max_nodes)
            if k > 0:
                node_logits[b, :k] = flat_node_logits[start:start + k]

        # ---- Extra logits from agent embedding ----
        extra_logits = self.extra_action_head(h_agent)         # [B, extra_action_dim]

        # ---- Concatenate -> [B, A] ----
        full_logits = torch.cat([node_logits, extra_logits], dim=1)

        # HARD SHAPE ASSERTS (prevents the CUDA-side garbage)
        if full_logits.dim() != 2:
            raise RuntimeError(f"full_logits must be 2D [B,A], got shape={tuple(full_logits.shape)}")
        if full_logits.shape[0] != B:
            raise RuntimeError(f"full_logits batch mismatch: got {full_logits.shape[0]} vs B={B}")
        if full_logits.shape[1] != A:
            raise RuntimeError(f"full_logits width mismatch: got {full_logits.shape[1]} vs A={A}")

        # ---- Apply action mask (must be [B, A]) ----
        mask = self._make_2d_mask(obs, B, A, device)
        if mask is not None:
            if mask.shape != full_logits.shape:
                raise RuntimeError(
                    f"Mask/logits shape mismatch: mask={tuple(mask.shape)} logits={tuple(full_logits.shape)}"
                )
            full_logits = full_logits.masked_fill(~mask, float("-inf"))

        # ---- Distribution/sample ----
        dist = Categorical(logits=full_logits)
        actions = torch.argmax(full_logits, dim=-1) if deterministic else dist.sample()
        log_probs = dist.log_prob(actions)

        # ---- Value ----
        values = self.value_head(h_agent).squeeze(-1)  # [B]

        return actions, values, log_probs

    def predict_values(self, obs):
        device = obs["x"].device
        node_h, pyg_batch = self.features_extractor(obs)
        ptr = pyg_batch.ptr
        B = ptr.numel() - 1

        if "agent_node_local_idx" in obs:
            agent_local = obs["agent_node_local_idx"].to(device).long()
            if agent_local.dim() == 0:
                agent_local = agent_local.unsqueeze(0).expand(B)
        else:
            agent_local = torch.zeros((B,), dtype=torch.long, device=device)

        agent_global = ptr[:-1] + agent_local
        h_agent = node_h[agent_global]
        return self.value_head(h_agent).squeeze(-1)

    def evaluate_actions(self, obs, actions):
        device = obs["x"].device
        node_h, pyg_batch = self.features_extractor(obs)
        ptr = pyg_batch.ptr
        B = ptr.numel() - 1

        if "agent_node_local_idx" in obs:
            agent_local = obs["agent_node_local_idx"].to(device).long()
            if agent_local.dim() == 0:
                agent_local = agent_local.unsqueeze(0).expand(B)
        else:
            agent_local = torch.zeros((B,), dtype=torch.long, device=device)

        agent_global = ptr[:-1] + agent_local
        h_agent = node_h[agent_global]

        max_nodes = self._infer_max_nodes(obs, ptr, B, device)
        A = max_nodes + self.extra_action_dim

        flat_node_logits = self.node_mlp(node_h).squeeze(-1)   # [sumN]
        node_logits = node_h.new_full((B, max_nodes), float("-inf"))
        for b in range(B):
            start = int(ptr[b].item())
            end = int(ptr[b + 1].item())
            k = min(end - start, max_nodes)
            if k > 0:
                node_logits[b, :k] = flat_node_logits[start:start + k]

        extra_logits = self.extra_action_head(h_agent)
        full_logits = torch.cat([node_logits, extra_logits], dim=1)

        mask = self._make_2d_mask(obs, B, A, device)
        if mask is not None:
            full_logits = full_logits.masked_fill(~mask, float("-inf"))

        dist = Categorical(logits=full_logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value_head(h_agent).squeeze(-1)
        return values, log_prob, entropy


# =============================================================================
# Policy Registry — version resolution and safe loading
# =============================================================================

POLICY_REGISTRY = {
    "v2":         GraphCTFNeighborBatchPolicy_SubgraphCompatible_v2,
    "v4":         GraphCTFNeighborBatchPolicy_SubgraphCompatible_v4,
    "v4_hybrid":  GraphCTFNeighborBatchPolicy_SubgraphCompatible_v4_hybrid,
}

# Versions that accept a use_attention kwarg
_ATTENTION_VERSIONS = {"v4", "v4_hybrid"}


def resolve_policy(version: str = "v4_hybrid", use_attention: bool = False,
                   mappo: bool = False):
    """Return (policy_class, policy_kwargs) for the requested version.

    Args:
        version:      "v2", "v4", or "v4_hybrid"  (default "v4_hybrid")
        use_attention: Enable GAT-style edge attention inside MPNN layers.
                       Honoured for v4 and v4_hybrid; silently ignored for v2.
        mappo:        If True, return the MAPPO variant of the backbone
                      (GraphCTFMAPPOPolicy / _v4_hybrid / _v2).
                      Requires GraphParallelEnvToSB3VecEnv_MAPPO as the vec env.

    Returns:
        (policy_class, policy_kwargs_dict)
        Pass policy_kwargs_dict as PPO(..., policy_kwargs=policy_kwargs_dict).
    """
    # MAPPO_POLICY_REGISTRY is defined later in this module (after the MAPPO
    # class definitions) and is resolved at call-time, not import-time.
    registry = MAPPO_POLICY_REGISTRY if mappo else POLICY_REGISTRY
    cls = registry.get(version)
    if cls is None:
        raise ValueError(
            f"Unknown policy version '{version}' (mappo={mappo}). "
            f"Available: {list(registry)}"
        )
    policy_kwargs = {}
    if version in _ATTENTION_VERSIONS:
        policy_kwargs["use_attention"] = use_attention
    return cls, policy_kwargs


def load_policy_safe(path: str, device=None, **load_kwargs):
    """Load a PPO checkpoint with automatic version fallback.

    Tries in order:
      1. PPO.load() with the class stored inside the zip (default SB3 behaviour).
         This works for any checkpoint regardless of version.
      2. If that fails, force-load with v4 class (state_dict strict=False).
      3. If that fails, force-load with v2 class (state_dict strict=False).

    Args:
        path:        Path to .zip file (with or without extension).
        device:      'cuda', 'cpu', or None (auto).
        **load_kwargs: Extra kwargs forwarded to PPO.load() (e.g. env=, learning_rate=).

    Returns:
        Loaded PPO model.
    """
    from stable_baselines3 import PPO as _PPO

    # Strip .zip if present — SB3 appends it internally, so passing it here
    # would produce .zip.zip.
    lp = path[:-4] if path.endswith('.zip') else path
    dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Attempt 1: auto-detect from zip (covers both v2 and v4 saved with SB3)
    _e_auto = None
    try:
        return _PPO.load(lp, device=dev, **load_kwargs)
    except Exception as e:
        _e_auto = e

    # Attempt 2–6: try all known policy classes in priority order
    _fallback_classes = [
        ("mappo_v4_hybrid", GraphCTFMAPPOPolicy_v4_hybrid),
        ("mappo_v4",        GraphCTFMAPPOPolicy),
        ("mappo_v2",        GraphCTFMAPPOPolicy_v2),
        ("ippo_v4",         GraphCTFNeighborBatchPolicy_SubgraphCompatible_v4),
        ("ippo_v2",         GraphCTFNeighborBatchPolicy_SubgraphCompatible_v2),
    ]
    _errors = [f"  auto: {_e_auto}"]
    for label, cls in _fallback_classes:
        try:
            return _PPO.load(lp, device=dev, custom_objects={"policy_class": cls}, **load_kwargs)
        except Exception as e:
            _errors.append(f"  {label}: {e}")

    raise RuntimeError(
        f"load_policy_safe: could not load '{lp}' with any known policy class.\n"
        + "\n".join(_errors)
    )


import networkx as nx
import numpy as np
from customCTF import make_seeded_rngs

def episode_graph_sim(graph_ctf_env, red_team_policy, blue_team_policy, seed=0):
    assert callable(red_team_policy) and callable(blue_team_policy)
    np.random.seed(seed=seed)
    env = graph_ctf_env
    graph = env.graph
    blue_flag_node = env.blue_flag_node
    red_flag_node = env.red_flag_node
    max_num_cycles = env.max_num_cycles
    
    pos = nx.get_node_attributes(graph, 'pos')

    agents = env.agents
    red_team_agents = env.red_team_agents
    blue_team_agents = env.blue_team_agents
    obs, info = env.reset()

    frames = []
    red_team_nodes, blue_team_nodes = [env.state[agent] for agent in red_team_agents], [env.state[agent] for agent in blue_team_agents]
    fig, ax = draw_scene(graph, blue_flag_node, red_flag_node, red_team_nodes, blue_team_nodes)
    frames.append((fig, ax))
    done = False

    while not done:
        actions_dict = {agent: None for agent in agents}
        for agent in red_team_agents: actions_dict[agent] = red_team_policy(obs[agent])
        for agent in blue_team_agents: actions_dict[agent] = blue_team_policy(obs[agent])
        
        obs, rewards, terminations, truncations, infos = env.step(actions_dict)
        for agent in agents:
            if terminations[agent] or truncations[agent]:
                done = True

        red_team_nodes, blue_team_nodes = [env.state[agent] for agent in red_team_agents], [env.state[agent] for agent in blue_team_agents]
        fig, ax = draw_scene(graph, blue_flag_node, red_flag_node, red_team_nodes, blue_team_nodes)
        frames.append((fig, ax))

    return frames
    
# 1. Shortest path heuristic. # Cooperative -> BR (1M or 2M). # Render.

### RENDERING CODE ###
def draw_graph(G, pos=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))

    pos_dict = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos=pos_dict, node_size=20, edge_color='black')

    return fig, ax

def draw_scene(G, blue_flag_node, red_flag_node, red_team_nodes, blue_team_nodes, visibility_radius=1.5):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    red_agents, blue_agents = [0, 1], [0, 1]

    fig, ax = draw_graph(G)
    pos = nx.get_node_attributes(G, 'pos')

    from matplotlib.patches import Circle
    for agent_iter, agent in enumerate(red_agents):
        red_team_pos = pos[red_team_nodes[agent_iter]]
        ax.scatter(*red_team_pos, color="red", s=35, zorder=6)
        circle = Circle(
            red_team_pos,          # (x, y)
            radius=visibility_radius,
            facecolor="red",  # fill color
            edgecolor="red",  # outline color (optional)
            alpha=0.3,          # transparency (0 = invisible, 1 = opaque)
            linewidth=2
        )
        ax.add_patch(circle)

    for agent_iter, agent in enumerate(blue_agents):
        blue_team_pos = pos[blue_team_nodes[agent_iter]]
        ax.scatter(*blue_team_pos, color="blue", s=35, zorder=6)
        circle = Circle(
            blue_team_pos,          # (x, y)
            radius=visibility_radius,
            facecolor="blue",  # fill color
            edgecolor="blue",  # outline color (optional)
            alpha=0.3,          # transparency (0 = invisible, 1 = opaque)
            linewidth=2
        )
        ax.add_patch(circle)

    blue_flag_img_path="flag_imgs/blue_flag.png"
    red_flag_img_path="flag_imgs/red_flag.png"
    blue_flag_loc, red_flag_loc = pos[blue_flag_node], pos[red_flag_node]
    if blue_flag_loc is not None:
        blue_flag_img = mpimg.imread(blue_flag_img_path)
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        blue_flag_img = OffsetImage(blue_flag_img, zoom=0.12)
        ab_blue = AnnotationBbox(blue_flag_img, blue_flag_loc, frameon=False)
        ax.add_artist(ab_blue)

    if red_flag_loc is not None:
        red_flag_img = mpimg.imread(red_flag_img_path)
        red_flag_img = OffsetImage(red_flag_img, zoom=0.12)
        ab_red = AnnotationBbox(red_flag_img, red_flag_loc, frameon=False)
        ax.add_artist(ab_red)
    plt.xlim(-1, +11) # (-2, +12)
    plt.ylim(-1, +11)
    plt.show()
    return fig, ax


# =============================================================================
# Learned Policy Wrapper for Batched Opponent Inference
# =============================================================================

# Keys injected as teammate_* obs for MAPPO policies.
# Mirrors _TEAMMATE_COPY_KEYS in customCTF.py — duplicated here to avoid a
# circular import (customCTF imports graph_policy).
_MAPPO_TEAMMATE_KEYS = (
    'x', 'node_visibility_mask', 'edge_visibility_mask',
    'edge_index', 'edge_attr', 'agent_node_local_idx',
)


class LearnedPolicyWrapper:
    """
    Wraps a PPO model (or path) to provide batch_action() for efficient opponent inference.

    This wrapper enables batched neural network inference when used as an opponent policy
    in GraphCTF.set_opponent_policy(). Instead of running N forward passes for N opponents,
    it stacks observations and runs a single batched forward pass.

    Usage:
        # From a saved model
        wrapper = LearnedPolicyWrapper("GNN_PPO_BlueGraphPolicy_subgraph_v2.zip")

        # From a PPO instance
        wrapper = LearnedPolicyWrapper(ppo_model)

        # Set as opponent
        env.set_opponent_policy(wrapper, active_team='Red')

        # Single inference
        action = wrapper(obs_dict)

        # Batched inference (called automatically by GraphCTF._compute_opponent_actions)
        actions = wrapper.batch_action([obs1, obs2, obs3])
    """

    def __init__(self, model_or_path, device=None, deterministic=True,
                 observation_space=None, obs_version=3):
        """
        Args:
            model_or_path: PPO model instance or path to saved .zip file
            device: torch device ('cuda', 'cpu', or None for auto)
            deterministic: If True, use argmax; if False, sample from distribution
            observation_space: Required if loading from path (gym.spaces.Dict)
            obs_version: Observation version for the dummy env (default 3)
        """
        import zipfile
        import io

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.deterministic = deterministic

        """
        if isinstance(model_or_path, str):
            # Load weights directly to avoid cloudpickle issues
            self.model = None  # We don't need the full PPO model for inference

            # Create a fresh policy instance
            if observation_space is None:
                # Create a dummy env to get observation space
                from customCTF import GraphCTF
                dummy_env = GraphCTF(ctf_player_config="2v2", obs_version=obs_version)
                observation_space = dummy_env.observation_space(dummy_env.agents[0])
                dummy_env.close()

            from gymnasium.spaces import Discrete

            # Infer action space from saved weights
            with zipfile.ZipFile(model_or_path, 'r') as z:
                with z.open('policy.pth') as f:
                    state_dict = torch.load(io.BytesIO(f.read()), map_location=device)

            n_actions = state_dict['action_net.weight'].shape[0]
            action_space = Discrete(n_actions)

            self.policy = GraphCTFNeighborBatchPolicy_SubgraphCompatible_v2(
                observation_space=observation_space,
                action_space=action_space,
                lr_schedule=lambda _: 0.0003,  # Dummy, not used
            )

            self.policy.load_state_dict(state_dict)

            self.policy.to(device)
        else:
            self.model = model_or_path
            self.policy = self.model.policy
        """
        if isinstance(model_or_path, str):
            self.model = load_policy_safe(model_or_path, device=device)
        else:
            self.model = model_or_path

        self.policy = self.model.policy
        self.policy.eval()

        # Auto-detect MAPPO policy by presence of any per-agent actor head_1.
        # GraphCTFMAPPOPolicy uses scoring_mlp_1; v4_hybrid and v2 use node_mlp_1.
        # When True, batch_action() injects agent_id + teammate_* before forward()
        # so each agent is routed through its correct per-agent head.
        self._is_mappo = hasattr(self.policy, 'scoring_mlp_1') or hasattr(self.policy, 'node_mlp_1')
        if self._is_mappo:
            print(f"  [LearnedPolicyWrapper] MAPPO policy detected — "
                  f"teammate obs injection enabled for opponent inference.")

    def _obs_to_tensors(self, obs_dict, device):
        """Convert single observation dict to tensor dict."""
        tensor_obs = {}
        for key, val in obs_dict.items():
            if isinstance(val, np.ndarray):
                tensor_obs[key] = torch.as_tensor(val, device=device).unsqueeze(0)
            elif isinstance(val, torch.Tensor):
                tensor_obs[key] = val.unsqueeze(0).to(device)
            else:
                # Scalar
                tensor_obs[key] = torch.tensor([val], device=device)
        return tensor_obs

    def _stack_obs_list(self, obs_list, device):
        """Stack list of observation dicts into batched tensor dict."""
        if not obs_list:
            return {}

        keys = obs_list[0].keys()
        batched = {}

        for key in keys:
            vals = []
            for obs in obs_list:
                val = obs[key]
                if isinstance(val, np.ndarray):
                    vals.append(torch.as_tensor(val, device=device))
                elif isinstance(val, torch.Tensor):
                    vals.append(val.to(device))
                else:
                    vals.append(torch.tensor(val, device=device))

            # Stack along batch dimension
            batched[key] = torch.stack(vals, dim=0)

        return batched

    def _inject_mappo_obs(self, obs_list):
        """Inject agent_id and teammate_* into each obs for MAPPO policy inference.

        GraphCTF._compute_opponent_actions() builds obs_list in stable passive-agent
        order — [obs_agent0, obs_agent1] — matching get_passive_agents() ordering.
        Agent i receives:
          - agent_id = np.array([i % 2], dtype=int64)   (team-local index)
          - teammate_{key} = obs_list[i ^ 1][key]        (the other agent's obs)

        This mirrors GraphParallelEnvToSB3VecEnv_MAPPO._stack_agent_obs() but
        operates on raw obs dicts rather than pre-stacked arrays.  Keys are taken
        from _MAPPO_TEAMMATE_KEYS (mirrors _TEAMMATE_COPY_KEYS in customCTF.py).

        If obs_list has only one element (single-agent query), agent_id=0 is
        injected and no teammate obs are added — phi_tm falls back to zeros inside
        the MAPPO critic, which is acceptable for single-agent inference.
        """
        n = len(obs_list)
        augmented = []
        for i, obs in enumerate(obs_list):
            aug = dict(obs)  # shallow copy — only adds new keys, never mutates values
            aug['agent_id'] = np.array([i % 2], dtype=np.int64)
            tm_idx = i ^ 1   # 0 ↔ 1; for a single-obs list, 1 >= n so no teammate added
            if tm_idx < n:
                tm_obs = obs_list[tm_idx]
                for key in _MAPPO_TEAMMATE_KEYS:
                    if key in tm_obs:
                        aug[f'teammate_{key}'] = tm_obs[key]
            augmented.append(aug)
        return augmented

    def __call__(self, obs):
        """
        Single observation inference.

        Args:
            obs: Observation dict (numpy arrays)

        Returns:
            action: int
        """
        with torch.no_grad():
            tensor_obs = self._obs_to_tensors(obs, self.device)
            actions, _, _ = self.policy.forward(tensor_obs, deterministic=self.deterministic)
            return int(actions.cpu().item())

    def batch_action(self, obs_list):
        """Batched inference for multiple observations.

        For MAPPO policies (auto-detected at construction via presence of
        scoring_mlp_1), injects agent_id and teammate_* obs before the forward
        pass so each agent is routed through its correct per-agent actor head.
        obs_list must be in stable passive-agent order: [obs_agent0, obs_agent1].

        Without this injection, _get_agent_id() inside the MAPPO policy falls
        back to zeros(B), causing every agent to use scoring_mlp_0 — effectively
        IPPO rollouts on a MAPPO policy.

        Args:
            obs_list: List of observation dicts (in agent order for MAPPO)

        Returns:
            List of actions (ints)
        """
        if not obs_list:
            return []

        if self._is_mappo:
            obs_list = self._inject_mappo_obs(obs_list)

        with torch.no_grad():
            batched_obs = self._stack_obs_list(obs_list, self.device)
            actions, _, _ = self.policy.forward(batched_obs, deterministic=self.deterministic)
            return actions.cpu().tolist()

    def set_deterministic(self, deterministic):
        """Toggle deterministic/stochastic action selection."""
        self.deterministic = deterministic

    def to(self, device):
        """Move policy to different device."""
        self.device = device
        self.policy.to(device)
        return self


def load_opponent_policy(path_or_model, device=None, deterministic=True):
    """
    Convenience function to load a learned policy as an opponent.

    Args:
        path_or_model: Path to .zip file or PPO model instance
        device: 'cuda', 'cpu', or None (auto)
        deterministic: Use greedy actions if True

    Returns:
        LearnedPolicyWrapper instance ready for set_opponent_policy()
    """
    return LearnedPolicyWrapper(path_or_model, device=device, deterministic=deterministic)


# =========================================================
# MAPPO Policy — DeepSets centralized critic + per-agent actor heads
# =========================================================

class GraphCTFMAPPOPolicy(GraphCTFNeighborBatchPolicy_SubgraphCompatible_v4):
    """
    MAPPO extension of v4 with:
      - Per-agent actor heads  (scoring_mlp_0/1, extra_action_head_0/1)
        Heterogeneous actors sharing the GNN backbone.
      - DeepSets centralized critic:
          phi_ego = φ(cat[h_ego, mean_ego, max_ego])
          phi_tm  = φ(cat[h_tm,  mean_tm,  max_tm ])   (same φ weights)
          V       = ρ(phi_ego + phi_tm)                  ← permutation invariant
        Requires teammate_* obs keys from GraphParallelEnvToSB3VecEnv_MAPPO.
      - Optional per-agent GRU (use_gru=True, gru_hidden_dim=128):
          h_actor = GRU_i(h_gnn, h_prev)  (i=0 or 1, NOT shared across agents)
          Stateful during rollout (self.training=False): h_buf persists across
          timesteps; reset on episode done via reset_hidden(indices).
          Stateless during gradient update (self.training=True): h0=zeros —
          avoids stale gradients from the flattened SB3 rollout buffer.
          Teammate path in critic always uses stateless GRU (_apply_gru_stateless)
          to avoid conflicting writes to h_buf.
    """

    _GRU_H = 128  # GRU hidden size (fixed; change here if needed)

    def __init__(self, *args, use_attention=False, use_gru=False, **kwargs):
        super().__init__(*args, use_attention=use_attention, **kwargs)
        D = self.hidden_dim  # 64
        self.use_gru = use_gru
        self._h_buf  = None   # [B, GRU_H] persistent hidden state; lazy-init on first forward

        # Per-agent GRUs (not shared across agents)
        if use_gru:
            self.gru_0 = nn.GRUCell(input_size=D, hidden_size=self._GRU_H)
            self.gru_1 = nn.GRUCell(input_size=D, hidden_size=self._GRU_H)
            A = self._GRU_H   # actor/critic head input dim = 128
        else:
            A = D             # 64 (no GRU)

        # Per-agent actor heads (heterogeneous actors, shared GNN backbone)
        # scoring_mlp input: cat[h_actor[A], neighbor_emb[D]] → A+D
        self.scoring_mlp_0 = nn.Sequential(
            nn.Linear(A + D, A), nn.ReLU(), nn.Linear(A, 1)
        )
        self.scoring_mlp_1 = nn.Sequential(
            nn.Linear(A + D, A), nn.ReLU(), nn.Linear(A, 1)
        )
        self.extra_action_head_0 = nn.Linear(A, self.extra_action_dim)
        self.extra_action_head_1 = nn.Linear(A, self.extra_action_dim)

        # DeepSets critic: φ (per-agent embedding) + ρ (team value)
        # phi_net input: cat[h_actor[A], mean_pool[D], max_pool[D]] → A+2D
        self.phi_net = nn.Sequential(
            nn.Linear(A + 2 * D, A), nn.ReLU(), nn.Linear(A, A)
        )
        self.rho_net = nn.Sequential(
            nn.Linear(A, A), nn.ReLU(), nn.Linear(A, 1)
        )

        # Remove inherited v4 single-agent heads (unused in MAPPO)
        del self.scoring_mlp
        del self.extra_action_head
        del self.value_head

    # ------------------------------------------------------------------
    # GRU helper
    # ------------------------------------------------------------------

    def _apply_gru(self, h: torch.Tensor, agent_id: torch.Tensor) -> torch.Tensor:
        """
        Per-agent GRUCell with temporal memory.

        Collection (self.training=False): reads self._h_buf as h0 and writes the
        new hidden state back, providing real cross-step temporal memory.
        Training (self.training=True): h0=zeros — avoids stale-gradient issues
        from the flattened SB3 rollout buffer while still learning recurrent weights.

        Args:
            h:        [B, D]  GNN node embedding
            agent_id: [B]     int tensor (0 or 1)
        Returns:
            h_out: [B, GRU_H=128]
        """
        B, device = h.size(0), h.device

        if not self.training:
            # Stateful: use / update persistent hidden buffer
            if self._h_buf is None or self._h_buf.shape[0] != B or self._h_buf.device != device:
                self._h_buf = torch.zeros(B, self._GRU_H, device=device)
            h0 = self._h_buf.detach()   # stop gradient through stored state
        else:
            # Gradient update: zero h0 (approximate BPTT)
            h0 = h.new_zeros(B, self._GRU_H)

        h_gru_0 = self.gru_0(h, h0)                              # [B, GRU_H]
        h_gru_1 = self.gru_1(h, h0)                              # [B, GRU_H]
        sel      = (agent_id == 0).unsqueeze(-1)                  # [B, 1] bool
        h_new    = torch.where(sel, h_gru_0, h_gru_1)            # [B, GRU_H]

        if not self.training:
            self._h_buf = h_new.detach()  # persist for next timestep

        return h_new

    def reset_hidden(self, indices=None) -> None:
        """
        Zero GRU hidden state for the given batch indices (or all rows if None).
        Call from a training callback whenever episode done flags are True.

        Args:
            indices: array-like int | None — rows of self._h_buf to zero.
        """
        if self._h_buf is None:
            return
        if indices is None:
            self._h_buf.zero_()
        else:
            self._h_buf[indices] = 0.0

    def _apply_gru_stateless(self, h: torch.Tensor, agent_id: torch.Tensor) -> torch.Tensor:
        """
        Always-stateless GRU (h0=zeros). Used exclusively for teammate embeddings
        in the centralized critic — we don't maintain a separate h_buf for teammate
        positions, so keeping it stateless avoids incorrect h_buf writes.
        """
        h0      = h.new_zeros(h.size(0), self._GRU_H)
        h_gru_0 = self.gru_0(h, h0)
        h_gru_1 = self.gru_1(h, h0)
        sel     = (agent_id == 0).unsqueeze(-1)
        return torch.where(sel, h_gru_0, h_gru_1)

    # ------------------------------------------------------------------
    # MAPPO-specific helpers
    # ------------------------------------------------------------------

    def _compute_actor_logits(self, h_actor, neighbor_embeddings, neighbor_mask, agent_id):
        """
        Route through per-agent scoring MLPs based on agent_id.

        Args:
            h_actor:             [B, A]  (A=GRU_H if use_gru else D)
            neighbor_embeddings: [B, max_deg, D]
            neighbor_mask:       [B, max_deg]  (1=valid, 0=padded)
            agent_id:            [B]  int tensor (0 or 1)
        Returns:
            nbr_logits:   [B, max_deg]          (padded entries set to -inf)
            extra_logits: [B, extra_action_dim]
        """
        max_deg = neighbor_embeddings.size(1)
        h_exp   = h_actor.unsqueeze(1).expand(-1, max_deg, -1)         # [B, max_deg, A]
        pair    = torch.cat([h_exp, neighbor_embeddings], dim=-1)       # [B, max_deg, A+D]

        nbr_0 = self.scoring_mlp_0(pair).squeeze(-1)                    # [B, max_deg]
        nbr_1 = self.scoring_mlp_1(pair).squeeze(-1)
        ext_0 = self.extra_action_head_0(h_actor)                       # [B, extra_dim]
        ext_1 = self.extra_action_head_1(h_actor)

        sel          = (agent_id == 0).to(h_actor.device)               # [B] bool
        nbr_logits   = torch.where(sel.unsqueeze(-1), nbr_0, nbr_1)    # [B, max_deg]
        nbr_logits   = nbr_logits.masked_fill(neighbor_mask == 0, float('-inf'))
        extra_logits = torch.where(sel.unsqueeze(-1), ext_0, ext_1)    # [B, extra_dim]
        return nbr_logits, extra_logits

    def _get_teammate_embeddings(self, obs):
        """Run the shared GNN on the teammate's ego-subgraph observation."""
        tm_obs = {
            'x':                    obs['teammate_x'],
            'node_visibility_mask': obs['teammate_node_visibility_mask'],
            'edge_index':           obs['teammate_edge_index'],
            'edge_attr':            obs['teammate_edge_attr'],
            'edge_visibility_mask': obs['teammate_edge_visibility_mask'],
        }
        return self.features_extractor(tm_obs)  # (node_h_tm [sumN, D], pyg_batch_tm)

    def _compute_values(self, node_h, pyg_batch, h_actor, obs=None, agent_id=None):
        """
        DeepSets centralized critic: V = ρ(φ_ego + φ_tm).

        When use_gru=True, the teammate's h_tm is also routed through its own
        per-agent GRU (1 - agent_id) before entering phi_net, keeping the
        phi_net input dimensionality consistent and preserving permutation
        invariance (φ_ego + φ_tm is the same sum regardless of which agent
        is "ego").

        Falls back to V = ρ(φ_ego + 0) when teammate obs is absent.
        """
        mean_s  = global_mean_pool(node_h, pyg_batch.batch)                    # [B, D]
        max_s   = global_max_pool(node_h, pyg_batch.batch)                     # [B, D]
        phi_ego = self.phi_net(torch.cat([h_actor, mean_s, max_s], dim=-1))    # [B, A]

        if obs is not None and 'teammate_x' in obs:
            node_h_tm, pyg_batch_tm = self._get_teammate_embeddings(obs)
            tm_local = obs['teammate_agent_node_local_idx'].long()
            if tm_local.dim() > 1:
                tm_local = tm_local.squeeze(-1)
            h_tm    = node_h_tm[pyg_batch_tm.ptr[:-1] + tm_local]              # [B, D]
            mean_tm = global_mean_pool(node_h_tm, pyg_batch_tm.batch)
            max_tm  = global_max_pool(node_h_tm, pyg_batch_tm.batch)

            if self.use_gru and agent_id is not None:
                tm_id  = 1 - agent_id                                           # [B]
                h_tm   = self._apply_gru_stateless(h_tm, tm_id)                # [B, GRU_H]

            phi_tm  = self.phi_net(torch.cat([h_tm, mean_tm, max_tm], dim=-1)) # [B, A]
        else:
            phi_tm = torch.zeros_like(phi_ego)  # IPPO / no-teammate fallback

        team_state = phi_ego + phi_tm                                           # [B, A] — invariant
        return self.rho_net(team_state).squeeze(-1)                             # [B]

    # ------------------------------------------------------------------
    # ActorCriticPolicy interface
    # ------------------------------------------------------------------

    def _get_agent_id(self, obs, B, device):
        """Extract agent_id tensor [B] from obs, defaulting to zeros."""
        if 'agent_id' in obs:
            return obs['agent_id'].long().squeeze(-1)
        return torch.zeros(B, dtype=torch.long, device=device)

    def forward(self, obs, deterministic=False):
        node_h, pyg_batch, B, device, h_agent, neighbor_embeddings, neighbor_mask = \
            self._get_embeddings(obs)

        agent_id = self._get_agent_id(obs, B, device)
        h_actor  = self._apply_gru(h_agent, agent_id) if self.use_gru else h_agent

        nbr_logits, extra_logits = self._compute_actor_logits(
            h_actor, neighbor_embeddings, neighbor_mask, agent_id
        )
        full_logits = torch.cat([nbr_logits, extra_logits], dim=-1)
        full_logits = self._apply_mask(full_logits, obs, B, device)

        dist      = Categorical(logits=full_logits)
        actions   = dist.sample() if not deterministic else torch.argmax(full_logits, dim=-1)
        log_probs = dist.log_prob(actions)
        values    = self._compute_values(node_h, pyg_batch, h_actor, obs, agent_id)
        return actions, values, log_probs

    def predict_values(self, obs):
        node_h, pyg_batch, B, device, h_agent, _, _ = self._get_embeddings(obs)
        agent_id = self._get_agent_id(obs, B, device)
        h_actor  = self._apply_gru(h_agent, agent_id) if self.use_gru else h_agent
        return self._compute_values(node_h, pyg_batch, h_actor, obs, agent_id)

    def evaluate_actions(self, obs, actions):
        node_h, pyg_batch, B, device, h_agent, neighbor_embeddings, neighbor_mask = \
            self._get_embeddings(obs)

        agent_id = self._get_agent_id(obs, B, device)
        h_actor  = self._apply_gru(h_agent, agent_id) if self.use_gru else h_agent

        nbr_logits, extra_logits = self._compute_actor_logits(
            h_actor, neighbor_embeddings, neighbor_mask, agent_id
        )
        full_logits = torch.cat([nbr_logits, extra_logits], dim=-1)
        full_logits = self._apply_mask(full_logits, obs, B, device)

        dist     = Categorical(logits=full_logits)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        values   = self._compute_values(node_h, pyg_batch, h_actor, obs, agent_id)
        return values, log_prob, entropy


# =========================================================
# MAPPO variants for v2 and v4_hybrid backbones
# (GraphCTFMAPPOPolicy above is unchanged — fully backward-compatible)
# =========================================================

class _MAPPOIndependentActorMixin:
    """
    Shared MAPPO logic for independent-actor variants (v2 / v4_hybrid backbones).

    Actor style: node_mlp_i(h_nbr)  — neighbor scoring does NOT use h_actor,
    shielding the GNN backbone from policy-loss gradient through movement logits
    (same rationale as v4_hybrid over v4 for IPPO).

    Critic: DeepSets V = ρ(φ_ego + φ_tm)  — identical to GraphCTFMAPPOPolicy.
    GRU  : optional per-agent GRUCells (use_gru=True) — identical to GraphCTFMAPPOPolicy.

    MRO must place this mixin BEFORE the backbone class so its methods take priority.
    Concrete subclasses call self._mappo_independent_setup(D, use_gru) at the end
    of their __init__ (after super().__init__ so nn.Module is already initialized).
    """

    _GRU_H = 128

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _mappo_independent_setup(self, D: int, use_gru: bool) -> None:
        self.use_gru = use_gru
        self._h_buf  = None

        A = self._GRU_H if use_gru else D   # actor/critic head input dim

        if use_gru:
            self.gru_0 = nn.GRUCell(input_size=D, hidden_size=self._GRU_H)
            self.gru_1 = nn.GRUCell(input_size=D, hidden_size=self._GRU_H)

        # Independent per-agent neighbor scoring: h_nbr → scalar
        self.node_mlp_0 = nn.Sequential(
            nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 1)
        )
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 1)
        )
        self.extra_action_head_0 = nn.Linear(A, self.extra_action_dim)
        self.extra_action_head_1 = nn.Linear(A, self.extra_action_dim)

        # DeepSets centralized critic (same architecture as GraphCTFMAPPOPolicy)
        self.phi_net = nn.Sequential(
            nn.Linear(A + 2 * D, A), nn.ReLU(), nn.Linear(A, A)
        )
        self.rho_net = nn.Sequential(
            nn.Linear(A, A), nn.ReLU(), nn.Linear(A, 1)
        )

    # ------------------------------------------------------------------
    # GRU helpers  (identical to GraphCTFMAPPOPolicy)
    # ------------------------------------------------------------------

    def _apply_gru(self, h: torch.Tensor, agent_id: torch.Tensor) -> torch.Tensor:
        B, device = h.size(0), h.device
        if not self.training:
            if self._h_buf is None or self._h_buf.shape[0] != B or self._h_buf.device != device:
                self._h_buf = torch.zeros(B, self._GRU_H, device=device)
            h0 = self._h_buf.detach()
        else:
            h0 = h.new_zeros(B, self._GRU_H)
        h_gru_0 = self.gru_0(h, h0)
        h_gru_1 = self.gru_1(h, h0)
        sel     = (agent_id == 0).unsqueeze(-1)
        h_new   = torch.where(sel, h_gru_0, h_gru_1)
        if not self.training:
            self._h_buf = h_new.detach()
        return h_new

    def reset_hidden(self, indices=None) -> None:
        if self._h_buf is None:
            return
        if indices is None:
            self._h_buf.zero_()
        else:
            self._h_buf[indices] = 0.0

    def _apply_gru_stateless(self, h: torch.Tensor, agent_id: torch.Tensor) -> torch.Tensor:
        h0      = h.new_zeros(h.size(0), self._GRU_H)
        h_gru_0 = self.gru_0(h, h0)
        h_gru_1 = self.gru_1(h, h0)
        sel     = (agent_id == 0).unsqueeze(-1)
        return torch.where(sel, h_gru_0, h_gru_1)

    # ------------------------------------------------------------------
    # Actor: independent scoring (no h_actor in neighbor logits)
    # ------------------------------------------------------------------

    def _compute_actor_logits(self, h_actor, neighbor_embeddings, neighbor_mask, agent_id):
        sel   = (agent_id == 0).to(h_actor.device)          # [B] bool
        nbr_0 = self.node_mlp_0(neighbor_embeddings).squeeze(-1)   # [B, max_deg]
        nbr_1 = self.node_mlp_1(neighbor_embeddings).squeeze(-1)
        ext_0 = self.extra_action_head_0(h_actor)                   # [B, extra_dim]
        ext_1 = self.extra_action_head_1(h_actor)

        nbr_logits   = torch.where(sel.unsqueeze(-1), nbr_0, nbr_1)
        nbr_logits   = nbr_logits.masked_fill(neighbor_mask == 0, float('-inf'))
        extra_logits = torch.where(sel.unsqueeze(-1), ext_0, ext_1)
        return nbr_logits, extra_logits

    # ------------------------------------------------------------------
    # Critic: DeepSets  (identical to GraphCTFMAPPOPolicy)
    # ------------------------------------------------------------------

    def _get_teammate_embeddings(self, obs):
        tm_obs = {
            'x':                    obs['teammate_x'],
            'node_visibility_mask': obs['teammate_node_visibility_mask'],
            'edge_index':           obs['teammate_edge_index'],
            'edge_attr':            obs['teammate_edge_attr'],
            'edge_visibility_mask': obs['teammate_edge_visibility_mask'],
        }
        return self.features_extractor(tm_obs)

    def _compute_values(self, node_h, pyg_batch, h_actor, obs=None, agent_id=None):
        mean_s  = global_mean_pool(node_h, pyg_batch.batch)
        max_s   = global_max_pool(node_h, pyg_batch.batch)
        phi_ego = self.phi_net(torch.cat([h_actor, mean_s, max_s], dim=-1))

        if obs is not None and 'teammate_x' in obs:
            node_h_tm, pyg_batch_tm = self._get_teammate_embeddings(obs)
            tm_local = obs['teammate_agent_node_local_idx'].long()
            if tm_local.dim() > 1:
                tm_local = tm_local.squeeze(-1)
            h_tm    = node_h_tm[pyg_batch_tm.ptr[:-1] + tm_local]
            mean_tm = global_mean_pool(node_h_tm, pyg_batch_tm.batch)
            max_tm  = global_max_pool(node_h_tm, pyg_batch_tm.batch)
            if self.use_gru and agent_id is not None:
                h_tm = self._apply_gru_stateless(h_tm, 1 - agent_id)
            phi_tm = self.phi_net(torch.cat([h_tm, mean_tm, max_tm], dim=-1))
        else:
            phi_tm = torch.zeros_like(phi_ego)

        return self.rho_net(phi_ego + phi_tm).squeeze(-1)

    # ------------------------------------------------------------------
    # Mask helper (v2 and v4_hybrid inline this; expose as a method here)
    # ------------------------------------------------------------------

    def _apply_mask(self, full_logits, obs, B, device):
        if 'action_mask' not in obs or obs['action_mask'] is None:
            return full_logits
        mask = obs['action_mask'].to(torch.bool).to(device)
        A = full_logits.shape[1]
        if mask.shape[1] > A:
            mask = mask[:, :A]
        elif mask.shape[1] < A:
            pad = torch.zeros(B, A - mask.shape[1], dtype=torch.bool, device=device)
            mask = torch.cat([mask, pad], dim=-1)
        return full_logits.masked_fill(~mask, float('-inf'))

    # ------------------------------------------------------------------
    # ActorCriticPolicy interface
    # ------------------------------------------------------------------

    def _get_agent_id(self, obs, B, device):
        if 'agent_id' in obs:
            return obs['agent_id'].long().squeeze(-1)
        return torch.zeros(B, dtype=torch.long, device=device)

    def forward(self, obs, deterministic=False):
        node_h, pyg_batch, B, device, h_agent, neighbor_embeddings, neighbor_mask = \
            self._get_embeddings(obs)
        agent_id = self._get_agent_id(obs, B, device)
        h_actor  = self._apply_gru(h_agent, agent_id) if self.use_gru else h_agent
        nbr_logits, extra_logits = self._compute_actor_logits(
            h_actor, neighbor_embeddings, neighbor_mask, agent_id
        )
        full_logits = self._apply_mask(
            torch.cat([nbr_logits, extra_logits], dim=-1), obs, B, device
        )
        dist      = Categorical(logits=full_logits)
        actions   = dist.sample() if not deterministic else torch.argmax(full_logits, dim=-1)
        log_probs = dist.log_prob(actions)
        values    = self._compute_values(node_h, pyg_batch, h_actor, obs, agent_id)
        return actions, values, log_probs

    def predict_values(self, obs):
        node_h, pyg_batch, B, device, h_agent, _, _ = self._get_embeddings(obs)
        agent_id = self._get_agent_id(obs, B, device)
        h_actor  = self._apply_gru(h_agent, agent_id) if self.use_gru else h_agent
        return self._compute_values(node_h, pyg_batch, h_actor, obs, agent_id)

    def evaluate_actions(self, obs, actions):
        node_h, pyg_batch, B, device, h_agent, neighbor_embeddings, neighbor_mask = \
            self._get_embeddings(obs)
        agent_id = self._get_agent_id(obs, B, device)
        h_actor  = self._apply_gru(h_agent, agent_id) if self.use_gru else h_agent
        nbr_logits, extra_logits = self._compute_actor_logits(
            h_actor, neighbor_embeddings, neighbor_mask, agent_id
        )
        full_logits = self._apply_mask(
            torch.cat([nbr_logits, extra_logits], dim=-1), obs, B, device
        )
        dist     = Categorical(logits=full_logits)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        values   = self._compute_values(node_h, pyg_batch, h_actor, obs, agent_id)
        return values, log_prob, entropy


class GraphCTFMAPPOPolicy_v4_hybrid(
    _MAPPOIndependentActorMixin,
    GraphCTFNeighborBatchPolicy_SubgraphCompatible_v4_hybrid,
):
    """
    MAPPO on the v4_hybrid backbone — independent actor + DeepSets centralized critic.

    Actor  : node_mlp_i(h_nbr)  — per-agent, independent (no h_actor dependency).
    Critic : V = ρ(φ_ego + φ_tm)  — DeepSets, permutation invariant.
    GRU    : optional per-agent GRUCells (use_gru=True).

    Warm-starting from a v4_hybrid IPPO checkpoint: the shared
    GraphCTFMPNNExtractor_SubgraphCompatible backbone weights transfer directly.
    All single-agent heads (node_mlp, extra_action_head, value_head) are removed
    and replaced with per-agent MAPPO counterparts.
    """

    def __init__(self, *args, use_attention: bool = False, use_gru: bool = False, **kwargs):
        super().__init__(*args, use_attention=use_attention, **kwargs)
        D = self.hidden_dim  # 64
        # Remove parent's single-agent heads
        del self.node_mlp
        del self.extra_action_head
        del self.value_head
        self._mappo_independent_setup(D, use_gru)


class GraphCTFMAPPOPolicy_v2(
    _MAPPOIndependentActorMixin,
    GraphCTFNeighborBatchPolicy_SubgraphCompatible_v2,
):
    """
    MAPPO on the v2 backbone — independent actor + DeepSets centralized critic.

    Actor  : node_mlp_i(h_nbr)  — per-agent, independent (no h_actor dependency).
    Critic : V = ρ(φ_ego + φ_tm)  — DeepSets, permutation invariant.
    GRU    : optional per-agent GRUCells (use_gru=True).

    Warm-starting from a v2 IPPO checkpoint: the shared
    GraphCTFMPNNExtractor_SubgraphCompatible backbone weights transfer directly.
    All single-agent heads (node_mlp, extra_action_head, value_head) are removed
    and replaced with per-agent MAPPO counterparts.

    Note: v2 does not accept use_attention; this class accepts use_gru only.
    """

    def __init__(self, *args, use_gru: bool = False, **kwargs):
        # v2 __init__ does not accept use_attention — do not forward it
        super().__init__(*args, **kwargs)
        D = 64  # v2 hardcodes hidden_dim locally (no self.hidden_dim attribute)
        # Remove parent's single-agent heads
        del self.node_mlp
        del self.extra_action_head
        del self.value_head
        self._mappo_independent_setup(D, use_gru)

    def _get_embeddings(self, obs):
        """
        v2 inlines this logic; expose it as a method so the mixin's
        forward/predict_values/evaluate_actions can call self._get_embeddings.
        """
        device = obs['x'].device
        node_h, pyg_batch = self.features_extractor(obs)
        ptr = pyg_batch.ptr
        B   = ptr.size(0) - 1

        agent_local_idx = obs['agent_node_local_idx'].long().to(device)
        if agent_local_idx.dim() > 1:
            agent_local_idx = agent_local_idx.squeeze(-1)
        agent_global_idx    = ptr[:-1] + agent_local_idx
        h_agent             = node_h[agent_global_idx]

        neighbor_local_idx  = obs['neighbor_local_idx'].long().to(device)
        neighbor_mask       = obs['neighbor_mask'].to(device)
        neighbor_global_idx = ptr[:-1].unsqueeze(1) + neighbor_local_idx
        neighbor_global_idx = neighbor_global_idx.clamp(0, node_h.size(0) - 1)
        neighbor_embeddings = node_h[neighbor_global_idx]

        return node_h, pyg_batch, B, device, h_agent, neighbor_embeddings, neighbor_mask


MAPPO_POLICY_REGISTRY = {
    "v2":        GraphCTFMAPPOPolicy_v2,
    "v4":        GraphCTFMAPPOPolicy,
    "v4_hybrid": GraphCTFMAPPOPolicy_v4_hybrid,
}

if __name__ == "__main__":
    from graphpolicy import GraphPolicy
    env = GraphCTF(ctf_player_config="2v2")
    parallel_api_test(env, num_cycles=100)

    if_block = True
    train = True
    load = False
    if if_block:
        num_envs = 4
        parallel_env_fns = [
            make_env(GraphCTF, seed=100, rank=i, env_kwargs={"ctf_player_config": "2v2"})
            for i in range(num_envs)
        ]
        graph_ctf_env = parallel_env_fns[0]()
        RedGraphPolicy = GraphPolicy(team='Red', graph=graph_ctf_env.graph, num_actions=graph_ctf_env.num_actions, node_to_idx=graph_ctf_env.node_to_idx, neighbors=graph_ctf_env.neighbours, opp_flag_node=graph_ctf_env.blue_flag_node, seed=0, mode='crude_attack')

        parallel_coop_env_fns = [
            make_env(GraphCoopEnv, seed=100, rank=i, env_args=(graph_ctf_env, RedGraphPolicy), env_kwargs={})
            for i in range(num_envs)
        ]
        coop_vec_env = GraphParallelEnvToSB3VecEnv_v1(GraphCoopEnv, num_envs=num_envs, env_args=(graph_ctf_env, RedGraphPolicy), env_kwargs={})

        policy_kwargs = dict(
            features_extractor_class=GraphCTFMPNNExtractor,
            features_extractor_kwargs=dict(embedding_dim=64)
        )

        if train:
            model = PPO(
                policy=GraphCTFNeighborBatchPolicy_SubgraphCompatible_v2,  # subgraph version with neighbor logits
                env=coop_vec_env,
                batch_size=32,
                n_steps=128,
                learning_rate=3e-4,
                policy_kwargs={},
                verbose=1
            )
            model.learn(total_timesteps=100_000, progress_bar=True)
            model.save("GNN_PPO_BlueGraphPolicy_subgraph_v2")
