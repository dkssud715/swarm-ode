import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv

import gymnasium as gym
import tarware
import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

from torchdiffeq import odeint

import numpy as np
from collections import deque
import random
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
import rware
from gymnasium.envs.registration import register

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from tarware.utils.utils import flatten_list, split_list
from tarware.warehouse import Agent, AgentType, Warehouse
from tarware.definitions import (Action, AgentType, Direction,
                                 RewardType, CollisionLayers)
from tarware.spaces.MultiAgentBaseObservationSpace import MultiAgentBaseObservationSpace
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = ArgumentParser(description="Run tests with vector environments on WarehouseEnv", formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument(
        "--num_episodes",
        default=1000,
        type=int,
        help="The seed to run with"
    )
parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="The seed to run with"
    )

parser.add_argument(
        "--render",
        default=False,
        action='store_true',
    )

args = parser.parse_args()

def info_statistics(infos, global_episode_return, episode_returns):
    _total_deliveries = 0
    _total_clashes = 0
    _total_stuck = 0
    for info in infos:
        _total_deliveries += info["shelf_deliveries"]
        _total_clashes += info["clashes"]
        _total_stuck += info["stucks"]
        info["total_deliveries"] = _total_deliveries
        info["total_clashes"] = _total_clashes
        info["total_stuck"] = _total_stuck
    last_info = infos[-1]
    last_info["episode_length"] = len(infos)
    last_info["global_episode_return"] = global_episode_return
    last_info["episode_returns"] = episode_returns
    return last_info

class HeteroGNNNetwork(nn.Module):
    """
    Heterogeneous Graph Neural Network for MARL (ODE Removed for Efficiency)
    GNN을 통해 에이전트 간의 관계와 상태를 인코딩하는 역할.
    """
    def __init__(self, node_dims, action_size, hidden_dim=64, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Node type dimensions
        self.agv_dim = node_dims['agv']
        self.picker_dim = node_dims['picker'] 
        self.location_dim = node_dims['location']
        
        # Initial node embeddings
        self.agv_embedding = nn.Linear(self.agv_dim, hidden_dim)
        self.picker_embedding = nn.Linear(self.picker_dim, hidden_dim)
        self.location_embedding = nn.Linear(self.location_dim, hidden_dim)
        
        # Heterogeneous GNN layers for spatial context
        self.hetero_convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                ('agv', 'targets', 'location'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ('location', 'is targeted by', 'agv'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ('agv', 'communicates', 'agv'): SAGEConv(hidden_dim, hidden_dim),
                ('picker', 'manages', 'location'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ('agv', 'cooperates with', 'picker'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ('picker', 'helps', 'agv'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            }
            self.hetero_convs.append(HeteroConv(conv_dict, aggr='mean'))

        # Action heads for each agent type
        self.agv_action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_size)
        )
        
        self.picker_action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_size)
        )
        
    def forward(self, hetero_data):
        # 1. Initial Embeddings
        x_dict = {
            'agv': self.agv_embedding(hetero_data['agv'].x),
            'picker': self.picker_embedding(hetero_data['picker'].x),
            'location': self.location_embedding(hetero_data['location'].x)
        }
        
        # 2. GNN Processing for Collaboration Context
        for conv in self.hetero_convs:
            x_dict = conv(x_dict, hetero_data.edge_index_dict)
            x_dict = {key: torch.relu(x) for key, x in x_dict.items()}
        
        # 3. Generate Action Values directly from GNN outputs
        agv_q_values = self.agv_action_head(x_dict['agv'])
        picker_q_values = self.picker_action_head(x_dict['picker'])
        
        # Return embeddings as well, they are needed for the global state
        return {
            'agv_q_values': agv_q_values,
            'picker_q_values': picker_q_values,
            'agv_embeddings': x_dict['agv'],
            'picker_embeddings': x_dict['picker'],
            'location_embeddings': x_dict['location']
        }
import torch.nn.functional as F

class QMixer(nn.Module):
    """
    QMixer Network
    개별 에이전트의 Q-value를 조합하여 팀 전체의 Q-value를 생성.
    전역 상태(Global State)를 이용해 가중치를 동적으로 생성하는 하이퍼네트워크 구조를 가짐.
    """
    def __init__(self, num_agents, state_dim, mixing_embed_dim=32):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.mixing_embed_dim = mixing_embed_dim

        # Hypernetwork to generate weights for the mixing layers
        self.hyper_w1 = nn.Linear(self.state_dim, self.mixing_embed_dim * self.num_agents)
        self.hyper_b1 = nn.Linear(self.state_dim, self.mixing_embed_dim)
        
        self.hyper_w2 = nn.Linear(self.state_dim, self.mixing_embed_dim)
        self.hyper_b2 = nn.Linear(self.state_dim, 1)

    def forward(self, agent_q_values, states):
        # agent_q_values shape: (batch_size, num_agents)
        # states shape: (batch_size, state_dim)
        
        batch_size = agent_q_values.size(0)
        agent_q_values = agent_q_values.view(batch_size, 1, self.num_agents)

        # First mixing layer
        w1 = torch.abs(self.hyper_w1(states)).view(batch_size, self.num_agents, self.mixing_embed_dim)
        b1 = self.hyper_b1(states).view(batch_size, 1, self.mixing_embed_dim)
        hidden = F.elu(torch.bmm(agent_q_values, w1) + b1)
        
        # Second mixing layer
        w2 = torch.abs(self.hyper_w2(states)).view(batch_size, self.mixing_embed_dim, 1)
        b2 = self.hyper_b2(states).view(batch_size, 1, 1)
        
        # Output team Q-value
        q_total = torch.bmm(hidden, w2) + b2
        return q_total.view(batch_size, -1)
    
import random
from collections import deque
import torch.optim as optim
import numpy as np

class GraphMARL_QMIX:
    """
    Graph-based Multi-Agent QMIX
    IQL 대신 QMIX를 사용하여 에이전트 간의 협력을 학습.
    """
    def __init__(self, node_dims, num_agv, num_picker, action_size, 
                 lr=1e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.999, 
                 epsilon_min=0.05, memory_size=10000, hidden_dim=64):
        
        self.num_agv = num_agv
        self.num_picker = num_picker
        self.num_agents = num_agv + num_picker
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-Network and Mixer
        self.q_network = HeteroGNNNetwork(node_dims, action_size, hidden_dim)
        self.target_q_network = HeteroGNNNetwork(node_dims, action_size, hidden_dim)
        
        # The global state is the mean of all node embeddings
        state_dim = hidden_dim 
        self.mixer = QMixer(self.num_agents, state_dim)
        self.target_mixer = QMixer(self.num_agents, state_dim)
        
        # Optimizer includes parameters from both Q-network and Mixer
        self.params = list(self.q_network.parameters()) + list(self.mixer.parameters())
        self.optimizer = optim.Adam(self.params, lr=lr)
        
        self.memory = deque(maxlen=memory_size)
        self.update_target_networks()
        
    def update_target_networks(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        
    def remember(self, state, actions, rewards, next_state, done):
        # Store experience. Note: rewards and done are for the team.
        self.memory.append((state, actions, rewards, next_state, done))
        
    def _get_global_state(self, output_dict):
        """Derive global state from node embeddings."""
        all_embeddings = torch.cat([
            output_dict['agv_embeddings'],
            output_dict['picker_embeddings'],
            output_dict['location_embeddings']
        ], dim=0)
        return torch.mean(all_embeddings, dim=0, keepdim=True) # Shape: (1, hidden_dim)

    def act(self, hetero_data, valid_action_masks):
        # Action selection is the same as before (epsilon-greedy on individual Q-values)
        if np.random.rand() <= self.epsilon:
            actions = {}
            valid_agv_actions = np.where(valid_action_masks['agv'] == 1)[1]
            valid_picker_actions = np.where(valid_action_masks['picker'] == 1)[1]
            actions['agv'] = np.random.choice(valid_agv_actions, self.num_agv)
            actions['picker'] = np.random.choice(valid_picker_actions, self.num_picker)
            return actions

        with torch.no_grad():
            output = self.q_network(hetero_data)
            agv_q = output['agv_q_values']
            picker_q = output['picker_q_values']
            
            # Apply masks
            agv_q[valid_action_masks['agv'] == 0] = -float('inf')
            picker_q[valid_action_masks['picker'] == 0] = -float('inf')
            
            actions = {
                'agv': agv_q.argmax(dim=1).cpu().numpy(),
                'picker': picker_q.argmax(dim=1).cpu().numpy()
            }
        return actions
        
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        
        # Prepare tensors for batch processing
        states, actions, rewards, next_states, dones = zip(*minibatch)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        chosen_action_qvals = []
        next_max_qvals = []
        global_states = []
        next_global_states = []

        for i in range(batch_size):
            state = states[i]
            action = actions[i]
            next_state = next_states[i]
            
            output = self.q_network(state)
            target_output = self.target_q_network(next_state)

            # Get Q-values for the actions that were actually taken
            agv_actions_tensor = torch.tensor(action['agv']).long().unsqueeze(1)
            picker_actions_tensor = torch.tensor(action['picker']).long().unsqueeze(1)
            
            agv_q = output['agv_q_values'].gather(1, agv_actions_tensor)
            picker_q = output['picker_q_values'].gather(1, picker_actions_tensor)
            chosen_action_qvals.append(torch.cat((agv_q, picker_q)).squeeze())

            # Get max Q-values for the next state from the target network
            next_agv_q = target_output['agv_q_values'].max(dim=1)[0]
            next_picker_q = target_output['picker_q_values'].max(dim=1)[0]
            next_max_qvals.append(torch.cat((next_agv_q, next_picker_q)))
            
            # Get global states
            global_states.append(self._get_global_state(output))
            next_global_states.append(self._get_global_state(target_output))

        # Combine into batches
        chosen_action_qvals = torch.stack(chosen_action_qvals)
        next_max_qvals = torch.stack(next_max_qvals)
        global_states = torch.cat(global_states)
        next_global_states = torch.cat(next_global_states)

        # --- QMIX Core Logic ---
        # 1. Calculate Q_tot for the current state-action
        q_total = self.mixer(chosen_action_qvals, global_states)

        # 2. Calculate target Q_tot for the next state
        with torch.no_grad():
            target_q_total = self.target_mixer(next_max_qvals, next_global_states)

        # 3. Calculate TD Target
        targets = rewards.unsqueeze(1) + self.gamma * target_q_total * (1 - dones.unsqueeze(1))

        # 4. Calculate Loss
        loss = F.mse_loss(q_total, targets.detach())

        # 5. Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 1.0) # Gradient clipping
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class MultiAgentGraphObservationSpace(MultiAgentBaseObservationSpace):
    def __init__(self, num_agvs, num_pickers, grid_size, shelf_locations, normalised_coordinates=False, topk_tasks=5, max_comm_distance=5.0, max_task_distance=10.0):
        super(MultiAgentGraphObservationSpace, self).__init__(num_agvs, num_pickers, grid_size, shelf_locations, normalised_coordinates)

        # Parameters for the graph observation space
        self.topk_tasks = topk_tasks
        self.max_comm_distance = max_comm_distance
        self.max_task_distance = max_task_distance

        # Node counts
        self.num_agv_nodes = num_agvs
        self.num_picker_nodes = num_pickers
        self.num_location_nodes = None

        # Node feature dimensions
        self.agv_feature_dim = 7 # [carrying_shelf, carrying_requested, toggle_loading, pos_y, pos_x, target_y, target_x]
        self.picker_feature_dim = 4 # [pos_y, pos_x, target_y, target_x]
        self.location_feature_dim = 2 # [has_shelf, is_requested]

        # Store extracted info from environment
        self._current_agents_info = []
        self._current_shelves_info = []
        self._rack_locations = []

        # Graph components
        self.node_features = None
        self.edge_index = None
        self.edge_features = None
        self.node_types = None

        self.obs_lengths = [self.obs_length for _ in range(self.num_agents)]
        self._current_agents_info = []
        self._current_shelves_info = []

    def extract_environment_info(self, environment):
        self._current_agents_info = []
        self._current_shelves_info = []
        self._rack_locations = [] #(x, y)

        # Extract agents info
        for agent in environment.agents:
            agent_info = []
            if agent.type == AgentType.AGV:
                if agent.carrying_shelf:
                    agent_info.extend([1, int(agent.carrying_shelf in environment.request_queue)])
                else:
                    agent_info.extend([0, 0])
                agent_info.extend([agent.req_action == Action.TOGGLE_LOAD])
            agent_info.extend(self.process_coordinates((agent.y, agent.x), environment))
            if agent.target:
                agent_info.extend(self.process_coordinates(environment.action_id_to_coords_map[agent.target], environment))
            else:
                agent_info.extend([0, 0])
            self._current_agents_info.append(agent_info)
    
        # Extract shelves info
        for group_idx, group in enumerate(environment.rack_groups):
            for (y, x) in group:
                self._rack_locations.append((x, y))
                id_shelf = environment.grid[CollisionLayers.SHELVES, y, x]
                if id_shelf != 0:
                    self._current_shelves_info.extend([1.0, int(environment.shelfs[id_shelf - 1] in environment.request_queue), group_idx])
                else:
                    self._current_shelves_info.extend([0, 0, 0])

    
    def _build_graph_from_extracted_info(self, env):
        self.num_location_nodes = len(self._rack_locations)

        agv_features = []
        picker_features = []
        location_features = []
        self.position_to_sections = {}
        # Extract node features from current_agents_info
        agv_count= 0
        picker_count = 0

        for i, agent in enumerate(env.agents):
            agent_info = self._current_agents_info[i]

            if agent.type == AgentType.AGV:
                agv_features.append(agent_info)
                agv_count += 1

            elif agent.type == AgentType.PICKER:
                picker_features.append(agent_info)
                picker_count += 1
        
        # Extract task features from current_shelves_info
        shelf_info_idx = 0
        for i, rack_pos in enumerate(self._rack_locations): # rack_pos is (x, y) format
            has_shelf = self._current_shelves_info[shelf_info_idx]
            is_requested = self._current_shelves_info[shelf_info_idx + 1]
            group_id = self._current_shelves_info[shelf_info_idx + 2]
            features = [has_shelf, is_requested] # , group_id]
            self.position_to_sections[rack_pos] = group_id  # x, y
            location_features.append(features)
            shelf_info_idx += 3
        
        self.node_types = self._create_node_types()
        
        # build edges dynamically based on the environment
        self.edge_list = self._build_edges(env) #  edge_list = [agv2location_edges, location2agv_edges, agv2agv_edges, picker2location_edges, agv2picker_edges]


        data = HeteroData()
        if agv_features:
            data['agv'].num_nodes = self.num_agv_nodes
            data['agv'].x = torch.tensor(agv_features, dtype=torch.float32)
        else:
            data['agv'].num_nodes = 0
            data['agv'].x = torch.empty((0, self.agv_feature_dim), dtype=torch.float32)
        
        if picker_features:
            data['picker'].num_nodes = self.num_picker_nodes
            data['picker'].x = torch.tensor(picker_features, dtype=torch.float32)
        else:
            data['picker'].num_nodes = 0
            data['picker'].x = torch.empty((0, self.picker_feature_dim), dtype=torch.float32)
        if location_features:
            data['location'].num_nodes = self.num_location_nodes
            data['location'].x = torch.tensor(location_features, dtype=torch.float32)
        else:
            data['location'].num_nodes = 0
            data['location'].x = torch.empty((0, self.location_feature_dim), dtype=torch.float32)
        
        if self.edge_list[0]:  # agv2location_edges
            data['agv', 'targets', 'location'].edge_index = torch.tensor(self.edge_list[0], dtype=torch.long).t().contiguous()
        else:
            data['agv', 'targets', 'location'].edge_index = torch.empty((2, 0), dtype=torch.long)
        if self.edge_list[1]:  # location2agv_edges
            data['location', 'is targeted by', 'agv'].edge_index = torch.tensor(self.edge_list[1], dtype=torch.long).t().contiguous()
        else:
            data['location', 'is targeted by', 'agv'].edge_index = torch.empty((2, 0), dtype=torch.long)
        if self.edge_list[2]:  # agv2agv_edges
            data['agv', 'communicates', 'agv'].edge_index = torch.tensor(self.edge_list[2], dtype=torch.long).t().contiguous()
        else:
            data['agv', 'communicates', 'agv'].edge_index = torch.empty((2, 0), dtype=torch.long)
        if self.edge_list[3]:  # picker2location_edges
            data['picker', 'manages', 'location'].edge_index = torch.tensor(self.edge_list[3], dtype=torch.long).t().contiguous()
        else:
            data['picker', 'manages', 'location'].edge_index = torch.empty((2, 0), dtype=torch.long)
        if self.edge_list[4]:  # agv2picker_edges
            data['agv', 'cooperates with', 'picker'].edge_index = torch.tensor(self.edge_list[4], dtype=torch.long).t().contiguous()
        else:
            data['agv', 'cooperates with', 'picker'].edge_index = torch.empty((2, 0), dtype=torch.long)
        if self.edge_list[5]:  # picker2agv_edges
            data['picker', 'helps', 'agv'].edge_index = torch.tensor(self.edge_list[5], dtype=torch.long).t().contiguous()
        else:
            data['picker', 'helps', 'agv'].edge_index = torch.empty((2, 0), dtype=torch.long)

        return data
    def _build_edges(self, env):
        edge_list = []

        # AGV-to-Task Edges
        agv2location_edges, location2agv_edges = self._build_agv_to_location_edges()
        # AGV-to-AGV Edges
        agv2agv_edges = self._build_agv_to_agv_edges()
        # Picker-to-Task Edges
        picker2location_edges = self._build_picker_to_location_edges()
        # AGV-to-Picker Edges
        agv2picker_edges, picker2agv_edges = self._build_agv_to_picker_edges()

        # Convert to arrays
        edge_list = [agv2location_edges, location2agv_edges, agv2agv_edges, picker2location_edges, agv2picker_edges, picker2agv_edges]
        return edge_list

    def _build_agv_to_location_edges(self):
        agv2location_edges = []
        location2agv_edges = []
        for agv_idx in range(self.num_agv_nodes):
            agent_info = self._current_agents_info[agv_idx]

            agv_target = np.array([agent_info[5], agent_info[6]]) # target_y, target_x
            has_target = not (agv_target[0] ==0 and agv_target[1] == 0)

            if has_target:
                for loc_idx, rack_pos in enumerate(self._rack_locations):
                    if rack_pos[0]  == agv_target[1] and rack_pos[1] == agv_target[0]:
                        agv2location_edges.extend([(agv_idx, loc_idx)])
                        location2agv_edges.extend([(loc_idx, agv_idx)])
                        break
            else:
                for location_idx, _ in enumerate(self._rack_locations):
                    location_info = self._current_shelves_info[location_idx * 3: (location_idx + 1) * 3]
                    has_shelf = location_info[0]
                    is_requested = location_info[1]
                    if has_shelf and is_requested:
                        agv2location_edges.extend([(agv_idx, loc_idx)])
                        location2agv_edges.extend([(loc_idx, agv_idx)])

        return agv2location_edges, location2agv_edges
    
    def _build_agv_to_agv_edges(self):
        """ Build AGV-to-AGV edges : distance  """
        agv2agv_edges = []
        for i in range(self.num_agv_nodes):
            for j in range(i+1, self.num_agv_nodes):
                agent_i_info = self._current_agents_info[i]
                agent_j_info = self._current_agents_info[j]

                pos_i = np.array([agent_i_info[4], agent_i_info[3]]) # pos_x, pos_y
                pos_j = np.array([agent_j_info[4], agent_j_info[3]]) # pos_x, pos_y

                target_i = np.array([agent_i_info[6], agent_i_info[5]]) # target_x, target_y
                target_j = np.array([agent_j_info[6], agent_j_info[5]]) # target_x, target_y

                dist = np.linalg.norm(pos_i - pos_j, ord=1)

                has_target_i = not (target_i[0] == 0 and target_i[1] == 0)
                has_target_j = not (target_j[0] == 0 and target_j[1] == 0)
                if has_target_i and has_target_j:
                    same_section = self._check_same_rack_group(target_i, target_j)

                if i != j and (dist <= self.max_comm_distance or same_section):
                    agv2agv_edges.extend([(i, j), (j, i)])

        return agv2agv_edges

    def _build_picker_to_location_edges(self):
        """Build Picker-Task edges: zone-based + assigned tasks"""
        picker2location_edges = []
        for picker_idx in range(self.num_picker_nodes):
            agent_info = self._current_agents_info[self.num_agv_nodes + picker_idx]
            
            # Get picker position and target
            picker_pos = np.array([agent_info[1], agent_info[0]])  # pos_x, pos_y
            picker_target = np.array([agent_info[3], agent_info[2]])  # target_x, target_y
            has_target = not (picker_target[0] == 0 and picker_target[1] == 0)
            
            picker_section = self.position_to_sections.get(picker_pos, None)

            for loc_idx, rack_pos in enumerate(self._rack_locations): # rack_pos is (x, y) format
                shelf_info = self._current_shelves_info[loc_idx * 3: (loc_idx + 1) * 3]
                loc_section = self.position_to_sections.get(rack_pos, None)
                has_shelf = shelf_info[0]
                is_requested = shelf_info[1]

                if has_target and (rack_pos == picker_target):
                    picker2location_edges.append((picker_idx, loc_idx))
                elif not has_target and picker_section == loc_section and (has_shelf and is_requested):
                    picker2location_edges.append((picker_idx, loc_idx))
        
        return picker2location_edges
    
    def _build_agv_to_picker_edges(self):
        """Build AGV-Picker edges: cooperation based"""
        agv2picker_edges = []
        picker2agv_edges = []

        for agv_idx in range(self.num_agv_nodes):
            for picker_idx in range(self.num_picker_nodes):
                agv_info = self._current_agents_info[agv_idx]
                picker_info = self._current_agents_info[self.num_agv_nodes + picker_idx]
                
                # Get positions and targets
                agv_pos = np.array([agv_info[4], agv_info[3]]) # pos_x, pos_y
                picker_pos = np.array([picker_info[1], picker_info[0]]) # pos_x, pos_y
                agv_target = np.array([agv_info[6], agv_info[5]]) # target_x, target_y
                picker_target = np.array([picker_info[3], picker_info[2]]) # target_x, target_y
                
                # Check cooperation conditions
                
                # Condition 1: Spatial proximity
                dist = np.linalg.norm(agv_pos - picker_pos, ord=1)
                close_proximity = dist <= self.max_comm_distance

                # Condition 2: Same target section coordination
                has_picker_taget = not (picker_target[0] == 0 and picker_target[1] == 0)
                has_agv_target = not (agv_target[0] == 0 and agv_target[1] == 0)
                if has_agv_target and has_picker_taget:
                    same_target = picker_target[0] == agv_target[0] and picker_target[1] == agv_target[1]
                    if not same_target:
                        agv_target_section = self.position_to_sections.get(agv_target, None)
                        picker_target_section = self.position_to_sections.get(picker_target, None)
                        if agv_target_section is not None and picker_target_section is not None:
                            same_target_section = agv_target_section == picker_target_section

                else:
                    picker_current_section = self.position_to_sections.get(picker_pos, None)
                    if has_agv_target:
                        agv_target_section = self.position_to_sections.get(agv_target, None)
                        agv_target_in_picker_section = picker_current_section == agv_target_section
                
                if close_proximity or same_target or same_target_section or agv_target_in_picker_section:
                    agv2picker_edges.append((agv_idx, self.num_agv_nodes + picker_idx))
                    picker2agv_edges.append((self.num_agv_nodes + picker_idx, agv_idx))
        
        return agv2picker_edges, picker2agv_edges
    
    def _check_same_rack_group(self, pos_i, pos_j):
        group_i = self.position_to_sections[pos_i]
        group_j = self.position_to_sections[pos_j]
        return (group_i is not None and group_j is not None and group_i == group_j)

# Initialize model
node_dims = {
    'agv': 7,  # AGV feature dimension
    'picker': 4,  # Picker feature dimension
    'location': 2  # Location feature dimension
}

# env = gym.make("tarware-medium-40agvs-10pickers-partialobs-v1")
env = gym.make("tarware-large-19agvs-9pickers-partialobs-v1")
agent = GraphMARL_DQN(
    node_dims=node_dims,
    action_size=env.action_size,
    lr=1e-3,
    gamma=0.99
)
batch_size = 64
seed = args.seed
completed_episodes = 0
scores = deque(maxlen=100)
for episode in range(args.num_episodes):
    state = env.reset()
    hetero_data = env.observation_space_mapper._build_graph_from_extracted_info(env)
    
    total_reward = 0
    episode_returns = np.zeros(env.num_agents, dtype=np.float64)
    done = False
    step = 0
    all_infos = []
    while not done and step < 1000:  # Max steps per episode
        # Get valid action masks
        valid_action_masks = env.compute_valid_action_masks()
        
        # Choose actions
        actions = agent.act(hetero_data, valid_action_masks)
        
        # Take environment step
        next_state, rewards, dones, truncated, info = env.step(actions)
        next_hetero_data = env.observation_space_mapper._build_graph_from_extracted_info(env)
        
        # Store experience
        agent.remember(hetero_data, actions, rewards, next_hetero_data, dones)
        
        # Update state
        hetero_data = next_hetero_data
        episode_returns += np.array(rewards, dtype=np.float64)
        total_reward += sum(rewards)
        done = all(dones)
        step += 1
        all_infos.append(info)
        # Train the model
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            
    # Update target network periodically
    if episode % 100 == 0:
        agent.update_target_network()
        
    scores.append(total_reward)
    
    # Logging
    if episode % 100 == 0:
        avg_score = np.mean(scores)
        print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
    last_info = info_statistics(info, total_reward, episode_returns)
    last_info["overall_pick_rate"] = last_info.get("total_deliveries") * 3600 / (5 * last_info['episode_length'])
    episode_length = len(info)
    print(f"Completed Episode {completed_episodes}: | [Overall Pick Rate={last_info.get('overall_pick_rate'):.2f}]| [Global return={last_info.get('global_episode_return'):.2f}]| [Total shelf deliveries={last_info.get('total_deliveries'):.2f}]| [Total clashes={last_info.get('total_clashes'):.2f}]| [Total stuck={last_info.get('total_stuck'):.2f}] | [FPS = {episode_length/(end-start):.2f}]")
    completed_episodes += 1

env.close()