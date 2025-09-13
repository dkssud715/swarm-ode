from datetime import datetime

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, Linear
from torch_geometric.data import HeteroData, Batch, Data
# Import the converter from your code
from torchdiffeq import odeint
import wandb
import h5py
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
from collections import defaultdict, deque

class GraphODEFunc(nn.Module):    
    def __init__(self, node_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        
        # GraphSAGE layers
        self.conv1 = SAGEConv(node_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, node_dim)
        
        self.activation = nn.ReLU()
        
    def forward(self, t: torch.Tensor, x: torch.Tensor, 
                edge_index: torch.Tensor) -> torch.Tensor:
        
        h = self.conv1(x, edge_index)
        h = self.activation(h)
        
        h = self.conv2(h, edge_index)
        h = self.activation(h)
        
        # Final layer (no activation for derivative)
        dx_dt = self.conv3(h, edge_index)
        
        return dx_dt

class GraphODE(nn.Module):
    """Simple Graph Neural ODE for trajectory prediction"""
    
    def __init__(self, node_dim: int, num_agvs: int, num_pickers: int, hidden_dim: int = 64, ode_solver: str = 'euler'):
        super().__init__()
        
        self.node_dim = node_dim
        self.num_agvs = num_agvs
        self.num_pickers = num_pickers
        self.ode_solver = ode_solver
        
        # ODE function
        self.ode_func = GraphODEFunc(
            node_dim=node_dim,
            hidden_dim=hidden_dim
        )
        
        # Simple position decoder
        self.position_decoder = nn.Linear(node_dim, 2)
        
    def forward(self, batch_data: Batch, time_span: torch.Tensor) -> Dict[str, torch.Tensor]:

        x0 = batch_data.x  # [total_nodes, node_dim]
        edge_index = batch_data.edge_index
        batch = batch_data.batch  # [total_nodes] batch indices

        # Create ODE function with fixed graph structure
        def ode_func_wrapper(t, x):
            return self.ode_func(t, x, edge_index)
        
        # Solve ODE
        solution = odeint(
            ode_func_wrapper,
            x0,
            time_span,
            method=self.ode_solver,
            rtol=1e-3,
            atol=1e-4
        )  # [num_time_points, total_nodes, node_dim]
        
        # Extract trajectories (positions) at all time points
        trajectories = []
        for t_idx in range(solution.size(0)):
            node_features = solution[t_idx]  # [total_nodes, node_dim]
            positions = self.position_decoder(node_features)  # [total_nodes, 2]
            trajectories.append(positions)

        trajectories = torch.stack(trajectories, dim=0)  # [num_time_points, total_nodes, 2]

        return {
            'trajectories': trajectories,
            'node_features': solution,
            'batch': batch
        }
    
    def predict_trajectory(self, batch_data: Batch, num_steps: int, dt: float = 0.1) -> torch.Tensor:

        time_span = torch.arange(0, num_steps + 1, dtype=torch.float32)
        result = self.forward(batch_data, time_span)
        return result['trajectories']
    
class GraphConverter:
    def __init__(self, num_agvs, num_pickers, distance_threshold: float = 3.0, temporal_window: int = 5):
        self.num_agvs = num_agvs
        self.num_pickers = num_pickers
        self.distance_threshold = distance_threshold
        self.temporal_window = temporal_window
        self.graph_history = deque(maxlen=temporal_window)
    
    def _build_graph_from_observation(self, observations: np.ndarray) -> Data:
        num_agents = len(observations)
        
        # 1. Standardize observations with zero-padding
        standardized_obs = self._standardize_observations(observations)
        
        # 2. Extract locations from observations based on agent types
        locations = self._extract_locations_by_agent_type(standardized_obs)
        
        # 3. Create node features
        node_features = torch.tensor(standardized_obs, dtype=torch.float32)
        
        # 4. Compute spatial edges (bidirectional) - local 인덱스
        spatial_edges = self._compute_spatial_edges(locations)
        
        # 5. Create and store current graph FIRST
        current_graph = Data(x=node_features, edge_index=spatial_edges)
        self.graph_history.append(current_graph)
        
        # 6. Now calculate global indexing
        current_position_in_window = len(self.graph_history) - 1  # 0, 1, 2, 3, 4, then 4, 4, 4...
        current_timestep_offset = current_position_in_window * num_agents
        
        # 7. Build global node features
        global_node_features = []
        for prev_graph in self.graph_history:
            global_node_features.append(prev_graph.x)
        global_node_features = torch.cat(global_node_features, dim=0)
        
        # 8. Build global edge index
        all_edges = []
        # 8a. Add all previous spatial edges (이미 저장된 것들은 global 인덱스로 변환)
        for i in range(len(self.graph_history) - 1):  # 현재 제외
            prev_graph = self.graph_history[i]
            prev_spatial_edges = prev_graph.edge_index.clone()
            prev_timestep_offset = i * num_agents
            prev_spatial_edges += prev_timestep_offset
            all_edges.append(prev_spatial_edges)
        
        # 8b. Add current spatial edges (global 인덱스로 변환)
        if spatial_edges.shape[1] > 0:
            current_spatial_edges = spatial_edges.clone()
            current_spatial_edges += current_timestep_offset
            all_edges.append(current_spatial_edges)
        
        # 8c. Add temporal edges (이미 global 인덱스)
        temporal_edges = self._compute_temporal_edges_with_window(num_agents)
        if temporal_edges.shape[1] > 0:
            all_edges.append(temporal_edges)
        
        # 9. Combine all edges
        if len(all_edges) > 0:
            global_edge_index = torch.cat(all_edges, dim=1)
        else:
            global_edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # 10. Create node-level mask for current agents
        total_nodes = global_node_features.size(0)
        is_current_agent = torch.zeros(total_nodes, dtype=torch.bool)
        
        # 현재 timestep의 agent nodes만 True
        current_start = current_timestep_offset
        current_end = current_timestep_offset + num_agents
        is_current_agent[current_start:current_end] = True

        # 11. Create global graph
        global_graph = Data(x=global_node_features, edge_index=global_edge_index, is_current_agent=is_current_agent)
        
        return global_graph
    
    def _extract_locations_by_agent_type(self, observations: np.ndarray) -> np.ndarray:
        """
        Extract (y, x) locations from observations based on agent types
        
        AGV observation: [picker_info(7)] + [other_agvs_info] + [other_pickers_info] + [shelf_info]
        - picker_info: [carrying_shelf, in_request_queue, req_action, y, x, target_y, target_x]
        
        Picker observation: [picker_info(4)] + [other_pickers_info] + [agvs_info]  
        - picker_info: [y, x, target_y, target_x]
        """
        locations = []
        
        for i, obs in enumerate(observations):
            if i < self.num_agvs:  # AGV agent
                # AGV의 경우: 자신의 picker 정보에서 위치 추출 (인덱스 3, 4)
                y, x = obs[3], obs[4]
            else:  # Picker agent
                # Picker의 경우: 관찰의 시작 부분에서 위치 추출 (인덱스 0, 1)
                y, x = obs[0], obs[1]
            
            locations.append([y, x])
        
        return np.array(locations)
    
    def _standardize_observations(self, observations: np.ndarray) -> np.ndarray:
        """Standardize observation sizes with zero-padding"""
        if isinstance(observations, np.ndarray) and observations.dtype == object:
            obs_list = observations.tolist()
        elif isinstance(observations, list):
            obs_list = observations
        else:
            return observations
        
        max_obs_length = max(len(obs) for obs in obs_list)
        standardized_obs = np.zeros((len(obs_list), max_obs_length), dtype=np.float32)
        
        for i, obs in enumerate(obs_list):
            obs_array = np.array(obs, dtype=np.float32)
            standardized_obs[i, :len(obs_array)] = obs_array
        
        return standardized_obs
    
    def _compute_spatial_edges(self, locations: np.ndarray) -> torch.Tensor:
        """Compute spatial edges based on distance threshold"""
        num_agents = locations.shape[0]
        edges = []
        
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                distance = np.sqrt(np.sum((locations[i] - locations[j]) ** 2))
                
                if distance < self.distance_threshold:
                    edges.append([i, j])
                    edges.append([j, i])
        
        if len(edges) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        
        return torch.tensor(edges, dtype=torch.long).t()
    
    def _compute_temporal_edges_with_window(self, num_agents: int) -> torch.Tensor:
        """Compute temporal edges within the temporal window"""
        edges = []
        
        # Window 내에서의 현재 위치 (append 후이므로 마지막 위치)
        current_window_position = len(self.graph_history) - 1
        
        if current_window_position > 0:
            # 직전 timestep과의 연결만 (chain 형태)
            prev_window_position = current_window_position - 1
            
            prev_timestep_offset = prev_window_position * num_agents
            current_timestep_offset = current_window_position * num_agents
            for agent_idx in range(num_agents):
                prev_node_idx = prev_timestep_offset + agent_idx
                current_node_idx = current_timestep_offset + agent_idx
                edges.append([prev_node_idx, current_node_idx])
        
        if len(edges) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        
        return torch.tensor(edges, dtype=torch.long).t()

    def reset_history(self):
        """Reset the graph history (useful for new episodes)"""
        self.graph_history.clear()

class TrajectoryBatch:
    def __init__(self, graphs: Batch, next_positions: torch.Tensor):
        self.graphs = graphs
        self.next_positions = next_positions

class WarehouseDataset(Dataset):
    def __init__(self, h5_file_path: str):
        self.h5_file_path = h5_file_path

        with h5py.File(h5_file_path, 'r') as f:
            episode_ids = [int(key.split('_')[1]) for key in f.keys() if key.startswith('episode_')]
        self.episode_ids = sorted(episode_ids)

        self.sequences = []
        self.node_dim = None
        self.num_agvs = None
        self.num_pickers = None

        self._load_all_sequences()

    def _load_all_sequences(self):
        with h5py.File(self.h5_file_path, 'r') as f:
            episode_ids = [int(key.split('_')[1]) for key in f.keys() if key.startswith('episode_')]
            self.episode_ids = sorted(episode_ids)
            
            for episode_id in self.episode_ids:
                episode_group = f[f'episode_{episode_id:06d}']
                steps_group = episode_group['steps']
                num_agvs = episode_group['metadata'].attrs['num_agvs']
                num_pickers = episode_group['metadata'].attrs['num_pickers']
                
                if self.num_agvs is None:
                    self.num_agvs = num_agvs
                    self.num_pickers = num_pickers
                    
                converter = GraphConverter(num_agvs, num_pickers, distance_threshold=5.0, temporal_window=5)
                step_data = []
                
                for step_name in sorted(steps_group.keys()):
                    step_group = steps_group[step_name]
                    observations = step_group['observations'][:]
                    graph_data = converter._build_graph_from_observation(observations)
                    
                    if self.node_dim is None:
                        self.node_dim = graph_data.x.size(1)
                        
                    positions = self._extract_positions_from_graph(graph_data, num_agvs, num_pickers)
                    step_data.append({
                        'graph': graph_data,
                        'positions': positions,
                    })
                
                # (현재 그래프, 다음 위치) 페어 생성
                for i in range(len(step_data) - 1):  # 마지막 스텝은 다음이 없으니 제외
                    current_graph = step_data[i]['graph']
                    next_positions = step_data[i + 1]['positions']
                    
                    self.sequences.append(TrajectoryBatch(current_graph, next_positions))
                    
        print(f"Loaded {len(self.sequences)} step pairs from {self.h5_file_path}")
        print(f"Node dimension: {self.node_dim}")
        print(f"Agents: {self.num_agvs} AGVs, {self.num_pickers} Pickers")
    
    def _extract_positions_from_graph(self, graph: Data, num_agvs: int, num_pickers: int) -> torch.Tensor:
        """Extract position information from graph"""
        positions = []
        
        # AGV positions (indices 3, 4 in standardized obs -> y, x -> x, y)
        if num_agvs > 0:
            agv_features = graph.x[:num_agvs]
            agv_pos = agv_features[:, [4, 3]]  # [x, y]
            positions.append(agv_pos)
        
        # Picker positions (indices 0, 1 in standardized obs -> y, x -> x, y)  
        if num_pickers > 0:
            picker_features = graph.x[num_agvs:num_agvs + num_pickers]
            picker_pos = picker_features[:, [1, 0]]  # [x, y]
            positions.append(picker_pos)
        
        # Concatenate all positions
        all_positions = torch.cat(positions, dim=0)  # [total_agents, 2]
        
        return all_positions
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]

def collate_trajectory_batches(batch_list: List[TrajectoryBatch]) -> TrajectoryBatch:
    """Collate function for DataLoader"""
    # Batch graphs
    graphs = [item.graphs for item in batch_list]
    batched_graph = Batch.from_data_list(graphs)
    
    # Stack next positions
    next_positions = torch.stack([item.next_positions for item in batch_list], dim=0)  # [B, N, 2]
    
    return TrajectoryBatch(
        graphs=batched_graph,
        next_positions=next_positions
    )

def check_h5_structure(file_path):
    with h5py.File(file_path, 'r') as f:
        print("File contents:")
        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group: {name}")
            else:
                print(f"Dataset: {name}, shape: {obj.shape}")
        
        # f.visititems(print_structure)
        
        # Episode 개수 확인
        episodes = [key for key in f.keys() if key.startswith('episode_')]
        print(f"Found episodes: {len(episodes)}")
        
        if episodes:
            ep = f[episodes[0 ]]
            if 'steps' in ep:
                steps = list(ep['steps'].keys())
                print(f"Steps in first episode: {len(steps)}")
                
                if steps and 'observations' in ep['steps'][steps[0]]:
                    obs_shape = ep['steps'][steps[0]]['observations'].shape
                    print(f"Observation shape: {obs_shape}")
                else:
                    print("No observations found in steps!")
            else:
                print("No steps found in episode!")

def debug_episode_loading(loader, episode_id=0):
    try:
        episode_data = loader.load_episode_data(episode_id)
        print(f"Loaded episode data: {len(episode_data)} steps")
        
        if episode_data:
            first_step = episode_data[0]
            print(f"First step keys: {first_step.keys()}")
            if 'graph' in first_step:
                graph = first_step['graph']
                print(f"Graph node types: {list(graph.node_types)}")
                for node_type in graph.node_types:
                    print(f"{node_type} nodes: {graph[node_type].num_nodes}")
        else:
            print("No episode data loaded!")
            
    except Exception as e:
        print(f"Error loading episode: {e}")
        import traceback
        traceback.print_exc()

# Usage Example
if __name__ == "__main__":
    # Initialize predictor
    # environments = [
    #     'tarware-tiny-3agvs-2pickers-partialobs-v1',
    #     'tarware-small-6agvs-3pickers-partialobs-v1', 
    #     'tarware-medium-10agvs-5pickers-partialobs-v1',
    #     'tarware-medium-19agvs-9pickers-partialobs-v1',
    #     'tarware-large-15agvs-8pickers-partialobs-v1'
    # ]
    parameters = {
        'num_epochs': 1000,
        'batch_size': 32,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        }
    seed = [0, 1000, 2000, 3000, 4000]
    env = 'tarware-small-6agvs-3pickers-partialobs-v1'#'tarware-tiny-3agvs-2pickers-partialobs-v1'
    file_path = [f'./warehouse_data_{env}_seed{s}.h5' for s in seed]
    # check_h5_structure(file_path[0])
    dataset = ConcatDataset([WarehouseDataset(fp) for fp in file_path])
    train_size = int(0.8 * len(dataset))
    val_size = int(0.2 * len(dataset))
    train_dataset, val_dataset= torch.utils.data.random_split(dataset, [train_size, val_size])
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=parameters['batch_size'], shuffle=True, collate_fn=collate_trajectory_batches)
    val_loader = DataLoader(val_dataset, batch_size=parameters['batch_size'], shuffle=False, collate_fn=collate_trajectory_batches)
    node_dim = train_dataset[0].graphs.x.size(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GraphODE(node_dim = node_dim, num_agvs=3, num_pickers=2, hidden_dim=64, ode_solver='euler')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['lr'], weight_decay=parameters['weight_decay'])
    save_model_path = f'./trained_models/{env}/{datetime.now().strftime("%Y%m%d_%H%M%S")}/'
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    wandb.init(
    project="graph-ode-warehouse", 
    config=parameters,
    name=f"experiment_{env}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
)
    for epoch in range(parameters['num_epochs']):
        # Training
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        best_val_loss = float('inf')
        for batch in train_loader:
            batch.graphs = batch.graphs.to(device)
            batch_next_positions = batch.next_positions.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            time_span = torch.tensor([0., 1.], device=device)  # 한 스텝 (t=0 -> t=1)
            result = model(batch.graphs, time_span)
            pred_trajectories = result['trajectories']  # [2, total_nodes, 2] (t=0, t=1)
            pred_next_positions = pred_trajectories[1]  # [total_nodes, 2] (t=1만 사용)
            mask = batch.graphs.is_current_agent
            # current_predictions = pred_next_positions[mask]
            # loss = F.mse_loss(current_predictions, batch.next_positions.view(-1, 2))
            # Compute loss: 예측 vs 실제 다음 위치
            loss = F.mse_loss(pred_next_positions[mask], batch.next_positions.view(-1, 2))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            num_train_batches += 1
        
        avg_train_loss = total_train_loss / num_train_batches
        
        # Validation
        val_loss = 0
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                total_val_loss = 0
                num_val_batches = 0
                
                for batch in val_loader:
                    batch.graphs = batch.graphs.to(device)
                    batch.next_positions = batch.next_positions.to(device)
                    
                    time_span = torch.tensor([0., 1.], device=device)
                    result = model(batch.graphs, time_span)
                    pred_next_positions = result['trajectories'][1]
                    mask = batch.graphs.is_current_agent
                    loss = F.mse_loss(pred_next_positions[mask], batch.next_positions.view(-1, 2))
                    total_val_loss += loss.item()
                    num_val_batches += 1
                
                val_loss = total_val_loss / num_val_batches
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_model_path + 'best_model.pth')
                print(f"Saved best model at epoch {epoch} with val loss {best_val_loss:.6f}")
            if epoch % 50 == 0:
                torch.save(model.state_dict(), save_model_path + f'checkpoint_epoch{epoch}.pth')
                print(f"Saved checkpoint at epoch {epoch}")
        print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': val_loss
        })