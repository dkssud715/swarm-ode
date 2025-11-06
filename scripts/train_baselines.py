from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import wandb
import h5py

class SequenceDataset(Dataset):
    """Dataset for sequential models (GRU/LSTM)"""
    def __init__(self, h5_file_path: str, seq_length: int = 5):
        self.h5_file_path = h5_file_path
        self.seq_length = seq_length
        self.sequences = []
        self.num_agvs = None
        self.num_pickers = None
        self.obs_dim = None
        
        self._load_all_sequences()
    
    def _load_all_sequences(self):
        with h5py.File(self.h5_file_path, 'r') as f:
            episode_ids = [int(key.split('_')[1]) for key in f.keys() if key.startswith('episode_')]
            episode_ids = sorted(episode_ids)
            
            for episode_id in episode_ids:
                episode_group = f[f'episode_{episode_id:06d}']
                steps_group = episode_group['steps']
                num_agvs = episode_group['metadata'].attrs['num_agvs']
                num_pickers = episode_group['metadata'].attrs['num_pickers']
                
                if self.num_agvs is None:
                    self.num_agvs = num_agvs
                    self.num_pickers = num_pickers
                
                # Collect all observations and positions in episode
                episode_data = []
                for step_name in sorted(steps_group.keys()):
                    step_group = steps_group[step_name]
                    observations = step_group['observations'][:]
                    
                    # Standardize observations
                    standardized_obs = self._standardize_observations(observations)
                    
                    if self.obs_dim is None:
                        self.obs_dim = standardized_obs.shape[1]
                    
                    # Extract positions
                    positions = self._extract_positions(standardized_obs, num_agvs, num_pickers)
                    
                    episode_data.append({
                        'observations': standardized_obs,
                        'positions': positions
                    })
                
                # Create sequences with sliding window
                for i in range(len(episode_data) - self.seq_length):
                    seq_obs = []
                    seq_pos = []
                    
                    for j in range(self.seq_length):
                        seq_obs.append(episode_data[i + j]['observations'])
                        seq_pos.append(episode_data[i + j]['positions'])
                    
                    # Target is the next position
                    target_pos = episode_data[i + self.seq_length]['positions']
                    
                    self.sequences.append({
                        'observations': np.stack(seq_obs, axis=0),  # [seq_len, num_agents, obs_dim]
                        'positions': np.stack(seq_pos, axis=0),      # [seq_len, num_agents, 2]
                        'target_positions': target_pos                # [num_agents, 2]
                    })
        
        print(f"Loaded {len(self.sequences)} sequences from {self.h5_file_path}")
        print(f"Observation dimension: {self.obs_dim}")
        print(f"Agents: {self.num_agvs} AGVs, {self.num_pickers} Pickers")
    
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
    
    def _extract_positions(self, observations: np.ndarray, num_agvs: int, num_pickers: int) -> np.ndarray:
        """Extract positions from observations"""
        positions = []
        
        # AGV positions (y, x at indices 3, 4 -> convert to x, y)
        for i in range(num_agvs):
            y, x = observations[i, 3], observations[i, 4]
            positions.append([x, y])
        
        # Picker positions (y, x at indices 0, 1 -> convert to x, y)
        for i in range(num_agvs, num_agvs + num_pickers):
            y, x = observations[i, 0], observations[i, 1]
            positions.append([x, y])
        
        return np.array(positions, dtype=np.float32)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        item = self.sequences[idx]
        return {
            'observations': torch.tensor(item['observations'], dtype=torch.float32),
            'positions': torch.tensor(item['positions'], dtype=torch.float32),
            'target_positions': torch.tensor(item['target_positions'], dtype=torch.float32)
        }


class GRUTrajectoryPredictor(nn.Module):
    """GRU-based trajectory prediction model"""
    def __init__(self, obs_dim: int, num_agents: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder: observation -> hidden
        self.encoder = nn.Linear(obs_dim, hidden_dim)
        
        # GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Decoder: hidden -> position
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: [batch_size, seq_len, num_agents, obs_dim]
        Returns:
            positions: [batch_size, num_agents, 2]
        """
        batch_size, seq_len, num_agents, obs_dim = observations.shape
        
        # Reshape for processing
        obs_flat = observations.view(batch_size * num_agents, seq_len, obs_dim)
        
        # Encode observations
        encoded = self.encoder(obs_flat)  # [B*N, seq_len, hidden_dim]
        
        # GRU
        gru_out, _ = self.gru(encoded)  # [B*N, seq_len, hidden_dim]
        
        # Use last hidden state
        last_hidden = gru_out[:, -1, :]  # [B*N, hidden_dim]
        
        # Decode to position
        positions = self.decoder(last_hidden)  # [B*N, 2]
        
        # Reshape back
        positions = positions.view(batch_size, num_agents, 2)
        
        return positions


class LSTMTrajectoryPredictor(nn.Module):
    """LSTM-based trajectory prediction model"""
    def __init__(self, obs_dim: int, num_agents: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder: observation -> hidden
        self.encoder = nn.Linear(obs_dim, hidden_dim)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Decoder: hidden -> position
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: [batch_size, seq_len, num_agents, obs_dim]
        Returns:
            positions: [batch_size, num_agents, 2]
        """
        batch_size, seq_len, num_agents, obs_dim = observations.shape
        
        # Reshape for processing
        obs_flat = observations.view(batch_size * num_agents, seq_len, obs_dim)
        
        # Encode observations
        encoded = self.encoder(obs_flat)  # [B*N, seq_len, hidden_dim]
        
        # LSTM
        lstm_out, _ = self.lstm(encoded)  # [B*N, seq_len, hidden_dim]
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # [B*N, hidden_dim]
        
        # Decode to position
        positions = self.decoder(last_hidden)  # [B*N, 2]
        
        # Reshape back
        positions = positions.view(batch_size, num_agents, 2)
        
        return positions


class PositionOnlyGRU(nn.Module):
    """GRU model using only position information"""
    def __init__(self, num_agents: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU for position sequences
        self.gru = nn.GRU(
            input_size=2,  # Only (x, y) positions
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: [batch_size, seq_len, num_agents, 2]
        Returns:
            next_positions: [batch_size, num_agents, 2]
        """
        batch_size, seq_len, num_agents, _ = positions.shape
        
        # Reshape
        pos_flat = positions.view(batch_size * num_agents, seq_len, 2)
        
        # GRU
        gru_out, _ = self.gru(pos_flat)
        last_hidden = gru_out[:, -1, :]
        
        # Decode
        next_positions = self.decoder(last_hidden)
        next_positions = next_positions.view(batch_size, num_agents, 2)
        
        return next_positions


class PositionOnlyLSTM(nn.Module):
    """LSTM model using only position information"""
    def __init__(self, num_agents: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM for position sequences
        self.lstm = nn.LSTM(
            input_size=2,  # Only (x, y) positions
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: [batch_size, seq_len, num_agents, 2]
        Returns:
            next_positions: [batch_size, num_agents, 2]
        """
        batch_size, seq_len, num_agents, _ = positions.shape
        
        # Reshape
        pos_flat = positions.view(batch_size * num_agents, seq_len, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(pos_flat)
        last_hidden = lstm_out[:, -1, :]
        
        # Decode
        next_positions = self.decoder(last_hidden)
        next_positions = next_positions.view(batch_size, num_agents, 2)
        
        return next_positions


def train_baseline_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    parameters: dict,
    save_path: str,
    device: str = 'cuda'
):
    """Train baseline model (GRU or LSTM)"""
    
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=parameters['lr'], 
        weight_decay=parameters['weight_decay']
    )
    
    wandb.init(
        project="graph-ode-warehouse",
        config=parameters,
        name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(parameters['num_epochs']):
        # Training
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        
        for batch in train_loader:
            observations = batch['observations'].to(device)
            positions = batch['positions'].to(device)
            target_positions = batch['target_positions'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if 'PositionOnly' in model.__class__.__name__:
                pred_positions = model(positions)
            else:
                pred_positions = model(observations)
            
            # Compute loss
            loss = F.mse_loss(pred_positions, target_positions)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            num_train_batches += 1
        
        avg_train_loss = total_train_loss / num_train_batches
        
        # Validation
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                observations = batch['observations'].to(device)
                positions = batch['positions'].to(device)
                target_positions = batch['target_positions'].to(device)
                
                if 'PositionOnly' in model.__class__.__name__:
                    pred_positions = model(positions)
                else:
                    pred_positions = model(observations)
                
                loss = F.mse_loss(pred_positions, target_positions)
                total_val_loss += loss.item()
                num_val_batches += 1
        
        val_loss = total_val_loss / num_val_batches
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
            print(f"Saved best model at epoch {epoch} with val loss {best_val_loss:.6f}")
        
        # Save checkpoint
        if epoch % 50 == 0 and epoch > 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'checkpoint_epoch{epoch}.pth'))
            print(f"Saved checkpoint at epoch {epoch}")
        
        print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': val_loss
        })
    
    wandb.finish()
    return model


if __name__ == "__main__":
    # Parameters
    parameters = {
        'num_epochs': 200,
        'batch_size': 32,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'seq_length': 5,
        'hidden_dim': 128,
        'num_layers': 2
    }
    
    seeds = [0, 1000, 2000, 3000, 4000]
    env = 'tarware-medium-19agvs-9pickers-partialobs-v1'
    file_paths = [f'./warehouse_data_{env}_seed{s}.h5' for s in seeds]
    
    # Load dataset
    print("Loading dataset...")
    dataset = ConcatDataset([SequenceDataset(fp, seq_length=parameters['seq_length']) for fp in file_paths])
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=parameters['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=parameters['batch_size'], shuffle=False)
    
    # Get dimensions from first sample
    sample = train_dataset[0]
    obs_dim = sample['observations'].shape[-1]
    num_agents = sample['observations'].shape[-2]
    
    print(f"Observation dim: {obs_dim}, Number of agents: {num_agents}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Train all baseline models
    models_to_train = [
        ('GRU', GRUTrajectoryPredictor(obs_dim, num_agents, parameters['hidden_dim'], parameters['num_layers'])),
        ('LSTM', LSTMTrajectoryPredictor(obs_dim, num_agents, parameters['hidden_dim'], parameters['num_layers'])),
        ('GRU_PosOnly', PositionOnlyGRU(num_agents, parameters['hidden_dim'], parameters['num_layers'])),
        ('LSTM_PosOnly', PositionOnlyLSTM(num_agents, parameters['hidden_dim'], parameters['num_layers']))
    ]
    
    for model_name, model in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print(f"{'='*60}\n")
        
        save_path = f'./trained_models/{env}/{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}/'
        os.makedirs(save_path, exist_ok=True)
        
        trained_model = train_baseline_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=model_name,
            parameters=parameters,
            save_path=save_path,
            device=device
        )
        
        print(f"\nFinished training {model_name}")
        print(f"Best model saved at: {save_path}")