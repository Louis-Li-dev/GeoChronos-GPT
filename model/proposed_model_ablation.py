import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2LMHeadModel, 
    GPT2Config,
    AdamW,
    get_linear_schedule_with_warmup
)
from typing import List, Dict, Set, Optional
import pickle
import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class SpatioTemporalEncoder(nn.Module):
    """Encodes spatial (lat, lon) and temporal features into embeddings."""
    
    def __init__(self, hidden_dim: int = 768, use_temporal: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_temporal = use_temporal
        
        # Spatial encoding layers
        self.spatial_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        if self.use_temporal:
            # Temporal encoding - extract hour, day_of_week, month features
            self.temporal_encoder = nn.Sequential(
                nn.Linear(4, hidden_dim // 2),  # hour, day_of_week, month, time_since_epoch
                nn.ReLU(), 
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
            
            # Fusion layer for spatial + temporal
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            # No fusion needed for spatial-only
            self.fusion = None
        
    def encode_temporal_features(self, timestamps):
        """Convert timestamps to temporal features."""
        if not self.use_temporal:
            # Return dummy temporal features
            batch_size = len(timestamps)
            return np.zeros((batch_size, 4), dtype=np.float32)
            
        # Handle pandas datetime64 or datetime objects
        if hasattr(timestamps, 'dt'):  # pandas datetime series
            dt_series = timestamps
        elif isinstance(timestamps, pd.Series):
            dt_series = pd.to_datetime(timestamps)
        else:
            dt_series = pd.to_datetime(timestamps)
        
        # Extract features
        hours = dt_series.dt.hour.values / 24.0  # Normalize to [0, 1]
        days = dt_series.dt.dayofweek.values / 7.0
        months = dt_series.dt.month.values / 12.0
        
        # Time since epoch (normalized)
        epochs = dt_series.astype('int64').values // 10**9  # Convert to seconds
        epochs = (epochs - epochs.min()) / (epochs.max() - epochs.min() + 1e-8)
        
        return np.stack([hours, days, months, epochs], axis=1)
        
    def forward(self, spatial_coords, temporal_features):
        """
        Args:
            spatial_coords: (batch, seq_len, 2) - lat, lon
            temporal_features: (batch, seq_len, 4) - hour, day, month, epoch_norm (ignored if use_temporal=False)
        """
        batch_size, seq_len = spatial_coords.shape[:2]
        
        # Ensure contiguous tensors before reshaping
        spatial_coords = spatial_coords.contiguous()
        
        # Encode spatial
        spatial_emb = self.spatial_encoder(spatial_coords.reshape(-1, 2))
        spatial_emb = spatial_emb.reshape(batch_size, seq_len, -1)
        
        if not self.use_temporal:
            # Return only spatial embeddings
            return spatial_emb
        
        # Encode temporal
        temporal_features = temporal_features.contiguous()
        temporal_emb = self.temporal_encoder(temporal_features.reshape(-1, 4))
        temporal_emb = temporal_emb.reshape(batch_size, seq_len, -1)
        
        # Fuse spatial and temporal
        combined = torch.cat([spatial_emb, temporal_emb], dim=-1)
        fused = self.fusion(combined.reshape(-1, self.hidden_dim * 2))
        
        return fused.reshape(batch_size, seq_len, self.hidden_dim)


class MaskedPOIDataset(Dataset):
    """Custom dataset for POI sequence data with spatial-temporal features and user-specific masking."""
    
    def __init__(self, sequences, spatial_coords, temporal_features, user_ids, max_length=512):
        self.sequences = sequences
        self.spatial_coords = spatial_coords
        self.temporal_features = temporal_features
        self.user_ids = user_ids
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        coords = self.spatial_coords[idx]
        temp_feat = self.temporal_features[idx]
        user_id = self.user_ids[idx]
        
        # Truncate if too long
        if len(seq) > self.max_length:
            seq = seq[-self.max_length:]
            coords = coords[-self.max_length:]
            temp_feat = temp_feat[-self.max_length:]
            
        return {
            'input_ids': torch.tensor(seq, dtype=torch.long),
            'spatial_coords': torch.tensor(coords, dtype=torch.float32),
            'temporal_features': torch.tensor(temp_feat, dtype=torch.float32),
            'attention_mask': torch.ones(len(seq), dtype=torch.long),
            'user_id': user_id
        }


class AblationSpatioTemporalGPT(nn.Module):
    """GPT model with configurable spatio-temporal encoding and user-specific masking for ablation study."""
    
    def __init__(self, vocab_size: int, hidden_dim: int = 768, num_layers: int = 12, 
                 num_heads: int = 12, max_length: int = 512, use_temporal: bool = True,
                 use_personal_masking: bool = True):
        super().__init__()
        
        self.use_temporal = use_temporal
        self.use_personal_masking = use_personal_masking
        
        # GPT2 configuration
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=hidden_dim,
            n_layer=num_layers,
            n_head=num_heads,
            n_positions=max_length,
            pad_token_id=0
        )
        
        self.gpt = GPT2LMHeadModel(config)
        self.spatio_temporal_encoder = SpatioTemporalEncoder(hidden_dim, use_temporal)
        
        # Attention mechanism to combine GPT embeddings with spatio-temporal features
        self.context_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads // 2,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, spatial_coords, temporal_features, attention_mask=None, 
                user_poi_masks=None):
        """
        Forward pass with optional user-specific POI masking.
        
        Args:
            user_poi_masks: (batch_size, vocab_size) tensor where 1 indicates allowed POIs
        """
        # Get GPT embeddings
        gpt_outputs = self.gpt.transformer(input_ids, attention_mask=attention_mask)
        gpt_hidden = gpt_outputs.last_hidden_state
        
        # Get spatio-temporal embeddings (temporal part ignored if use_temporal=False)
        st_embeddings = self.spatio_temporal_encoder(spatial_coords, temporal_features)
        
        # Combine using attention
        enhanced_hidden, _ = self.context_attention(
            gpt_hidden, st_embeddings, st_embeddings,
            key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )
        
        # Residual connection and layer norm
        combined_hidden = self.layer_norm(gpt_hidden + enhanced_hidden)
        combined_hidden = self.dropout(combined_hidden)
        
        # Get logits
        logits = self.gpt.lm_head(combined_hidden)
        
        # Apply user-specific POI masking only if enabled
        if self.use_personal_masking and user_poi_masks is not None:
            # Expand mask to match logits dimensions
            # user_poi_masks: (batch_size, vocab_size)
            # logits: (batch_size, seq_len, vocab_size)
            mask_expanded = user_poi_masks.unsqueeze(1).expand_as(logits)
            
            # Set logits for disallowed POIs to very large negative value
            logits = logits.masked_fill(~mask_expanded.bool(), -1e9)
        
        return logits


class AblationPOIGPTRecommender:
    """POI recommender with configurable features for ablation study."""
    
    def __init__(self, hidden_dim: int = 768, num_layers: int = 8, 
                 num_heads: int = 12, max_length: int = 512, device: str = None,
                 use_temporal: bool = True, use_personal_masking: bool = True):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length
        self.use_temporal = use_temporal
        self.use_personal_masking = use_personal_masking
        
        # Will be set during fit
        self.model = None
        self.category_to_id = {}
        self.id_to_category = {}
        self.user_to_id = {}
        self.id_to_user = {}
        self.spatial_mean = None
        self.spatial_std = None
        self.user_last_sequences = {}  # Store last sequence for each user
        
        # User-specific POI sets for masking (only used if use_personal_masking=True)
        self.user_poi_sets: Dict[int, Set[int]] = {}  # user_id -> set of POI category IDs
        
        config_str = f"temporal={use_temporal}, masking={use_personal_masking}"
        print(f"Initialized Ablation POI GPT Recommender on {self.device} ({config_str})")
        
    def _prepare_data(self, df: pd.DataFrame):
        """Convert dataframe to sequences and mappings, optionally recording user-specific POI sets."""
        print(f"Processing {len(df)} records for {df['user_id'].nunique()} unique users...")
        
        # Create category mappings
        unique_categories = sorted(df['category'].unique())
        self.category_to_id = {cat: i+1 for i, cat in enumerate(unique_categories)}  # +1 for padding
        self.id_to_category = {i: cat for cat, i in self.category_to_id.items()}
        print(f"Found {len(unique_categories)} unique categories")
        
        # Create user mappings
        unique_users = sorted(df['user_id'].unique())
        self.user_to_id = {user: i for i, user in enumerate(unique_users)}
        self.id_to_user = {i: user for user, i in self.user_to_id.items()}
        
        # Initialize user POI sets only if personal masking is enabled
        if self.use_personal_masking:
            self.user_poi_sets = {}
        
        # Group by user and sort by time
        user_sequences = []
        user_spatial = []
        user_temporal = []
        user_ids = []
        
        # Calculate global spatial stats for normalization
        all_coords = df[['lat', 'lon']].values
        self.spatial_mean = all_coords.mean(axis=0)
        self.spatial_std = all_coords.std(axis=0) + 1e-8
        
        encoder = SpatioTemporalEncoder(use_temporal=self.use_temporal)
        
        print("Processing user sequences...")
        for user_id in tqdm(unique_users):
            user_data = df[df['user_id'] == user_id].sort_values('utc_time')
            
            # Skip users with too few visits
            if len(user_data) < 3:
                continue
                
            # Category sequence
            categories = [self.category_to_id[cat] for cat in user_data['category']]
            user_sequences.append(categories)
            user_ids.append(user_id)
            
            # Record user's POI set only if personal masking is enabled
            if self.use_personal_masking:
                user_poi_set = set(categories)
                user_poi_set.add(0)  # Always allow padding token
                self.user_poi_sets[user_id] = user_poi_set
            
            # Spatial coordinates (normalize using global stats)
            coords = user_data[['lat', 'lon']].values.astype(np.float32)
            coords = (coords - self.spatial_mean) / self.spatial_std
            user_spatial.append(coords)
            
            # Temporal features (will be dummy if use_temporal=False)
            temp_features = encoder.encode_temporal_features(user_data['utc_time'])
            user_temporal.append(temp_features.astype(np.float32))
            
            # Store the last sequence for each user for future prediction
            self.user_last_sequences[user_id] = {
                'categories': categories,
                'coords': coords,
                'temp_features': temp_features.astype(np.float32)
            }
            
        print(f"Prepared {len(user_sequences)} user sequences")
        if self.use_personal_masking:
            print(f"Average POIs per user: {np.mean([len(poi_set) for poi_set in self.user_poi_sets.values()]):.1f}")
        else:
            print("Personal masking disabled - all users can access all POIs")
            
        return user_sequences, user_spatial, user_temporal, user_ids
    
    def _create_user_poi_mask(self, user_ids: List[int], vocab_size: int) -> Optional[torch.Tensor]:
        """Create a mask tensor for user-specific POIs. Returns None if personal masking is disabled."""
        if not self.use_personal_masking:
            return None
            
        batch_size = len(user_ids)
        mask = torch.zeros(batch_size, vocab_size, dtype=torch.bool, device=self.device)
        
        for i, user_id in enumerate(user_ids):
            if user_id in self.user_poi_sets:
                allowed_pois = list(self.user_poi_sets[user_id])
                mask[i, allowed_pois] = True
            else:
                # If user not found, allow all POIs (shouldn't happen in normal cases)
                mask[i, :] = True
                
        return mask
    
    def fit(self, train_df: pd.DataFrame, epochs: int = 5, batch_size: int = 16, 
            lr: float = 5e-5):
        """Train the model on the training data with configurable features."""
        
        print("Preparing training data...")
        sequences, spatial_coords, temporal_features, user_ids = self._prepare_data(train_df)
        
        # Initialize model with ablation settings
        vocab_size = len(self.category_to_id) + 1  # +1 for padding
        self.model = AblationSpatioTemporalGPT(
            vocab_size=vocab_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            max_length=self.max_length,
            use_temporal=self.use_temporal,
            use_personal_masking=self.use_personal_masking
        ).to(self.device)
        
        # Create dataset and dataloader
        train_dataset = MaskedPOIDataset(sequences, spatial_coords, temporal_features, 
                                       user_ids, self.max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 collate_fn=self._collate_fn)
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=lr)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                   num_warmup_steps=total_steps//10,
                                                   num_training_steps=total_steps)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in pbar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                spatial_coords = batch['spatial_coords'].to(self.device)
                temporal_features = batch['temporal_features'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                batch_user_ids = batch['user_ids']  # List of user IDs
                
                # Create user-specific POI masks (None if personal masking disabled)
                user_poi_masks = self._create_user_poi_mask(batch_user_ids, vocab_size)
                
                # Shift for next-token prediction
                if input_ids.size(1) > 1:
                    inputs = input_ids[:, :-1]
                    targets = input_ids[:, 1:]
                    mask = attention_mask[:, :-1]
                    coords = spatial_coords[:, :-1]
                    temp_feat = temporal_features[:, :-1]
                    
                    logits = self.model(inputs, coords, temp_feat, mask, user_poi_masks)
                    
                    # Calculate loss
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), 
                        targets.reshape(-1),
                        ignore_index=0
                    )
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
                    
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        print("Training completed!")
        return self
    
    def _collate_fn(self, batch):
        """Custom collate function to handle variable length sequences."""
        max_len = max(len(item['input_ids']) for item in batch)
        
        input_ids = []
        spatial_coords = []
        temporal_features = []
        attention_masks = []
        user_ids = []
        
        for item in batch:
            seq_len = len(item['input_ids'])
            pad_len = max_len - seq_len
            
            # Pad sequences
            padded_ids = F.pad(item['input_ids'], (0, pad_len), value=0)
            padded_mask = F.pad(item['attention_mask'], (0, pad_len), value=0)
            
            # Pad spatial and temporal features
            padded_coords = F.pad(item['spatial_coords'], (0, 0, 0, pad_len), value=0.0)
            padded_temp = F.pad(item['temporal_features'], (0, 0, 0, pad_len), value=0.0)
            
            input_ids.append(padded_ids)
            attention_masks.append(padded_mask)
            spatial_coords.append(padded_coords)
            temporal_features.append(padded_temp)
            user_ids.append(item['user_id'])
        
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks),
            'spatial_coords': torch.stack(spatial_coords),
            'temporal_features': torch.stack(temporal_features),
            'user_ids': user_ids
        }
    
    @torch.no_grad()
    def predict(
        self,
        n_tokens: int = 10,
        temperature: float = 1.0
    ) -> Dict[int, List[str]]:
        """
        Autoregressively generate `n_tokens` future category tokens for every user
        seen during training, using configurable masking.

        Returns
        -------
        Dict[int, List[str]]
            Mapping: user_id -> list of predicted category strings
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        self.model.eval()
        predictions: Dict[int, List[str]] = {}
        vocab_size = len(self.category_to_id) + 1

        # Iterate only over users for whom we've stored a sequence
        for user_id, seq_info in tqdm(
            self.user_last_sequences.items(), desc="Generating predictions"
        ):
            categories = seq_info["categories"]
            coords     = seq_info["coords"]
            temp_feat  = seq_info["temp_features"]

            # Need at least two tokens to start autoregression
            if len(categories) < 2:
                continue

            # Move data to device
            input_ids        = torch.tensor([categories], dtype=torch.long,   device=self.device)
            spatial_coords   = torch.tensor([coords],     dtype=torch.float32, device=self.device)
            temporal_features= torch.tensor([temp_feat],  dtype=torch.float32, device=self.device)
            
            # Create user-specific mask (None if personal masking disabled)
            user_poi_mask = self._create_user_poi_mask([user_id], vocab_size)

            # Generate `n_tokens` new IDs
            generated_ids = self._generate(
                input_ids=input_ids,
                spatial_coords=spatial_coords,
                temporal_features=temporal_features,
                user_poi_mask=user_poi_mask,
                n_tokens=n_tokens,
                temperature=temperature,
            )

            # Take only the freshly generated part
            new_token_ids: List[int] = generated_ids[0, -n_tokens:].tolist()

            # Map back to category strings
            predicted_categories: List[str] = [
                self.id_to_category[token_id]
                for token_id in new_token_ids
                if token_id != 0 and token_id in self.id_to_category  # skip PAD / OOV
            ]

            predictions[user_id] = predicted_categories

        return predictions

    
    def _generate(self, input_ids, spatial_coords, temporal_features, user_poi_mask, 
                  n_tokens, temperature=1.0):
        """Generate new tokens autoregressively with configurable masking."""
        generated = input_ids.clone()
        
        for _ in range(n_tokens):
            # Use last part of sequence if too long
            if generated.size(1) > self.max_length - 1:
                curr_input = generated[:, -(self.max_length-1):]
                curr_coords = spatial_coords[:, -(self.max_length-1):]
                curr_temp = temporal_features[:, -(self.max_length-1):]
            else:
                curr_input = generated
                curr_coords = spatial_coords[:, :generated.size(1)]
                curr_temp = temporal_features[:, :generated.size(1)]
            
            # Get predictions with configurable masking
            logits = self.model(curr_input, curr_coords, curr_temp, 
                              user_poi_masks=user_poi_mask)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply softmax and sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Extend spatial/temporal with last values
            last_coord = spatial_coords[:, -1:, :]
            last_temp = temporal_features[:, -1:, :]
            spatial_coords = torch.cat([spatial_coords, last_coord], dim=1)
            temporal_features = torch.cat([temporal_features, last_temp], dim=1)
        
        return generated
    
    def get_user_poi_stats(self) -> Dict[str, float]:
        """Get statistics about user POI sets."""
        if not self.use_personal_masking or not self.user_poi_sets:
            return {'personal_masking': False}
        
        poi_counts = [len(poi_set) for poi_set in self.user_poi_sets.values()]
        total_pois = len(self.category_to_id)
        
        return {
            'personal_masking': True,
            'total_categories': total_pois,
            'avg_pois_per_user': np.mean(poi_counts),
            'min_pois_per_user': np.min(poi_counts),
            'max_pois_per_user': np.max(poi_counts),
            'median_pois_per_user': np.median(poi_counts),
            'coverage_ratio': np.mean(poi_counts) / total_pois
        }
    
    def get_config(self) -> Dict[str, any]:
        """Get the current configuration for ablation study tracking."""
        return {
            'use_temporal': self.use_temporal,
            'use_personal_masking': self.use_personal_masking,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'max_length': self.max_length
        }
    
    def save(self, path: str):
        """Save the model, mappings, and configuration."""
        os.makedirs(path, exist_ok=True)
        
        # Save model
        if self.model is not None:
            torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))
        
        # Convert numpy types to Python types for JSON serialization
        def convert_keys_values(d):
            """Convert numpy types to Python types."""
            if isinstance(d, dict):
                return {str(k): (int(v) if isinstance(v, (np.integer, np.int32, np.int64)) else v) 
                       for k, v in d.items()}
            return d
        
        # Save mappings and config
        config = {
            'category_to_id': convert_keys_values(self.category_to_id),
            'id_to_category': convert_keys_values(self.id_to_category),
            'user_to_id': convert_keys_values(self.user_to_id),
            'id_to_user': convert_keys_values(self.id_to_user),
            'hidden_dim': int(self.hidden_dim),
            'num_layers': int(self.num_layers),
            'num_heads': int(self.num_heads),
            'max_length': int(self.max_length),
            'use_temporal': self.use_temporal,
            'use_personal_masking': self.use_personal_masking,
            'spatial_mean': self.spatial_mean.tolist() if self.spatial_mean is not None else None,
            'spatial_std': self.spatial_std.tolist() if self.spatial_std is not None else None
        }
        
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save user sequences
        with open(os.path.join(path, 'user_sequences.pkl'), 'wb') as f:
            pickle.dump(self.user_last_sequences, f)
        
        # Save user POI sets only if personal masking is enabled
        if self.use_personal_masking:
            user_poi_sets_serializable = {
                str(user_id): list(poi_set) for user_id, poi_set in self.user_poi_sets.items()
            }
            with open(os.path.join(path, 'user_poi_sets.pkl'), 'wb') as f:
                pickle.dump(user_poi_sets_serializable, f)
            
        print(f"Ablation model saved to {path}")
        print(f"Configuration: {self.get_config()}")
    
    def load(self, path: str):
        """Load the model, mappings, and configuration."""
        # Load config
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Convert string keys back to appropriate types
        self.category_to_id = config['category_to_id']
        self.id_to_category = {int(k): v for k, v in config['id_to_category'].items()}
        
        # Handle both string and int user IDs
        self.user_to_id = {}
        for k, v in config['user_to_id'].items():
            try:
                key = int(k) if k.isdigit() else k
            except (ValueError, AttributeError):
                key = k
            self.user_to_id[key] = v
            
        self.id_to_user = {int(k): v for k, v in config['id_to_user'].items()}
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.num_heads = config['num_heads']
        self.max_length = config['max_length']
        self.use_temporal = config.get('use_temporal', True)
        self.use_personal_masking = config.get('use_personal_masking', True)
        self.spatial_mean = np.array(config['spatial_mean']) if config.get('spatial_mean') else None
        self.spatial_std = np.array(config['spatial_std']) if config.get('spatial_std') else None
        
        # Load user sequences
        with open(os.path.join(path, 'user_sequences.pkl'), 'rb') as f:
            self.user_last_sequences = pickle.load(f)
        
        # Load user POI sets only if personal masking is enabled
        if self.use_personal_masking:
            poi_sets_path = os.path.join(path, 'user_poi_sets.pkl')
            if os.path.exists(poi_sets_path):
                with open(poi_sets_path, 'rb') as f:
                    user_poi_sets_loaded = pickle.load(f)
                    
                # Convert back to proper format
                self.user_poi_sets = {}
                for user_id_str, poi_list in user_poi_sets_loaded.items():
                    try:
                        user_id = int(user_id_str) if user_id_str.isdigit() else user_id_str
                    except (ValueError, AttributeError):
                        user_id = user_id_str
                    self.user_poi_sets[user_id] = set(poi_list)
        else:
            self.user_poi_sets = {}
        
        # Initialize and load model
        vocab_size = len(self.category_to_id) + 1
        self.model = AblationSpatioTemporalGPT(
            vocab_size=vocab_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            max_length=self.max_length,
            use_temporal=self.use_temporal,
            use_personal_masking=self.use_personal_masking
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(os.path.join(path, 'model.pt'), 
                                            map_location=self.device))
        
        print(f"Ablation model loaded from {path}")
        print(f"Configuration: {self.get_config()}")
        print(f"User POI masking statistics: {self.get_user_poi_stats()}")
        return self




# Example usage for ablation study:
"""
# Four different configurations for ablation study

# 1. Full model (temporal + personal masking)
model_full = create_full_model(hidden_dim=768, num_layers=8)

# 2. No temporal (spatial only + personal masking)  
model_no_temporal = create_no_temporal_model(hidden_dim=768, num_layers=8)

# 3. No personal masking (temporal + spatial, no user-specific masking)
model_no_masking = create_no_masking_model(hidden_dim=768, num_layers=8)

# 4. Baseline (spatial only, no personal masking)
model_baseline = create_baseline_model(hidden_dim=768, num_layers=8)

# Train each model
models = {
    'full': model_full,
    'no_temporal': model_no_temporal, 
    'no_masking': model_no_masking,
    'baseline': model_baseline
}

for name, model in models.items():
    print(f"Training {name} model...")
    model.fit(train_df, epochs=5)
    model.save(f'models/{name}')
    
    # Generate predictions
    predictions = model.predict(n_tokens=10)
    print(f"{name} model completed!")
"""