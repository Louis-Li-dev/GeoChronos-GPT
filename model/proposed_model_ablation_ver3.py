import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2LMHeadModel, 
    GPT2Config,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
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
            
            # SIMPLIFIED: Direct fusion without complex attention
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.LayerNorm(hidden_dim)
            )
        else:
            self.fusion = None
        
    def encode_temporal_features(self, timestamps):
        """Convert timestamps to temporal features."""
        if not self.use_temporal:
            batch_size = len(timestamps)
            return np.zeros((batch_size, 4), dtype=np.float32)
            
        if hasattr(timestamps, 'dt'):
            dt_series = timestamps
        elif isinstance(timestamps, pd.Series):
            dt_series = pd.to_datetime(timestamps)
        else:
            dt_series = pd.to_datetime(timestamps)
        
        # Extract features - FIXED: Use consistent normalization
        hours = dt_series.dt.hour.values / 23.0  # 0-23 -> [0,1]
        days = dt_series.dt.dayofweek.values / 6.0  # 0-6 -> [0,1]
        months = (dt_series.dt.month.values - 1) / 11.0  # 1-12 -> [0,1]
        
        # Time since epoch (normalized by dataset range)
        epochs = dt_series.astype('int64').values // 10**9
        if len(epochs) > 1:
            epochs = (epochs - epochs.min()) / (epochs.max() - epochs.min() + 1e-8)
        else:
            epochs = np.zeros_like(epochs, dtype=np.float32)
        
        return np.stack([hours, days, months, epochs], axis=1)
        
    def forward(self, spatial_coords, temporal_features, use_temporal_for_batch=None):
        """
        Args:
            spatial_coords: (batch, seq_len, 2) - lat, lon
            temporal_features: (batch, seq_len, 4) - hour, day, month, epoch_norm
            use_temporal_for_batch: (batch,) - boolean mask for using temporal per sample
        """
        batch_size, seq_len = spatial_coords.shape[:2]
        
        spatial_coords = spatial_coords.contiguous()
        spatial_emb = self.spatial_encoder(spatial_coords.reshape(-1, 2))
        spatial_emb = spatial_emb.reshape(batch_size, seq_len, -1)
        
        if not self.use_temporal:
            return spatial_emb
        
        # Encode temporal
        temporal_features = temporal_features.contiguous()
        temporal_emb = self.temporal_encoder(temporal_features.reshape(-1, 4))
        temporal_emb = temporal_emb.reshape(batch_size, seq_len, -1)
        
        # Apply user-aware temporal masking if provided
        if use_temporal_for_batch is not None:
            temporal_mask = use_temporal_for_batch.unsqueeze(1).unsqueeze(2)
            temporal_mask = temporal_mask.expand(batch_size, seq_len, self.hidden_dim)
            temporal_emb = temporal_emb * temporal_mask.float()
        
        # Fuse spatial and temporal
        combined = torch.cat([spatial_emb, temporal_emb], dim=-1)
        fused = self.fusion(combined.reshape(-1, self.hidden_dim * 2))
        
        return fused.reshape(batch_size, seq_len, self.hidden_dim)


class MaskedPOIDataset(Dataset):
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
    """SIMPLIFIED GPT model with configurable features."""
    
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
        
        # SIMPLIFIED: Replace complex attention with simple residual fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Learnable fusion weight
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, input_ids, spatial_coords, temporal_features, attention_mask=None, 
                user_poi_masks=None, use_temporal_for_batch=None):
        # Get GPT embeddings
        gpt_outputs = self.gpt.transformer(input_ids, attention_mask=attention_mask)
        gpt_hidden = gpt_outputs.last_hidden_state
        
        # Get spatio-temporal embeddings
        st_embeddings = self.spatio_temporal_encoder(
            spatial_coords, temporal_features, use_temporal_for_batch
        )
        
        # SIMPLIFIED FUSION: Concatenate and project, then blend with original
        combined = torch.cat([gpt_hidden, st_embeddings], dim=-1)
        enhanced = self.fusion_layer(combined)
        
        # Learnable weighted combination instead of attention
        fusion_alpha = torch.sigmoid(self.fusion_weight)
        combined_hidden = fusion_alpha * enhanced + (1 - fusion_alpha) * gpt_hidden
        
        # Get logits
        logits = self.gpt.lm_head(combined_hidden)
        
        # Apply user-specific POI masking
        if self.use_personal_masking and user_poi_masks is not None:
            mask_expanded = user_poi_masks.unsqueeze(1).expand_as(logits)
            # LESS AGGRESSIVE MASKING: Use -1e4 instead of -1e9 to avoid numerical issues
            logits = logits.masked_fill(~mask_expanded.bool(), -1e4)
        
        return logits


class AblationPOIGPTRecommender:
    """POI recommender with simplified and fixed ablation study."""
    
    def __init__(self, hidden_dim: int = 768, num_layers: int = 8, 
                 num_heads: int = 12, max_length: int = 512, device: str = None,
                 use_temporal: bool = True, use_personal_masking: bool = True,
                 user_aware_temporal_fusion: bool = False, temporal_overlap_threshold: float = 0.1):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length
        self.use_temporal = use_temporal
        self.use_personal_masking = use_personal_masking
        self.user_aware_temporal_fusion = user_aware_temporal_fusion
        self.temporal_overlap_threshold = temporal_overlap_threshold  # MUCH MORE LENIENT
        
        # Will be set during fit
        self.model = None
        self.category_to_id = {}
        self.id_to_category = {}
        self.user_to_id = {}
        self.id_to_user = {}
        self.spatial_mean = None
        self.spatial_std = None
        self.user_last_sequences = {}
        
        # User-specific POI sets for masking
        self.user_poi_sets: Dict[int, Set[int]] = {}
        
        # Training temporal patterns for user-aware fusion
        self.user_training_temporal_patterns = {}
        
        config_str = f"temporal={use_temporal}, masking={use_personal_masking}, user_aware={user_aware_temporal_fusion}"
        print(f"Initialized Ablation POI GPT Recommender on {self.device} ({config_str})")

    def _prepare_data(self, df: pd.DataFrame):
        """Convert dataframe to sequences and mappings."""
        print(f"Processing {len(df)} records for {df['user_id'].nunique()} unique users...")
        
        # Create category mappings
        unique_categories = sorted(df['category'].unique())
        self.category_to_id = {cat: i+1 for i, cat in enumerate(unique_categories)}
        self.id_to_category = {i: cat for cat, i in self.category_to_id.items()}
        print(f"Found {len(unique_categories)} unique categories")
        
        # Create user mappings
        unique_users = sorted(df['user_id'].unique())
        self.user_to_id = {user: i for i, user in enumerate(unique_users)}
        self.id_to_user = {i: user for user, i in self.user_to_id.items()}
        
        if self.use_personal_masking:
            self.user_poi_sets = {}
        
        user_sequences = []
        user_spatial = []
        user_temporal = []
        user_ids = []
        
        # Calculate global spatial stats
        all_coords = df[['lat', 'lon']].values
        self.spatial_mean = all_coords.mean(axis=0)
        self.spatial_std = all_coords.std(axis=0) + 1e-8
        
        encoder = SpatioTemporalEncoder(use_temporal=self.use_temporal)
        self.user_training_temporal_patterns = {}
        
        print("Processing user sequences...")
        for user_id in tqdm(unique_users):
            user_data = df[df['user_id'] == user_id].sort_values('utc_time')
            
            if len(user_data) < 3:
                continue
                
            categories = [self.category_to_id[cat] for cat in user_data['category']]
            user_sequences.append(categories)
            user_ids.append(user_id)
            
            # Record user's POI set with LESS RESTRICTIVE masking
            if self.use_personal_masking:
                user_poi_set = set(categories)
                user_poi_set.add(0)  # Always allow padding token
                
                # LESS RESTRICTIVE: Allow popular POIs even if user hasn't visited
                category_counts = df['category'].value_counts()
                popular_pois = category_counts.head(20).index  # Top 20 popular POIs
                for poi_name in popular_pois:
                    if poi_name in self.category_to_id:
                        user_poi_set.add(self.category_to_id[poi_name])
                        
                self.user_poi_sets[user_id] = user_poi_set
            
            # Spatial coordinates
            coords = user_data[['lat', 'lon']].values.astype(np.float32)
            coords = (coords - self.spatial_mean) / self.spatial_std
            user_spatial.append(coords)
            
            # Temporal features
            temp_features = encoder.encode_temporal_features(user_data['utc_time'])
            user_temporal.append(temp_features.astype(np.float32))
            
            # Store temporal patterns for user-aware fusion
            if self.use_temporal and self.user_aware_temporal_fusion:
                # FIXED: Use quantized temporal features for better overlap detection
                unique_hours = set(np.round(temp_features[:, 0] * 23).astype(int))  # 0-23 hours
                unique_days = set(np.round(temp_features[:, 1] * 6).astype(int))    # 0-6 days
                
                self.user_training_temporal_patterns[user_id] = {
                    'hours': unique_hours,
                    'days': unique_days,
                }
            
            self.user_last_sequences[user_id] = {
                'categories': categories,
                'coords': coords,
                'temp_features': temp_features.astype(np.float32)
            }
                
        print(f"Prepared {len(user_sequences)} user sequences")
        
        if self.use_personal_masking:
            avg_pois = np.mean([len(poi_set) for poi_set in self.user_poi_sets.values()])
            print(f"Average POIs per user (with popular POIs): {avg_pois:.1f}")
        
        if self.use_temporal and self.user_aware_temporal_fusion:
            print(f"Stored temporal patterns for {len(self.user_training_temporal_patterns)} users")
            
        return user_sequences, user_spatial, user_temporal, user_ids

    def _should_use_temporal_for_user(self, user_id: int, user_temporal_features: np.ndarray) -> bool:
        """MUCH MORE LENIENT temporal pattern matching."""
        if not self.use_temporal or not self.user_aware_temporal_fusion:
            return self.use_temporal
            
        if user_id not in self.user_training_temporal_patterns:
            return True  # Default to using temporal if no patterns stored
        
        training_patterns = self.user_training_temporal_patterns[user_id]
        training_hours = training_patterns['hours']
        training_days = training_patterns['days']
        
        # Convert current temporal features to same format
        current_hours = set(np.round(user_temporal_features[:, 0] * 23).astype(int))
        current_days = set(np.round(user_temporal_features[:, 1] * 6).astype(int))
        
        # Calculate overlap ratios
        hour_overlap_ratio = len(training_hours & current_hours) / len(current_hours) if current_hours else 0
        day_overlap_ratio = len(training_days & current_days) / len(current_days) if current_days else 0
        
        # MUCH MORE LENIENT: Use temporal if ANY meaningful overlap (10% threshold)
        should_use = (hour_overlap_ratio >= self.temporal_overlap_threshold or 
                     day_overlap_ratio >= self.temporal_overlap_threshold)
        
        return should_use

    def _create_user_poi_mask(self, user_ids: List[int], vocab_size: int) -> Optional[torch.Tensor]:
        """Create user-specific POI masks."""
        if not self.use_personal_masking:
            return None
            
        batch_size = len(user_ids)
        mask = torch.zeros(batch_size, vocab_size, dtype=torch.bool, device=self.device)
        
        for i, user_id in enumerate(user_ids):
            if user_id in self.user_poi_sets:
                allowed_pois = list(self.user_poi_sets[user_id])
                mask[i, allowed_pois] = True
            else:
                mask[i, :] = True  # Allow all POIs if user not found
                
        return mask
    
    def fit(self, train_df: pd.DataFrame, epochs: int = 5, batch_size: int = 16, 
            lr: float = 5e-5):
        """Train the model with CONSISTENT train/test temporal usage."""
        
        print("Preparing training data...")
        sequences, spatial_coords, temporal_features, user_ids = self._prepare_data(train_df)
        
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
        
        train_dataset = MaskedPOIDataset(sequences, spatial_coords, temporal_features, 
                                       user_ids, self.max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 collate_fn=self._collate_fn)
        
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)  # Added weight decay
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                   num_warmup_steps=total_steps//10,
                                                   num_training_steps=total_steps)
        
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
                batch_user_ids = batch['user_ids']
                
                user_poi_masks = self._create_user_poi_mask(batch_user_ids, vocab_size)
                
                # FIXED: Apply same user-aware logic during training as during inference
                use_temporal_for_batch = None
                if self.use_temporal and self.user_aware_temporal_fusion:
                    use_temporal_for_batch = torch.zeros(len(batch_user_ids), dtype=torch.bool, device=self.device)
                    for i, user_id in enumerate(batch_user_ids):
                        # During training, use the user's own temporal features to decide
                        user_temp_features = temporal_features[i].cpu().numpy()
                        should_use = self._should_use_temporal_for_user(user_id, user_temp_features)
                        use_temporal_for_batch[i] = should_use
                elif self.use_temporal:
                    use_temporal_for_batch = torch.ones(len(batch_user_ids), dtype=torch.bool, device=self.device)
                
                if input_ids.size(1) > 1:
                    inputs = input_ids[:, :-1]
                    targets = input_ids[:, 1:]
                    mask = attention_mask[:, :-1]
                    coords = spatial_coords[:, :-1]
                    temp_feat = temporal_features[:, :-1]
                    
                    logits = self.model(inputs, coords, temp_feat, mask, user_poi_masks, use_temporal_for_batch)
                    
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), 
                        targets.reshape(-1),
                        ignore_index=0,
                        label_smoothing=0.1  # Added label smoothing to reduce overfitting
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
        """Custom collate function."""
        max_len = max(len(item['input_ids']) for item in batch)
        
        input_ids = []
        spatial_coords = []
        temporal_features = []
        attention_masks = []
        user_ids = []
        
        for item in batch:
            seq_len = len(item['input_ids'])
            pad_len = max_len - seq_len
            
            padded_ids = F.pad(item['input_ids'], (0, pad_len), value=0)
            padded_mask = F.pad(item['attention_mask'], (0, pad_len), value=0)
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
    def predict(self, n_tokens: int = 10, temperature: float = 1.0, 
                test_df: Optional[pd.DataFrame] = None) -> Dict[int, List[str]]:
        """Generate predictions with consistent temporal logic."""
        
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        self.model.eval()
        predictions: Dict[int, List[str]] = {}
        vocab_size = len(self.category_to_id) + 1
        
        # Analyze temporal patterns for user-aware fusion
        user_temporal_decisions = {}
        temporal_usage_count = 0
        
        if self.use_temporal and self.user_aware_temporal_fusion and test_df is not None:
            print("Analyzing user-specific temporal patterns...")
            encoder = SpatioTemporalEncoder(use_temporal=True)
            
            for user_id in test_df['user_id'].unique():
                if user_id in self.user_last_sequences:
                    user_test_data = test_df[test_df['user_id'] == user_id]
                    test_temp_features = encoder.encode_temporal_features(user_test_data['utc_time'])
                    should_use_temporal = self._should_use_temporal_for_user(user_id, test_temp_features)
                    user_temporal_decisions[user_id] = should_use_temporal
                    if should_use_temporal:
                        temporal_usage_count += 1
            
            if user_temporal_decisions:
                total_users = len(user_temporal_decisions)
                temporal_ratio = temporal_usage_count / total_users
                print(f"Temporal usage: {temporal_usage_count}/{total_users} ({temporal_ratio:.1%}) users using temporal features")

        # Generate predictions
        for user_id, seq_info in tqdm(self.user_last_sequences.items(), desc="Generating predictions"):
            categories = seq_info["categories"]
            coords = seq_info["coords"]
            temp_feat = seq_info["temp_features"]

            if len(categories) < 2:
                continue

            input_ids = torch.tensor([categories], dtype=torch.long, device=self.device)
            spatial_coords = torch.tensor([coords], dtype=torch.float32, device=self.device)
            temporal_features = torch.tensor([temp_feat], dtype=torch.float32, device=self.device)
            
            user_poi_mask = self._create_user_poi_mask([user_id], vocab_size)
            
            # Apply temporal decision
            use_temporal_for_batch = None
            if self.use_temporal and self.user_aware_temporal_fusion:
                should_use_temporal = user_temporal_decisions.get(user_id, True)
                use_temporal_for_batch = torch.tensor([should_use_temporal], dtype=torch.bool, device=self.device)

            generated_ids = self._generate(
                input_ids=input_ids,
                spatial_coords=spatial_coords,
                temporal_features=temporal_features,
                user_poi_mask=user_poi_mask,
                use_temporal_for_batch=use_temporal_for_batch,
                n_tokens=n_tokens,
                temperature=temperature,
            )

            new_token_ids: List[int] = generated_ids[0, -n_tokens:].tolist()
            predicted_categories: List[str] = [
                self.id_to_category[token_id]
                for token_id in new_token_ids
                if token_id != 0 and token_id in self.id_to_category
            ]

            predictions[user_id] = predicted_categories

        return predictions

    def _generate(self, input_ids, spatial_coords, temporal_features, user_poi_mask, 
                  use_temporal_for_batch, n_tokens, temperature=1.0):
        """Generate new tokens autoregressively."""
        generated = input_ids.clone()
        
        for _ in range(n_tokens):
            if generated.size(1) > self.max_length - 1:
                curr_input = generated[:, -(self.max_length-1):]
                curr_coords = spatial_coords[:, -(self.max_length-1):]
                curr_temp = temporal_features[:, -(self.max_length-1):]
            else:
                curr_input = generated
                curr_coords = spatial_coords[:, :generated.size(1)]
                curr_temp = temporal_features[:, :generated.size(1)]
            
            logits = self.model(curr_input, curr_coords, curr_temp, 
                              user_poi_masks=user_poi_mask,
                              use_temporal_for_batch=use_temporal_for_batch)
            next_token_logits = logits[:, -1, :] / temperature
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Extend spatial/temporal with last values
            last_coord = spatial_coords[:, -1:, :]
            last_temp = temporal_features[:, -1:, :]
            spatial_coords = torch.cat([spatial_coords, last_coord], dim=1)
            temporal_features = torch.cat([temporal_features, last_temp], dim=1)
        
        return generated

    def get_config(self) -> Dict[str, any]:
        """Get configuration."""
        return {
            'use_temporal': self.use_temporal,
            'use_personal_masking': self.use_personal_masking,
            'user_aware_temporal_fusion': self.user_aware_temporal_fusion,
            'temporal_overlap_threshold': self.temporal_overlap_threshold,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'max_length': self.max_length
        }

    def get_user_poi_stats(self) -> Dict[str, float]:
        """Get user POI statistics."""
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

    def save(self, path: str):
        """Save the model, mappings, and configuration."""
        os.makedirs(path, exist_ok=True)
        
        if self.model is not None:
            torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))
        
        def convert_keys_values(d):
            if isinstance(d, dict):
                return {str(k): (int(v) if isinstance(v, (np.integer, np.int32, np.int64)) else v) 
                    for k, v in d.items()}
            return d
        
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
            'user_aware_temporal_fusion': self.user_aware_temporal_fusion,
            'temporal_overlap_threshold': self.temporal_overlap_threshold,
            'spatial_mean': self.spatial_mean.tolist() if self.spatial_mean is not None else None,
            'spatial_std': self.spatial_std.tolist() if self.spatial_std is not None else None,
        }
        
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        with open(os.path.join(path, 'user_sequences.pkl'), 'wb') as f:
            pickle.dump(self.user_last_sequences, f)
        
        if self.use_personal_masking:
            user_poi_sets_serializable = {
                str(user_id): list(poi_set) for user_id, poi_set in self.user_poi_sets.items()
            }
            with open(os.path.join(path, 'user_poi_sets.pkl'), 'wb') as f:
                pickle.dump(user_poi_sets_serializable, f)
        
        if self.use_temporal and self.user_aware_temporal_fusion and hasattr(self, 'user_training_temporal_patterns'):
            temporal_patterns_serializable = {}
            for user_id, patterns in self.user_training_temporal_patterns.items():
                temporal_patterns_serializable[str(user_id)] = {
                    'hours': list(patterns['hours']),
                    'days': list(patterns['days']),
                }
            
            with open(os.path.join(path, 'user_temporal_patterns.pkl'), 'wb') as f:
                pickle.dump(temporal_patterns_serializable, f)
            
        print(f"Fixed POI model saved to {path}")
        print(f"Configuration: {self.get_config()}")

    def load(self, path: str):
        """Load the model, mappings, and configuration."""
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)
        
        self.category_to_id = config['category_to_id']
        self.id_to_category = {int(k): v for k, v in config['id_to_category'].items()}
        
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
        self.user_aware_temporal_fusion = config.get('user_aware_temporal_fusion', False)
        self.temporal_overlap_threshold = config.get('temporal_overlap_threshold', 0.1)
        self.spatial_mean = np.array(config['spatial_mean']) if config.get('spatial_mean') else None
        self.spatial_std = np.array(config['spatial_std']) if config.get('spatial_std') else None
        
        with open(os.path.join(path, 'user_sequences.pkl'), 'rb') as f:
            self.user_last_sequences = pickle.load(f)
        
        if self.use_personal_masking:
            poi_sets_path = os.path.join(path, 'user_poi_sets.pkl')
            if os.path.exists(poi_sets_path):
                with open(poi_sets_path, 'rb') as f:
                    user_poi_sets_loaded = pickle.load(f)
                    
                self.user_poi_sets = {}
                for user_id_str, poi_list in user_poi_sets_loaded.items():
                    try:
                        user_id = int(user_id_str) if user_id_str.isdigit() else user_id_str
                    except (ValueError, AttributeError):
                        user_id = user_id_str
                    self.user_poi_sets[user_id] = set(poi_list)
        else:
            self.user_poi_sets = {}
        
        temporal_patterns_path = os.path.join(path, 'user_temporal_patterns.pkl')
        if self.use_temporal and self.user_aware_temporal_fusion and os.path.exists(temporal_patterns_path):
            with open(temporal_patterns_path, 'rb') as f:
                temporal_patterns_loaded = pickle.load(f)
            
            self.user_training_temporal_patterns = {}
            for user_id_str, patterns in temporal_patterns_loaded.items():
                try:
                    user_id = int(user_id_str) if user_id_str.isdigit() else user_id_str
                except (ValueError, AttributeError):
                    user_id = user_id_str
                    
                self.user_training_temporal_patterns[user_id] = {
                    'hours': set(patterns['hours']),
                    'days': set(patterns['days']),
                }
            
            print(f"Loaded temporal patterns for {len(self.user_training_temporal_patterns)} users")
        else:
            self.user_training_temporal_patterns = {}
        
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
        
        print(f"Fixed POI model loaded from {path}")
        print(f"Configuration: {self.get_config()}")
        return self


# Factory functions for creating different ablation configurations
def create_full_model(hidden_dim: int = 512, num_layers: int = 6, **kwargs):
    """Create full model with all features enabled but smaller size."""
    return AblationPOIGPTRecommender(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_temporal=True,
        use_personal_masking=True,
        user_aware_temporal_fusion=True,
        temporal_overlap_threshold=0.1,  # Much more lenient
        **kwargs
    )

def create_no_temporal_model(hidden_dim: int = 512, num_layers: int = 6, **kwargs):
    """Create model without temporal features."""
    return AblationPOIGPTRecommender(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_temporal=False,
        use_personal_masking=True,
        user_aware_temporal_fusion=False,
        **kwargs
    )

def create_no_masking_model(hidden_dim: int = 512, num_layers: int = 6, **kwargs):
    """Create model without personal masking."""
    return AblationPOIGPTRecommender(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_temporal=True,
        use_personal_masking=False,
        user_aware_temporal_fusion=True,
        temporal_overlap_threshold=0.1,
        **kwargs
    )

def create_baseline_model(hidden_dim: int = 512, num_layers: int = 6, **kwargs):
    """Create baseline model (spatial only, no masking)."""
    return AblationPOIGPTRecommender(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_temporal=False,
        use_personal_masking=False,
        user_aware_temporal_fusion=False,
        **kwargs
    )

def create_simplified_full_model(hidden_dim: int = 512, num_layers: int = 6, **kwargs):
    """Create full model without user-aware temporal fusion."""
    return AblationPOIGPTRecommender(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_temporal=True,
        use_personal_masking=True,
        user_aware_temporal_fusion=False,  # Disable the problematic feature
        **kwargs
    )


# Example usage with SMALLER MODELS to avoid overfitting:
"""
# Create models with smaller dimensions to reduce overfitting
models = {
    'full': create_full_model(hidden_dim=512, num_layers=6),
    'simplified_full': create_simplified_full_model(hidden_dim=512, num_layers=6),
    'no_temporal': create_no_temporal_model(hidden_dim=512, num_layers=6),
    'no_masking': create_no_masking_model(hidden_dim=512, num_layers=6),
    'baseline': create_baseline_model(hidden_dim=512, num_layers=6)
}

# Train each model
for name, model in models.items():
    print(f"Training {name} model...")
    model.fit(train_df, epochs=3, batch_size=32, lr=1e-4)  # Reduced epochs and lr
    model.save(f'models_fixed/{name}')
    
    # Generate predictions
    if name == 'full':
        predictions = model.predict(n_tokens=10, test_df=test_df)
    else:
        predictions = model.predict(n_tokens=10)
    
    print(f"{name} model completed with {len(predictions)} predictions!")
"""