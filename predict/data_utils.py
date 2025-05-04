import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.utils import remove_self_loops, k_hop_subgraph
import numpy as np
import pickle
import logging
import os
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from imblearn.over_sampling import SMOTE
from collections import Counter

# Setup logging
def setup_logging(log_path):
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
    )

# Data augmentation functions
def add_noise(data, noise_factor=0.05):
    noise = torch.randn_like(data) * noise_factor
    return data + noise

def random_mask(data, mask_prob=0.15):
    mask = torch.rand_like(data) > mask_prob
    return data * mask

def shuffle_segments(data, n_segments=4):
    seq_len = data.size(0)
    segment_size = seq_len // n_segments
    if segment_size == 0:
        return data
    segments = [data[i:i+segment_size] for i in range(0, seq_len, segment_size)]
    np.random.shuffle(segments)
    return torch.cat(segments, dim=0)

def mixup(data1, data2, alpha=0.1):
    lam = np.random.beta(alpha, alpha)
    min_len = min(data1.size(0), data2.size(0))
    data1_trunc = data1[:min_len]
    data2_trunc = data2[:min_len]
    return lam * data1_trunc + (1 - lam) * data2_trunc

ATOM_TO_NUM = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86
}

class DynamicEdgeConv(nn.Module):
    """Dynamic edge feature learning"""
    def __init__(self, feat_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 1)
        )
    
    def forward(self, x, edge_index):
        row, col = edge_index
        edge_feat = torch.cat([x[row], x[col]], dim=1)
        return torch.sigmoid(self.edge_mlp(edge_feat))

class ProteinLigandDataset(Dataset):
    def __init__(self, pkl_path, normalize_affinity=False, training=False, 
                 noise_level=0.05, use_augmentation=False, aug_prob=0.3,
                 clean_and_balance=True, z_threshold=3.0, n_bins=5, smote_multiplier=1.0,
                 use_dynamic_edges=False, is_classification=False, mean=None, std=None):
        logging.info(f"Initializing dataset with pkl_path: {pkl_path}")
        with open(pkl_path, 'rb') as f:
            self.data_list = pickle.load(f)
        
        self.normalize_affinity = normalize_affinity
        self.training = training
        self.noise_level = noise_level
        self.use_augmentation = use_augmentation
        self.aug_prob = aug_prob
        self.clean_and_balance = clean_and_balance
        self.z_threshold = z_threshold
        self.n_bins = n_bins
        self.smote_multiplier = smote_multiplier
        self.use_dynamic_edges = use_dynamic_edges
        self.is_classification = is_classification
        self.mean = mean
        self.std = std

        # Initialize dynamic edge convolutions if enabled and training
        if self.use_dynamic_edges and self.training:
            self.prot_edge_conv = DynamicEdgeConv(feat_dim=9)
            self.mol_edge_conv = DynamicEdgeConv(feat_dim=43)
            self.clique_edge_conv = DynamicEdgeConv(feat_dim=1)

        # Step 1: Remove invalid values
        self._remove_invalid_values()

        # Step 2: Clean and balance data if specified and training
        if self.training and self.clean_and_balance:
            self._clean_and_balance_data()

        # Step 3: Normalize affinity if specified and training
        if self.training and self.normalize_affinity and not self.is_classification:
            self._normalize_affinity()

        if len(self.data_list) == 0:
            logging.warning(f"Dataset is empty after processing: {pkl_path}")
        else:
            logging.info(f"Dataset initialized with {len(self.data_list)} samples")

    def _remove_invalid_values(self):
        """Remove entries with invalid values, adjusting for training vs prediction."""
        original_len = len(self.data_list)
        if self.training:
            if self.is_classification:
                self.data_list = [d for d in self.data_list 
                                 if hasattr(d, 'classification_label') and 
                                 np.isfinite(d.classification_label)]
            else:
                self.data_list = [d for d in self.data_list 
                                 if hasattr(d, 'regression_label') and 
                                 (torch.is_tensor(d.regression_label) and torch.isfinite(d.regression_label).all() or 
                                  not torch.is_tensor(d.regression_label) and np.isfinite(d.regression_label))]
        else:
            self.data_list = [d for d in self.data_list 
                             if hasattr(d, 'chemBERTa') and torch.isfinite(d.chemBERTa).all() and 
                             hasattr(d, 'prot_node_evo') and torch.isfinite(d.prot_node_evo).all()]

        removed_count = original_len - len(self.data_list)
        if removed_count > 0:
            logging.info(f"Removed {removed_count} entries with invalid values. New size: {len(self.data_list)}")

    def _clean_and_balance_data(self):
        """Clean outliers and balance data using SMOTE."""
        if self.is_classification:
            labels = np.array([d.classification_label for d in self.data_list])
            if len(np.unique(labels)) < 2:
                logging.warning("Only one class present, skipping balancing")
                return
            features = np.zeros((len(self.data_list), 4))
            for i, d in enumerate(self.data_list):
                features[i, 0] = d.prot_node_aa.size(0) if hasattr(d, 'prot_node_aa') and d.prot_node_aa.numel() > 0 else 0.0
                features[i, 1] = d.mol_x_feat.size(0) if hasattr(d, 'mol_x_feat') and d.mol_x_feat.numel() > 0 else 0.0
                features[i, 2] = d.prot_node_evo.mean().item() if hasattr(d, 'prot_node_evo') and d.prot_node_evo.numel() > 0 else 0.0
                features[i, 3] = d.chemBERTa.mean().item() if hasattr(d, 'chemBERTa') and d.chemBERTa.numel() > 0 else 0.0
            
            smote = SMOTE(random_state=42, k_neighbors=min(5, len(features)-1))
            try:
                features_resampled, labels_resampled = smote.fit_resample(features, labels)
                original_data_dict = {i: d.clone() for i, d in enumerate(self.data_list)}
                new_data_list = []
                for i in range(len(features_resampled)):
                    if i < len(self.data_list):
                        new_data_list.append(original_data_dict[i])
                    else:
                        base_idx = np.random.choice(np.where(labels == labels_resampled[i])[0])
                        base_data = original_data_dict[base_idx].clone()
                        base_data.classification_label = labels_resampled[i]
                        new_data_list.append(base_data)
                self.data_list = new_data_list
                logging.info(f"Balanced classification data using SMOTE. New size: {len(self.data_list)}")
                logging.info(f"Class distribution after SMOTE: {Counter(labels_resampled)}")
            except ValueError as e:
                logging.warning(f"SMOTE failed for classification: {str(e)}. Skipping balancing.")
        else:
            affinities = np.array([d.regression_label.item() if torch.is_tensor(d.regression_label) else d.regression_label 
                                  for d in self.data_list])
            if len(affinities) == 0:
                logging.warning("No data to process for cleaning and balancing")
                return

            # Step 1: Remove outliers using Z-scores
            mean_aff = np.mean(affinities)
            std_aff = np.std(affinities) if np.std(affinities) > 0 else 1.0
            z_scores = np.abs((affinities - mean_aff) / std_aff)
            mask = z_scores <= self.z_threshold
            
            original_len = len(self.data_list)
            self.data_list = [d for d, m in zip(self.data_list, mask) if m]
            affinities_cleaned = affinities[mask]
            
            removed_count = original_len - len(self.data_list)
            logging.info(f"Removed {removed_count} outliers (Z-score > {self.z_threshold}). "
                        f"New size: {len(self.data_list)}")

            if len(self.data_list) == 0:
                logging.warning("All data removed after Z-score filtering")
                return

            # Step 2: Balance data using SMOTE
            if len(self.data_list) > 1:
                bins = np.linspace(min(affinities_cleaned), max(affinities_cleaned), self.n_bins + 1)
                labels = np.digitize(affinities_cleaned, bins[:-1]) - 1
                
                features = np.zeros((len(self.data_list), 4))
                for i, d in enumerate(self.data_list):
                    features[i, 0] = d.prot_node_aa.size(0) if hasattr(d, 'prot_node_aa') and d.prot_node_aa.numel() > 0 else 0.0
                    features[i, 1] = d.mol_x_feat.size(0) if hasattr(d, 'mol_x_feat') and d.mol_x_feat.numel() > 0 else 0.0
                    features[i, 2] = d.prot_node_evo.mean().item() if hasattr(d, 'prot_node_evo') else 0.0
                    features[i, 3] = d.chemBERTa.mean().item() if hasattr(d, 'chemBERTa') else 0.0
                
                label_counts = Counter(labels)
                logging.info(f"Original bin distribution: {label_counts}")
                majority_class_count = max(label_counts.values())
                
                sampling_strategy = {}
                for label, count in label_counts.items():
                    if count < majority_class_count:
                        target_count = max(count, int(majority_class_count * self.smote_multiplier * 0.1))
                        sampling_strategy[label] = target_count
                    else:
                        sampling_strategy[label] = count
                
                smote = SMOTE(
                    sampling_strategy=sampling_strategy,
                    k_neighbors=min(3, len(features)-1),
                    random_state=42
                )
                try:
                    features_resampled, labels_resampled = smote.fit_resample(features, labels)
                    
                    original_data_dict = {i: d.clone() for i, d in enumerate(self.data_list)}
                    new_data_list = []
                    
                    for i in range(len(features_resampled)):
                        if i < len(self.data_list):
                            new_data_list.append(original_data_dict[i])
                        else:
                            label = labels_resampled[i]
                            possible_indices = [idx for idx, l in enumerate(labels) if l == label]
                            if not possible_indices:
                                continue
                            base_idx = np.random.choice(possible_indices)
                            base_data = original_data_dict[base_idx].clone()
                            
                            prot_scale = features_resampled[i, 0] / features[base_idx, 0] if features[base_idx, 0] != 0 else 1.0
                            mol_scale = features_resampled[i, 1] / features[base_idx, 1] if features[base_idx, 1] != 0 else 1.0
                            evo_scale = features_resampled[i, 2] / features[base_idx, 2] if features[base_idx, 2] != 0 else 1.0
                            chem_scale = features_resampled[i, 3] / features[base_idx, 3] if features[base_idx, 3] != 0 else 1.0

                            base_data.prot_node_aa = base_data.prot_node_aa * prot_scale
                            base_data.mol_x_feat = base_data.mol_x_feat * mol_scale
                            base_data.prot_node_evo = base_data.prot_node_evo * evo_scale
                            base_data.chemBERTa = base_data.chemBERTa * chem_scale
                            base_data.regression_label = torch.tensor(bins[label] + (bins[label + 1] - bins[label]) * np.random.rand(), 
                                                                      dtype=torch.float32)
                            
                            new_data_list.append(base_data)
                    
                    self.data_list = new_data_list
                    logging.info(f"Balanced regression data using SMOTE. New size: {len(self.data_list)}")
                    
                    new_affinities = [d.regression_label.item() for d in self.data_list]
                    new_labels = np.digitize(new_affinities, bins[:-1]) - 1
                    unique, counts = np.unique(new_labels, return_counts=True)
                    logging.info(f"Bin distribution after SMOTE balancing: {dict(zip(unique, counts))}")
                except ValueError as e:
                    logging.warning(f"SMOTE failed for regression: {str(e)}. Skipping balancing.")

    def _normalize_affinity(self):
        """Normalize regression_label values using provided or computed mean and std."""
        affinities = [d.regression_label.item() if torch.is_tensor(d.regression_label) else d.regression_label 
                     for d in self.data_list]
        if not affinities:
            self.mean = 0.0 if self.mean is None else self.mean
            self.std = 1.0 if self.std is None else self.std
            return
        
        # Use provided mean and std if available, else compute
        if self.mean is None or self.std is None:
            self.mean = np.mean(affinities)
            self.std = np.std(affinities) if np.std(affinities) > 0 else 1.0
        
        for data in self.data_list:
            aff = data.regression_label.clone().detach() if torch.is_tensor(data.regression_label) else torch.tensor(data.regression_label)
            data.regression_label = (aff - self.mean) / self.std

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx].clone()
        atom_types = data.atom_types.split('|')
        smiles_atomic = torch.tensor([ATOM_TO_NUM.get(atom, 0) for atom in atom_types], 
                                   dtype=torch.float32)
        data.smiles_atomic = F.pad(smiles_atomic, (0, 128 - len(atom_types)), value=0)
        
        if not hasattr(data, 'chemBERTa'):
            raise ValueError("chemBERTa feature missing in dataset")
        data.chemBERTa = data.chemBERTa.float()
        
        if not hasattr(data, 'prot_node_evo'):
            raise ValueError("prot_node_evo feature missing in dataset")
        data.prot_node_evo = data.prot_node_evo.float()

        # Ensure label type consistency (only for training)
        if self.training:
            if self.is_classification and hasattr(data, 'classification_label'):
                data.classification_label = torch.tensor(data.classification_label, dtype=torch.long)
            elif not self.is_classification and hasattr(data, 'regression_label'):
                if not torch.is_tensor(data.regression_label):
                    data.regression_label = torch.tensor(data.regression_label, dtype=torch.float32)

        # Apply augmentation only during training
        if self.training and self.use_augmentation and torch.rand(1).item() < self.aug_prob:
            aug_choice = np.random.choice(4)
            if aug_choice == 0:
                data.chemBERTa = add_noise(data.chemBERTa, self.noise_level)
                data.prot_node_evo = add_noise(data.prot_node_evo, self.noise_level)
            elif aug_choice == 1:
                data.chemBERTa = random_mask(data.chemBERTa)
                data.prot_node_evo = random_mask(data.prot_node_evo)
            elif aug_choice == 2 and data.prot_x_protein.size(0) >= 4:
                data.prot_x_protein = shuffle_segments(data.prot_x_protein)
            elif aug_choice == 3 and idx > 0:
                other_idx = np.random.randint(0, len(self.data_list))
                other_data = self.data_list[other_idx]
                data.prot_x_protein = mixup(data.prot_x_protein, other_data.prot_x_protein)

        if self.training:
            if torch.rand(1).item() < 0.3:
                chemberta_noise = torch.randn_like(data.chemBERTa) * self.noise_level
                data.chemBERTa += chemberta_noise
                prot_node_evo_noise = torch.randn_like(data.prot_node_evo) * self.noise_level
                data.prot_node_evo += prot_node_evo_noise
                if not self.is_classification and hasattr(data, 'regression_label'):
                    data.regression_label += torch.randn_like(data.regression_label) * self.noise_level

        # Apply dynamic edge convolution only during training
        if self.use_dynamic_edges and self.training:
            if hasattr(data, 'prot_edge_index') and data.prot_edge_index.size(1) > 0:
                data.prot_edge_weight = self.prot_edge_conv(data.prot_node_aa, data.prot_edge_index)
            if hasattr(data, 'mol_edge_index') and data.mol_edge_index.size(1) > 0:
                data.mol_edge_attr = self.mol_edge_conv(data.mol_x_feat, data.mol_edge_index)
            if hasattr(data, 'clique_edge_index') and data.clique_edge_index.size(1) > 0:
                data.clique_edge_weight = self.clique_edge_conv(data.clique_x, data.clique_edge_index)

        # Type casting for all features
        data.prot_node_aa = data.prot_node_aa.float()
        data.prot_node_evo = data.prot_node_evo.float()
        data.prot_node_pos = data.prot_node_pos.long()
        data.prot_one_hot = data.prot_one_hot.float()
        data.mol_x_feat = data.mol_x_feat.float()
        data.clique_x = data.clique_x.float()
        if hasattr(data, 'prot_edge_weight'):
            data.prot_edge_weight = data.prot_edge_weight.float()
        if hasattr(data, 'mol_edge_attr'):
            data.mol_edge_attr = data.mol_edge_attr.float()
        if hasattr(data, 'clique_edge_weight'):
            data.clique_edge_weight = data.clique_edge_weight.float()
        return data

def custom_collate_fn(data_list):
    batch = Batch.from_data_list(data_list)
    batch.prot_batch = torch.cat([torch.full((d.prot_node_aa.size(0),), i, dtype=torch.long) for i, d in enumerate(data_list)])
    batch.mol_batch = torch.cat([torch.full((d.mol_x_feat.size(0),), i, dtype=torch.long) for i, d in enumerate(data_list)])
    batch.clique_batch = torch.cat([torch.full((d.clique_x.size(0),), i, dtype=torch.long) for i, d in enumerate(data_list)])
    batch.prot_x_protein = pad_sequence([d.prot_x_protein for d in data_list], batch_first=True)
    batch.smiles_atomic = torch.stack([d.smiles_atomic for d in data_list])
    batch.chemBERTa = torch.stack([d.chemBERTa for d in data_list])
    
    batch_size = len(data_list)
    max_nodes = max(d.mol_x_feat.size(0) for d in data_list)
    mol_x_feat_seq = torch.zeros(batch_size, max_nodes, 43, dtype=torch.float32)
    for i, d in enumerate(data_list):
        num_nodes = d.mol_x_feat.size(0)
        mol_x_feat_seq[i, :num_nodes, :] = d.mol_x_feat
    batch.mol_x_feat_seq = mol_x_feat_seq
    
    # Stack labels if present
    if hasattr(data_list[0], 'classification_label'):
        batch.classification_label = torch.stack([d.classification_label for d in data_list])
    elif hasattr(data_list[0], 'regression_label'):
        batch.regression_label = torch.stack([d.regression_label for d in data_list])
    return batch