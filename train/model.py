import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINConv, global_max_pool, GraphNorm
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from torch_geometric.utils import k_hop_subgraph, remove_self_loops
from torch.nn.utils.rnn import pad_sequence

# Gradient reversal layer
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)

# Graph Attention Pooling with Multi-Scale Pooling
class GraphAttentionPooling(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, batch):
        x = x.to(dtype=torch.float32)
        attn_scores = self.attn(x)
        attn_weights = F.softmax(attn_scores.view(-1), dim=0).view_as(attn_scores)
        weighted_x = x * attn_weights
        num_graphs = batch.max().item() + 1
        
        attn_pooled = torch.zeros(num_graphs, x.size(-1), device=x.device, dtype=x.dtype)
        attn_pooled = attn_pooled.scatter_add_(0, batch.unsqueeze(-1).expand_as(weighted_x), weighted_x)
        
        max_pooled = global_max_pool(x, batch)
        
        mean_pooled = torch.zeros(num_graphs, x.size(-1), device=x.device, dtype=x.dtype)
        mean_pooled = mean_pooled.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
        counts = torch.bincount(batch, minlength=num_graphs).float().clamp(min=1).unsqueeze(-1).to(dtype=x.dtype)
        mean_pooled = mean_pooled / counts
        
        return torch.cat([attn_pooled, max_pooled, mean_pooled], dim=-1)

# Sparse GATv2Conv with Dropout on Attention Weights
class SparseGATv2Conv(GATv2Conv):
    def __init__(self, *args, dropout=0.3, **kwargs):
        super().__init__(*args, dropout=dropout, **kwargs)
        self.dropout_prob = dropout
        self.attn_dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=None):
        if return_attention_weights:
            out, (edge_idx, alpha) = super().forward(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            alpha = self.attn_dropout(alpha)
            return out, (edge_idx, alpha)
        else:
            out = super().forward(x, edge_index, edge_attr=edge_attr, return_attention_weights=False)
            return out

# Protein Graph Encoder
class ProteinGraphEncoder(nn.Module):
    def __init__(self, evo_dim=1280, pos_dim=32, one_hot_dim=21, aa_dim=9, hidden_dim=128, gat_heads=4, transformer_nhead=4, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pos_embed = nn.Embedding(12400, pos_dim)
        
        self.evo_fc = nn.Sequential(
            nn.Linear(evo_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        self.evo_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=transformer_nhead, dim_feedforward=hidden_dim * 4, dropout=dropout, activation='gelu', batch_first=True),
            num_layers=2
        )
        self.evo_norm = nn.LayerNorm(hidden_dim)
        self.evo_bn = nn.BatchNorm1d(hidden_dim)
        self.one_hot_fc = nn.Sequential(
            nn.Linear(one_hot_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.aa_fc = nn.Sequential(
            nn.Linear(aa_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        input_dim = pos_dim + hidden_dim * 3
        self.edge_encoder_prot = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        self.gat1 = SparseGATv2Conv(input_dim, hidden_dim, heads=4, edge_dim=hidden_dim, dropout=dropout)
        self.gat_norm1 = GraphNorm(hidden_dim * 4)
        self.gat2 = SparseGATv2Conv(hidden_dim * 4, hidden_dim, heads=4, edge_dim=hidden_dim, dropout=dropout)
        self.gat_norm2 = GraphNorm(hidden_dim * 4)
        self.residual_proj = nn.Linear(input_dim, hidden_dim * 4)
        self.pool = GraphAttentionPooling(hidden_dim * 4)
        self.pool_proj = nn.Linear(hidden_dim * 4 * 3, hidden_dim)
        self.raw_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.gat1_attn_weights = None
        self.gat2_attn_weights = None
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)

    def _apply_seq(self, sequential, x):
        for layer in sequential:
            if isinstance(layer, nn.BatchNorm1d) and x.size(0) == 1:
                continue
            x = layer(x)
        return x

    def forward(self, data):
        pos_embed = self.pos_embed(data.prot_node_pos.squeeze(-1))
        evo_input = self.evo_fc(data.prot_node_evo)
        evo_feat = self.evo_transformer(evo_input.unsqueeze(0)).squeeze(0)
        evo_feat = self.evo_norm(evo_feat)
        if evo_feat.size(0) > 1:
            evo_feat = self.evo_bn(evo_feat)
        one_hot_feat = self._apply_seq(self.one_hot_fc, data.prot_one_hot)
        aa_feat = self._apply_seq(self.aa_fc, data.prot_node_aa)
        prot_feat = torch.cat([pos_embed, evo_feat, one_hot_feat, aa_feat], dim=-1)
        edge_attr = self.edge_encoder_prot(data.prot_edge_weight.unsqueeze(-1))
        gat1_out, (edge_idx1, attn1) = self.gat1(prot_feat, data.prot_edge_index, edge_attr=edge_attr, return_attention_weights=True)
        self.gat1_attn_weights = attn1
        gat1_out = F.elu(self.gat_norm1(gat1_out))
        residual = self.residual_proj(prot_feat)
        x, (edge_idx2, attn2) = self.gat2(gat1_out, data.prot_edge_index, edge_attr=edge_attr, return_attention_weights=True)
        self.gat2_attn_weights = attn2
        x = self.gat_norm2(x)
        x = x + residual
        x = F.elu(x)
        pooled = self.pool(x, data.prot_batch)
        pooled = self.pool_proj(pooled)
        raw_proj = self.raw_proj(x)
        return pooled, raw_proj

# Ligand Graph Encoder
class LigandGraphEncoder(nn.Module):
    def __init__(self, atom_feat_dim=43, clique_feat_dim=1, mol_edge_attr_dim=1, hidden_dim=128, chemberta_dim=768, gat_heads=1, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gat_heads = gat_heads
        self.atom_fc = nn.Sequential(
            nn.Linear(atom_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        self.edge_encoder_mol = nn.Sequential(
            nn.Linear(mol_edge_attr_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        self.gat_mol = GATv2Conv(hidden_dim, hidden_dim, heads=gat_heads, edge_dim=hidden_dim, dropout=dropout)
        self.gat_mol_norm = GraphNorm(hidden_dim * gat_heads)
        self.mol_residual = nn.Linear(hidden_dim, hidden_dim * gat_heads)
        self.clique_fc = nn.Sequential(
            nn.Linear(clique_feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        self.clique_interaction = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        )
        self.gat_clique = GATv2Conv(hidden_dim, hidden_dim, heads=gat_heads, edge_dim=1, dropout=dropout)
        self.gat_clique_norm = GraphNorm(hidden_dim * gat_heads)
        self.clique_residual = nn.Linear(hidden_dim, hidden_dim * gat_heads)
        
        self.chemberta_fc = nn.Sequential(
            nn.Linear(chemberta_dim, hidden_dim * gat_heads * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * gat_heads * 2, hidden_dim * gat_heads),
            nn.BatchNorm1d(hidden_dim * gat_heads),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        self.chemberta_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim * gat_heads, nhead=4, dim_feedforward=hidden_dim * gat_heads * 4, dropout=dropout, activation='gelu', batch_first=True),
            num_layers=1
        )
        self.chemberta_norm = nn.LayerNorm(hidden_dim * gat_heads)
        self.chemberta_proj = nn.Linear(hidden_dim * gat_heads, hidden_dim)
        self.mol_pool = GraphAttentionPooling(hidden_dim * gat_heads)
        self.mol_pool_proj = nn.Linear(hidden_dim * gat_heads * 3, hidden_dim)
        self.clique_pool = GraphAttentionPooling(hidden_dim * gat_heads)
        self.clique_pool_proj = nn.Linear(hidden_dim * gat_heads * 3, hidden_dim)
        self.fusion_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout)
        self.fusion_weights = nn.Linear(hidden_dim * 3, hidden_dim)
        self.fusion_bn = nn.BatchNorm1d(hidden_dim)
        self.raw_proj = nn.Linear(hidden_dim * gat_heads, hidden_dim)
        self.gat_mol_attn_weights = None
        self.gat_clique_attn_weights = None
        self.fusion_attn_weights = None
        self.chemberta_attn_weights = []
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)

    def _apply_seq(self, sequential, x):
        for layer in sequential:
            if isinstance(layer, nn.BatchNorm1d) and x.size(0) == 1:
                continue
            x = layer(x)
        return x

    def forward(self, data):
        atom_feat = self._apply_seq(self.atom_fc, data.mol_x_feat)
        edge_attr = self.edge_encoder_mol(data.mol_edge_attr.unsqueeze(-1))
        mol_residual = self.mol_residual(atom_feat)
        mol_gat_out, (mol_idx, mol_attn) = self.gat_mol(atom_feat, data.mol_edge_index, edge_attr=edge_attr, return_attention_weights=True)
        self.gat_mol_attn_weights = mol_attn
        mol_gat_out = self.gat_mol_norm(mol_gat_out)
        mol_feat = F.elu(mol_gat_out + mol_residual)
        mol_pooled = self.mol_pool(mol_feat, data.mol_batch)
        mol_pooled = self.mol_pool_proj(mol_pooled)
        
        clique_feat = self._apply_seq(self.clique_fc, data.clique_x)
        if hasattr(data, 'cross_edge_index') and data.cross_edge_index is not None:
            clique_feat = self.clique_interaction(clique_feat, data.cross_edge_index)
        clique_residual = self.clique_residual(clique_feat)
        clique_gat_out, (clique_idx, clique_attn) = self.gat_clique(clique_feat, data.clique_edge_index, edge_attr=data.clique_edge_weight.unsqueeze(-1), return_attention_weights=True)
        self.gat_clique_attn_weights = clique_attn
        clique_gat_out = self.gat_clique_norm(clique_gat_out)
        clique_feat = F.elu(clique_gat_out + clique_residual)
        clique_pooled = self.clique_pool(clique_feat, data.clique_batch)
        clique_pooled = self.clique_pool_proj(clique_pooled)
        
        chemberta_input = data.chemBERTa.squeeze(1) if data.chemBERTa.dim() == 3 else data.chemBERTa
        chemberta_feat = self.chemberta_fc(chemberta_input)
        chemberta_input = chemberta_feat.unsqueeze(0)
        self.chemberta_attn_weights = []
        chemberta_out = chemberta_input
        for layer in self.chemberta_transformer.layers:
            out, attn_weights = layer.self_attn(chemberta_out, chemberta_out, chemberta_out)
            self.chemberta_attn_weights.append(attn_weights)
            chemberta_out = layer(chemberta_out)
        chemberta_feat = chemberta_out.squeeze(0)
        chemberta_feat = self.chemberta_norm(chemberta_feat)
        chemberta_feat = self.chemberta_proj(chemberta_feat)
        
        mol_pooled = mol_pooled.unsqueeze(0)
        clique_pooled = clique_pooled.unsqueeze(0)
        chemberta_pooled = chemberta_feat.unsqueeze(0)
        attn_input = torch.cat([mol_pooled, clique_pooled, chemberta_pooled], dim=0)
        fused_feat, fusion_attn = self.fusion_attn(attn_input, attn_input, attn_input)
        self.fusion_attn_weights = fusion_attn
        fused_feat = fused_feat + attn_input.mean(dim=0, keepdim=True)
        batch_size = fused_feat.size(1)
        fused_flat = fused_feat.permute(1, 0, 2).reshape(batch_size, -1)
        weights = self.fusion_weights(fused_flat)
        if weights.size(0) > 1:
            weights = self.fusion_bn(weights)
        raw_proj = self.raw_proj(mol_feat)
        return weights, raw_proj

# Protein Sequence Encoder
class ProteinSequenceEncoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.residual_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, dropout=dropout)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.attn_weights = None
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, protein):
        out1, _ = self.lstm1(protein)
        residual = self.residual_proj(out1)
        out2, _ = self.lstm2(out1)
        out = out2 + residual
        out = out.permute(1, 0, 2)
        attn_out, attn_weights = self.attention(out, out, out)
        self.attn_weights = attn_weights
        attn_out = attn_out.permute(1, 2, 0)
        return self.pool(attn_out).squeeze(-1)

# Ligand Sequence Encoder
class LigandSequenceEncoder(nn.Module):
    def __init__(self, smiles_dim=128, mol_dim=43, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.smiles_fc = nn.Sequential(
            nn.Linear(smiles_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mol_lstm = nn.LSTM(mol_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.residual_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, smiles_atomic, mol_x_feat):
        smiles_out = self.smiles_fc(smiles_atomic)
        smiles_residual = self.residual_proj(smiles_out)
        if smiles_out.dim() == 3:
            smiles_out = self.pool(smiles_out.permute(0, 2, 1)).squeeze(-1)
        mol_out, _ = self.mol_lstm(mol_x_feat)
        mol_out = self.pool(mol_out.permute(0, 2, 1)).squeeze(-1)
        combined = torch.cat([smiles_out, mol_out], dim=-1)
        combined = combined + smiles_residual
        output = self.output_proj(combined)
        return output

# Cross Transformer Fusion
class CrossTransformerFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, num_layers=1, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.cross_layers = nn.ModuleList([
            nn.ModuleDict({
                'g2s_attn': nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout),
                'g_norm1': nn.LayerNorm(hidden_dim),
                'g_ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout)
                ),
                'g_norm2': nn.LayerNorm(hidden_dim),
                's2g_attn': nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout),
                's_norm1': nn.LayerNorm(hidden_dim),
                's_ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout)
                ),
                's_norm2': nn.LayerNorm(hidden_dim)
            }) for _ in range(num_layers)
        ])
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.g2s_attn_weights = []
        self.s2g_attn_weights = []
    
    def forward(self, graph_feat, seq_feat):
        g_x = graph_feat
        s_x = seq_feat
        self.g2s_attn_weights = []
        self.s2g_attn_weights = []
        
        for layer in self.cross_layers:
            g_x_res = g_x
            g_attn, g2s_attn = layer['g2s_attn'](g_x, s_x, s_x)
            self.g2s_attn_weights.append(g2s_attn)
            g_x = layer['g_norm1'](g_x + g_attn)
            g_ffn = layer['g_ffn'](g_x)
            g_x = layer['g_norm2'](g_x + g_ffn + g_x_res)
            
            s_x_res = s_x
            s_attn, s2g_attn = layer['s2g_attn'](s_x, g_x, g_x)
            self.s2g_attn_weights.append(s2g_attn)
            s_x = layer['s_norm1'](s_x + s_attn)
            s_ffn = layer['s_ffn'](s_x)
            s_x = layer['s_norm2'](s_x + s_ffn + s_x_res)
        
        combined = torch.cat([g_x, s_x], dim=-1)
        gate = self.fusion_gate(combined)
        fused = gate * g_x + (1 - gate) * s_x
        
        output = self.output_proj(fused)
        return output

# Balanced Cross Transformer with Gating Mechanism
class BalancedCrossTransformer(CrossTransformerFusion):
    def __init__(self, hidden_dim, num_heads=4, num_layers=1, dropout=0.1):
        super().__init__(hidden_dim, num_heads, num_layers, dropout)
        self.chem_gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, graph_feat, seq_feat):
        gate_weight = self.chem_gate(seq_feat)
        seq_feat = gate_weight * seq_feat
        return super().forward(graph_feat, seq_feat)

# Fused Affinity Predictor with GRL
class FusedAffinityPredictor(nn.Module):
    def __init__(self, atom_feat_dim=43, mol_edge_attr_dim=1, hidden_dim=128, chemberta_dim=768,
                 gat_heads_protein=4, gat_heads_ligand=1, transformer_nhead=4, cross_heads=4,
                 input_dropout=0.2, hidden_dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dropout = input_dropout
        self.hidden_dropout = hidden_dropout
        
        self.prot_graph_enc = ProteinGraphEncoder(
            hidden_dim=hidden_dim, 
            gat_heads=gat_heads_protein, 
            transformer_nhead=transformer_nhead,
            dropout=input_dropout
        )
        self.lig_graph_enc = LigandGraphEncoder(
            atom_feat_dim=atom_feat_dim, 
            mol_edge_attr_dim=mol_edge_attr_dim, 
            hidden_dim=hidden_dim, 
            chemberta_dim=chemberta_dim, 
            gat_heads=gat_heads_ligand,
            dropout=input_dropout
        )
        self.prot_seq_enc = ProteinSequenceEncoder(
            input_dim=6,
            hidden_dim=hidden_dim,
            dropout=input_dropout
        )
        self.lig_seq_enc = LigandSequenceEncoder(
            smiles_dim=128,
            mol_dim=43,
            hidden_dim=hidden_dim,
            dropout=input_dropout
        )
        self.cross_modal_fusion = BalancedCrossTransformer(
            hidden_dim=hidden_dim, 
            num_heads=cross_heads,
            dropout=input_dropout
        )
        
        self.early_fusion = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=input_dropout)
        
        self.raw_pool = GraphAttentionPooling(hidden_dim)
        self.raw_pool_proj = nn.Linear(hidden_dim * 3, hidden_dim)
        
        self.feature_proj = nn.Linear(11, hidden_dim * 5)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.LayerNorm(hidden_dim * 2),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_dim * 2, nhead=4, dim_feedforward=hidden_dim * 8, dropout=hidden_dropout, activation='gelu', batch_first=True),
                num_layers=1
            )
        )
        self.residual_fc = nn.Linear(hidden_dim * 5, hidden_dim * 2)
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Domain Classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  
        )
        
        self.early_fusion_attn_weights = None
        self._initialize_weights()
        
        # GRL’s alpha parameter
        self.alpha = 1.0

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, batch=None, intermediate_features=None, domain_label=None):
        if intermediate_features is not None:
            combined = self.feature_proj(intermediate_features)
            fused_feat = self.fusion_layer(combined)
            residual = self.residual_fc(combined)
            fused_feat = fused_feat + residual
            regression_output = self.regressor(fused_feat)
            classification_output = self.classifier(fused_feat)
            
            # Domain Classifier
            if domain_label is not None:
                reversed_feat = grad_reverse(fused_feat, self.alpha)
                domain_output = self.domain_classifier(reversed_feat)
                return regression_output, classification_output, domain_output
            return regression_output, classification_output
        
        else:
            prot_graph_feat, prot_graph_raw = self.prot_graph_enc(batch)
            lig_graph_feat, lig_graph_raw = self.lig_graph_enc(batch)
            prot_seq_feat = self.prot_seq_enc(batch.prot_x_protein)
            lig_seq_feat = self.lig_seq_enc(batch.smiles_atomic, batch.mol_x_feat_seq)
            
            prot_graph_raw_pooled = self.raw_pool(prot_graph_raw.to(dtype=torch.float32), batch.prot_batch)
            prot_graph_raw_pooled = self.raw_pool_proj(prot_graph_raw_pooled)
            lig_graph_raw_pooled = self.raw_pool(lig_graph_raw.to(dtype=torch.float32), batch.mol_batch)
            lig_graph_raw_pooled = self.raw_pool_proj(lig_graph_raw_pooled)

            prot_graph_raw_pooled = prot_graph_raw_pooled.unsqueeze(0)
            lig_graph_raw_pooled = lig_graph_raw_pooled.unsqueeze(0)
            
            early_fused, early_attn = self.early_fusion(prot_graph_raw_pooled, lig_graph_raw_pooled, lig_graph_raw_pooled)
            self.early_fusion_attn_weights = early_attn
            
            prot_graph_feat = prot_graph_feat.unsqueeze(0)
            lig_graph_feat = lig_graph_feat.unsqueeze(0)
            prot_seq_feat = prot_seq_feat.unsqueeze(0)
            lig_seq_feat = lig_seq_feat.unsqueeze(0)
            
            prot_fused = self.cross_modal_fusion(prot_graph_feat, prot_seq_feat)
            lig_fused = self.cross_modal_fusion(lig_graph_feat, lig_seq_feat)
            
            combined = torch.cat([prot_fused, lig_fused, early_fused, prot_graph_feat, lig_graph_feat], dim=-1)
            fused_feat = self.fusion_layer(combined.squeeze(0))
            residual = self.residual_fc(combined.squeeze(0))
            fused_feat = fused_feat + residual
            
            regression_output = self.regressor(fused_feat)
            classification_output = self.classifier(fused_feat)
            
            # Domain Classifier
            if domain_label is not None:
                reversed_feat = grad_reverse(fused_feat, self.alpha)
                domain_output = self.domain_classifier(reversed_feat)
                return regression_output, classification_output, domain_output
            
            return regression_output, classification_output

