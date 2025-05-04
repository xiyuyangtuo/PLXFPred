import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import os
import pickle
from lifelines.utils import concordance_index
import seaborn as sns
from dataset import ProteinLigandData
from model import FusedAffinityPredictor
from data_utils import ProteinLigandDataset, setup_logging, custom_collate_fn
import shap
import pandas as pd
from torch_geometric.nn import global_max_pool
from visualize_attention import (visualize_all_attentions, shap_summary_bar, shap_waterfall_plot, 
                                 shap_density_scatter, shap_hierarchical_clustering, 
                                 shap_dependence_plots, shap_force_plot, shap_decision_plot, shap_heatmap_plot)  

# Initialize logging
log_dir = "log"
log_path = os.path.join(log_dir, "train_log_multi_task.txt")
setup_logging(log_path)
logging.info("Starting multi-task training process")

# MultiTaskLossWrapper
class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num=2):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, regression_pred, classification_pred, 
                regression_target, classification_target, criterion_reg, criterion_cls):
        try:
            if regression_pred.size(0) != regression_target.size(0):
                loss_reg = torch.tensor(0.0, device=regression_pred.device)
            else:
                loss_reg = criterion_reg(regression_pred, regression_target)
            if classification_pred.size(0) != classification_target.size(0):
                loss_cls = torch.tensor(0.0, device=classification_pred.device)
            else:
                loss_cls = criterion_cls(classification_pred, classification_target)
            
            precision_reg = torch.exp(-self.log_vars[0])
            loss_reg_weighted = precision_reg * loss_reg + self.log_vars[0]
            precision_cls = torch.exp(-self.log_vars[1])
            loss_cls_weighted = precision_cls * loss_cls + self.log_vars[1]
            total_loss = loss_reg_weighted + loss_cls_weighted
            return total_loss, loss_reg, loss_cls
        except Exception as e:
            logging.error(f"Error in loss calculation: {e}")
            return torch.tensor(float('inf'), device=regression_pred.device), torch.tensor(0.0), torch.tensor(0.0)

# Weighted MSE Loss
class WeightedMSELoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=1.5, max_weight=1000.0):
        super(WeightedMSELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.max_weight = max_weight
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, inputs, targets):
        mse_loss = self.mse(inputs, targets)
        pt = torch.exp(-mse_loss)
        weights = self.alpha * (1 - pt) ** self.gamma
        weights = torch.clamp(weights, max=self.max_weight)
        return (weights * mse_loss).mean()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enhanced Evaluate Model Function
def evaluate_model(model, loader, device, is_classification=False):
    model.eval()
    y_true_reg, y_pred_reg = [], []
    y_true_cls, y_pred_cls, y_pred_cls_prob = [], [], []
    try:
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                regression_output, classification_output = model(batch)
                if hasattr(batch, 'regression_label'):
                    y_true_reg.extend(batch.regression_label.cpu().numpy())
                    y_pred_reg.extend(regression_output.cpu().numpy().flatten())
                if hasattr(batch, 'classification_label'):
                    y_true_cls.extend(batch.classification_label.cpu().numpy())
                    y_pred_cls_prob.extend(torch.sigmoid(classification_output).cpu().numpy().flatten())
                    y_pred_cls.extend(torch.sigmoid(classification_output).round().cpu().numpy().flatten())
        
        metrics = {}
        if y_true_reg and not is_classification:
            y_true_reg = np.array(y_true_reg)
            y_pred_reg = np.array(y_pred_reg)
            valid_mask = np.isfinite(y_true_reg) & np.isfinite(y_pred_reg)
            if valid_mask.sum() > 0:
                y_true_reg_clean = y_true_reg[valid_mask]
                y_pred_reg_clean = y_pred_reg[valid_mask]
                n_samples = len(y_true_reg_clean)
                n_features = 1  # Assuming single-output regression
                r2 = r2_score(y_true_reg_clean, y_pred_reg_clean)
                adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
                ci = concordance_index(y_true_reg_clean, y_pred_reg_clean)
                metrics.update({
                    'r2': r2,
                    'adjusted_r2': adjusted_r2,
                    'mse': mean_squared_error(y_true_reg_clean, y_pred_reg_clean),
                    'rmse': np.sqrt(mean_squared_error(y_true_reg_clean, y_pred_reg_clean)),
                    'mae': mean_absolute_error(y_true_reg_clean, y_pred_reg_clean),
                    'pcc': pearsonr(y_true_reg_clean, y_pred_reg_clean)[0] if not np.isnan(pearsonr(y_true_reg_clean, y_pred_reg_clean)[0]) else 0.0,
                    'spearman': spearmanr(y_true_reg_clean, y_pred_reg_clean)[0] if not np.isnan(spearmanr(y_true_reg_clean, y_pred_reg_clean)[0]) else 0.0,
                    'ci': ci,
                    'y_true_reg': y_true_reg_clean,
                    'y_pred_reg': y_pred_reg_clean,
                    'residuals': y_true_reg_clean - y_pred_reg_clean
                })
            else:
                metrics.update({
                    'r2': 0.0, 'adjusted_r2': 0.0, 'mse': 0.0, 'rmse': 0.0, 'mae': 0.0,
                    'pcc': 0.0, 'spearman': 0.0, 'ci': 0.0, 'y_true_reg': np.array([]), 'y_pred_reg': np.array([]),
                    'residuals': np.array([])
                })
        if y_true_cls and is_classification:
            y_true_cls = np.array(y_true_cls)
            y_pred_cls = np.array(y_pred_cls)
            y_pred_cls_prob = np.array(y_pred_cls_prob)
            valid_mask = np.isfinite(y_true_cls) & np.isfinite(y_pred_cls_prob)
            if valid_mask.sum() > 0:
                y_true_cls_clean = y_true_cls[valid_mask]
                y_pred_cls_clean = y_pred_cls[valid_mask]
                y_pred_cls_prob_clean = y_pred_cls_prob[valid_mask]
                precision, recall, _ = precision_recall_curve(y_true_cls_clean, y_pred_cls_prob_clean)
                metrics.update({
                    'accuracy': accuracy_score(y_true_cls_clean, y_pred_cls_clean),
                    'roc_auc': roc_auc_score(y_true_cls_clean, y_pred_cls_prob_clean),
                    'f1': f1_score(y_true_cls_clean, y_pred_cls_clean, average='weighted'),
                    'fpr': roc_curve(y_true_cls_clean, y_pred_cls_prob_clean)[0],
                    'tpr': roc_curve(y_true_cls_clean, y_pred_cls_prob_clean)[1],
                    'precision': precision,
                    'recall': recall,
                    'ap': average_precision_score(y_true_cls_clean, y_pred_cls_prob_clean),
                    'conf_matrix': confusion_matrix(y_true_cls_clean, y_pred_cls_clean)
                })
            else:
                metrics.update({
                    'accuracy': 0.0, 'roc_auc': 0.0, 'f1': 0.0, 'fpr': np.array([]), 'tpr': np.array([]),
                    'precision': np.array([]), 'recall': np.array([]), 'ap': 0.0, 'conf_matrix': np.array([[0, 0], [0, 0]])
                })
        return metrics
    except Exception as e:
        logging.error(f"Error in evaluate_model: {e}")
        return {
            'r2': 0.0, 'adjusted_r2': 0.0, 'mse': 0.0, 'rmse': 0.0, 'mae': 0.0, 'pcc': 0.0, 'spearman': 0.0, 'ci': 0.0,
            'accuracy': 0.0, 'roc_auc': 0.0, 'f1': 0.0, 'fpr': np.array([]), 'tpr': np.array([]),
            'precision': np.array([]), 'recall': np.array([]), 'ap': 0.0, 'conf_matrix': np.array([[0, 0], [0, 0]]),
            'y_true_reg': np.array([]), 'y_pred_reg': np.array([]), 'residuals': np.array([])
        }

# Visualization Functions
def plot_losses(train_losses_reg, train_losses_cls, val_losses_reg, val_losses_cls, log_dir):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_reg, label='Train Reg Loss', marker='o')
    plt.plot(val_losses_reg, label='Val Reg Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Regression Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(train_losses_cls, label='Train Cls Loss', marker='o')
    plt.plot(val_losses_cls, label='Val Cls Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Classification Loss')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(log_dir, 'loss_plot_multi_task.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Loss plot saved to {plot_path}")

def plot_classification_metrics(train_metrics_cls, val_metrics_cls, log_dir):
    epochs = range(1, len(train_metrics_cls['accuracy']) + 1)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_metrics_cls['accuracy'], label='Train Accuracy', marker='o')
    plt.plot(epochs, val_metrics_cls['accuracy'], label='Val Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_metrics_cls['roc_auc'], label='Train ROC-AUC', marker='o')
    plt.plot(epochs, val_metrics_cls['roc_auc'], label='Val ROC-AUC', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('ROC-AUC')
    plt.title('ROC-AUC over Epochs')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_metrics_cls['f1'], label='Train F1', marker='o')
    plt.plot(epochs, val_metrics_cls['f1'], label='Val F1', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score over Epochs')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(log_dir, 'classification_metrics_over_epochs.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Classification metrics plot saved to {plot_path}")

def plot_roc_curve(fpr, tpr, roc_auc, dataset_name, log_dir):
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Cold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {dataset_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plot_path = os.path.join(log_dir, f'roc_curve_{dataset_name.lower()}.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"ROC curve for {dataset_name} saved to {plot_path}")

def plot_precision_recall_curve(precision, recall, ap, dataset_name, log_dir):
    plt.figure()
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AP = {ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {dataset_name}')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(log_dir, f'pr_curve_{dataset_name.lower()}.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Precision-Recall curve for {dataset_name} saved to {plot_path}")

def plot_combined_affinity_error(metrics_dict, dataset_name, log_dir):
    y_true = metrics_dict['y_true_reg']
    y_pred = metrics_dict['y_pred_reg']
    residuals = metrics_dict['residuals']
    if len(y_true) > 0 and len(y_pred) > 0:
        pcc = pearsonr(y_true, y_pred)[0] if not np.isnan(pearsonr(y_true, y_pred)[0]) else 0.0
        r2 = r2_score(y_true, y_pred)
        abs_residuals = np.abs(residuals)
        vmin, vmax = 0, np.max(abs_residuals)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(y_true, y_pred, c=abs_residuals, cmap='RdBu_r', vmin=vmin, vmax=vmax, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Ideal')
        plt.colorbar(scatter, label='Absolute Residual')
        plt.xlabel('True Affinity')
        plt.ylabel('Predicted Affinity')
        plt.title(f'Predicted vs True Affinity - {dataset_name} (PCC = {pcc:.4f}, R2 = {r2:.4f})')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(log_dir, f'combined_affinity_error_{dataset_name.lower()}.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Combined affinity and error plot for {dataset_name} saved to {plot_path}")

def plot_residuals(metrics_dict, dataset_name, log_dir):
    residuals = metrics_dict['residuals']
    if len(residuals) > 0:
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(residuals)), residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Sample Index')
        plt.ylabel('Residual (True - Predicted)')
        plt.title(f'Residual Plot - {dataset_name}')
        plt.grid(True)
        plot_path = os.path.join(log_dir, f'residual_plot_{dataset_name.lower()}.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Residual plot for {dataset_name} saved to {plot_path}")

def plot_confusion_matrix(metrics_dict, dataset_name, log_dir):
    conf_matrix = metrics_dict['conf_matrix']
    if conf_matrix.size > 0:
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {dataset_name}')
        plot_path = os.path.join(log_dir, f'confusion_matrix_{dataset_name.lower()}.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Confusion matrix for {dataset_name} saved to {plot_path}")

# Function to compute intermediate features with corrected scatter_add_
def compute_intermediate_features(batch, model, device):
    batch = batch.to(device)
    with torch.no_grad():
        # Compute raw features
        _, prot_graph_raw = model.prot_graph_enc(batch)
        _, lig_graph_raw = model.lig_graph_enc(batch)
        
        # Ensure all attributes are tensors and log their shapes
        logging.debug(f"batch.prot_batch shape: {batch.prot_batch.shape}, max: {batch.prot_batch.max()}")
        logging.debug(f"batch.clique_batch shape: {batch.clique_batch.shape}, max: {batch.clique_batch.max()}")
        logging.debug(f"batch.mol_batch shape: {batch.mol_batch.shape}, max: {batch.mol_batch.max()}")
        
        # Pool protein edge features
        num_graphs = batch.prot_batch.max().item() + 1
        prot_edge_weight_pooled = torch.zeros(num_graphs, 1, device=device, dtype=torch.float32)
        num_edges = batch.prot_edge_index.size(1)
        
        if batch.prot_edge_weight.size(0) != num_edges:
            batch.prot_edge_weight = batch.prot_edge_weight[:num_edges] if batch.prot_edge_weight.size(0) > num_edges else torch.cat([batch.prot_edge_weight, torch.ones(num_edges - batch.prot_edge_weight.size(0), device=device)])
        
        valid_prot_edge_mask = (batch.prot_edge_index[0] < num_graphs) & (batch.prot_edge_index[1] < num_graphs)
        prot_edge_index_filtered = batch.prot_edge_index[:, valid_prot_edge_mask]
        prot_edge_weight_filtered = batch.prot_edge_weight[valid_prot_edge_mask]
        
        if prot_edge_index_filtered.size(1) > 0:
            prot_indices = batch.prot_batch[prot_edge_index_filtered[0]].unsqueeze(-1)
            prot_edge_weight_pooled = prot_edge_weight_pooled.scatter_add_(0, prot_indices, prot_edge_weight_filtered.unsqueeze(-1))
            counts = torch.bincount(batch.prot_batch[prot_edge_index_filtered[0]], minlength=num_graphs).float().clamp(min=1).unsqueeze(-1)
            prot_edge_weight_pooled = prot_edge_weight_pooled / counts
        
        # Pool clique edge features
        num_graphs_clique = batch.clique_batch.max().item() + 1
        clique_edge_weight_pooled = torch.zeros(num_graphs_clique, 1, device=device, dtype=torch.float32)
        num_edges_clique = batch.clique_edge_index.size(1)
        
        if batch.clique_edge_weight.size(0) != num_edges_clique:
            batch.clique_edge_weight = batch.clique_edge_weight[:num_edges_clique] if batch.clique_edge_weight.size(0) > num_edges_clique else torch.cat([batch.clique_edge_weight, torch.ones(num_edges_clique - batch.clique_edge_weight.size(0), device=device)])
        
        valid_clique_edge_mask = (batch.clique_edge_index[0] < num_graphs_clique) & (batch.clique_edge_index[1] < num_graphs_clique)
        clique_edge_index_filtered = batch.clique_edge_index[:, valid_clique_edge_mask]
        clique_edge_weight_filtered = batch.clique_edge_weight[valid_clique_edge_mask]
        
        if clique_edge_index_filtered.size(1) > 0:
            clique_indices = batch.clique_batch[clique_edge_index_filtered[0]].unsqueeze(-1)
            clique_edge_weight_pooled = clique_edge_weight_pooled.scatter_add_(0, clique_indices, clique_edge_weight_filtered.unsqueeze(-1))
            counts_clique = torch.bincount(batch.clique_batch[clique_edge_index_filtered[0]], minlength=num_graphs_clique).float().clamp(min=1).unsqueeze(-1)
            clique_edge_weight_pooled = clique_edge_weight_pooled / counts_clique
        
        # Pool node features
        prot_node_evo_pooled = global_max_pool(batch.prot_node_evo, batch.prot_batch)
        prot_node_aa_pooled = global_max_pool(batch.prot_node_aa, batch.prot_batch)
        prot_one_hot_pooled = global_max_pool(batch.prot_one_hot, batch.prot_batch)  
        mol_x_feat_pooled = global_max_pool(batch.mol_x_feat, batch.mol_batch)
        clique_x_pooled = global_max_pool(batch.clique_x, batch.clique_batch)
        
        # Pool molecular edge features
        num_nodes_mol = batch.mol_batch.size(0)
        mol_edge_attr_expanded = batch.mol_edge_attr.unsqueeze(-1)
        valid_mol_edge_mask = (batch.mol_edge_index[1] < num_nodes_mol)
        mol_edge_index_filtered = batch.mol_edge_index[:, valid_mol_edge_mask]
        mol_edge_attr_filtered = mol_edge_attr_expanded[valid_mol_edge_mask]
        
        mol_node_edge_attr = torch.zeros(num_nodes_mol, 1, device=device, dtype=torch.float32)
        mol_node_edge_attr = mol_node_edge_attr.scatter_add_(0, mol_edge_index_filtered[1].unsqueeze(-1), mol_edge_attr_filtered)
        counts_mol_edges = torch.bincount(mol_edge_index_filtered[1], minlength=num_nodes_mol).float().clamp(min=1).unsqueeze(-1)
        mol_node_edge_attr = mol_node_edge_attr / counts_mol_edges
        mol_edge_attr_pooled = global_max_pool(mol_node_edge_attr, batch.mol_batch)
        
        # Flatten sequence features
        prot_x_protein_pooled = batch.prot_x_protein.view(batch.prot_x_protein.size(0), -1)
        chemBERTa_pooled = batch.chemBERTa.squeeze(1) if batch.chemBERTa.dim() == 3 else batch.chemBERTa
        smiles_atomic_pooled = batch.smiles_atomic  
        
        # Project features to scalars
        proj_prot_node_evo = nn.Linear(1280, 1).to(device)(prot_node_evo_pooled)
        proj_prot_node_aa = nn.Linear(9, 1).to(device)(prot_node_aa_pooled)
        proj_prot_one_hot = nn.Linear(21, 1).to(device)(prot_one_hot_pooled)  
        prot_x_protein_dim = prot_x_protein_pooled.size(1)
        proj_prot_x_protein = nn.Linear(prot_x_protein_dim, 1).to(device)(prot_x_protein_pooled)
        proj_mol_x_feat = nn.Linear(43, 1).to(device)(mol_x_feat_pooled)
        proj_mol_edge_attr = nn.Linear(1, 1).to(device)(mol_edge_attr_pooled)
        proj_clique_x = nn.Linear(1, 1).to(device)(clique_x_pooled)
        proj_chemBERTa = nn.Linear(768, 1).to(device)(chemBERTa_pooled)
        proj_smiles_atomic = nn.Linear(128, 1).to(device)(smiles_atomic_pooled)  
        proj_prot_edge_weight = nn.Linear(1, 1).to(device)(prot_edge_weight_pooled)
        proj_clique_edge_weight = nn.Linear(1, 1).to(device)(clique_edge_weight_pooled)
        
        # Concatenate projected features
        intermediate_features = torch.cat([
            proj_prot_node_evo, proj_prot_node_aa, proj_prot_one_hot, proj_prot_x_protein,
            proj_mol_x_feat, proj_mol_edge_attr, proj_clique_x, proj_chemBERTa, proj_smiles_atomic,
            proj_prot_edge_weight, proj_clique_edge_weight
        ], dim=-1)
        logging.info(f"intermediate_features shape: {intermediate_features.shape}")
        
    return intermediate_features

def shap_summary(model, loader, device, log_dir):
    logging.info("Starting SHAP summary computation...")
    model.eval()

    # Get the background dataset
    background_batch = next(iter(loader)).to(device)
    background_features = compute_intermediate_features(background_batch, model, device)
    logging.info(f"Background features computed: {background_features.shape}")

    # Make sure the background dataset has the correct shape
    if background_features.shape[1] != 11:
        logging.error(f"Expected 11 features in background_features, but got {background_features.shape[1]}")
        raise ValueError(f"Feature dimension mismatch: expected 11, got {background_features.shape[1]}")

    # Define a model wrapper function for use by SHAP
    def model_wrapper(features):
        features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
        with torch.no_grad():
            reg_output, _ = model(intermediate_features=features_tensor)
        return reg_output.cpu().numpy() 

    # Creating an explainer using shap.Explainer
    explainer = shap.Explainer(model_wrapper, background_features.cpu().numpy())
    logging.info("SHAP explainer initialized.")

    # Calculate expected_value in advance
    background_predictions = model_wrapper(background_features.cpu().numpy())
    expected_value = np.mean(background_predictions)
    logging.info(f"Expected value for SHAP plots: {expected_value}")

    shap_values_list = []
    features_list = []
    id_list = []  
    feature_names = [
        'prot_node_evo', 'prot_node_aa', 'prot_one_hot', 'prot_x_protein',
        'mol_x_feat', 'mol_edge_attr', 'clique_x', 'chemBERTa', 'smiles_atomic',
        'prot_edge_weight', 'clique_edge_weight'
    ]

    # Check if the number of feature_names matches
    if len(feature_names) != background_features.shape[1]:
        logging.error(f"Feature names count ({len(feature_names)}) does not match feature dimension ({background_features.shape[1]})")
        raise ValueError(f"Feature names count ({len(feature_names)}) does not match feature dimension ({background_features.shape[1]})")

    # Iterate over the data loader and calculate the SHAP value
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch_features = compute_intermediate_features(batch, model, device)
            shap_values = explainer(batch_features.cpu().numpy())
            shap_values_list.append(shap_values.values)
            features_list.append(batch_features.cpu().numpy())

            # Extract the ID in the batch
            if hasattr(batch, 'id'):
                batch_ids = batch.id
                if isinstance(batch_ids, torch.Tensor):
                    batch_ids = batch_ids.cpu().numpy().tolist()
                elif not isinstance(batch_ids, list):
                    batch_ids = [batch_ids] * batch_features.shape[0]
            else:
                logging.warning("Batch does not have 'id' attribute. Using 'Unknown' as ID.")
                batch_ids = ['Unknown'] * batch_features.shape[0]
            id_list.extend(batch_ids)

    # Merge all SHAP values ​​and features
    shap_values = np.concatenate(shap_values_list, axis=0)
    features = np.concatenate(features_list, axis=0)
    logging.info(f"SHAP values shape: {shap_values.shape}")
    logging.info(f"All features shape: {features.shape}")
    logging.info(f"Number of IDs collected: {len(id_list)}")

    # Make sure the number of IDs matches the number of samples
    if len(id_list) != shap_values.shape[0]:
        logging.warning(f"ID count ({len(id_list)}) does not match sample count ({shap_values.shape[0]}). Padding with 'Unknown'.")
        id_list = id_list + ['Unknown'] * (shap_values.shape[0] - len(id_list))

    # 1. Top 10 samples with the largest overall positive SHAP contribution
    total_positive_shap_per_sample = np.sum(np.maximum(shap_values, 0), axis=1)
    top_10_indices_overall = np.argsort(total_positive_shap_per_sample)[::-1][:10]
    top_10_ids_overall = [id_list[idx] for idx in top_10_indices_overall]

    # 2. Top 10 samples with the largest positive SHAP contribution of prot_node_aa feature
    prot_node_aa_idx = feature_names.index('prot_node_aa')  # Index 1
    prot_node_aa_shap = shap_values[:, prot_node_aa_idx]
    positive_prot_node_aa_shap = np.maximum(prot_node_aa_shap, 0)
    top_10_indices_prot_node_aa = np.argsort(positive_prot_node_aa_shap)[::-1][:10]
    top_10_ids_prot_node_aa = [id_list[idx] for idx in top_10_indices_prot_node_aa]

    # 3. prot_x_protein  Top 10 
    prot_x_protein_idx = feature_names.index('prot_x_protein')  # Index 3
    prot_x_protein_shap = shap_values[:, prot_x_protein_idx]
    positive_prot_x_protein_shap = np.maximum(prot_x_protein_shap, 0)
    top_10_indices_prot_x_protein = np.argsort(positive_prot_x_protein_shap)[::-1][:10]
    top_10_ids_prot_x_protein = [id_list[idx] for idx in top_10_indices_prot_x_protein]

    # 4. Cross S2G (mol_x_feat)  Top 10 
    cross_s2g_idx = feature_names.index('mol_x_feat')  # Index 4
    cross_s2g_shap = shap_values[:, cross_s2g_idx]
    positive_cross_s2g_shap = np.maximum(cross_s2g_shap, 0)
    top_10_indices_cross_s2g = np.argsort(positive_cross_s2g_shap)[::-1][:10]
    top_10_ids_cross_s2g = [id_list[idx] for idx in top_10_indices_cross_s2g]

    # 5. Early Fusion (prot_node_evo)  Top 10
    early_fusion_idx = feature_names.index('prot_node_evo')  # Index 0
    early_fusion_shap = shap_values[:, early_fusion_idx]
    positive_early_fusion_shap = np.maximum(early_fusion_shap, 0)
    top_10_indices_early_fusion = np.argsort(positive_early_fusion_shap)[::-1][:10]
    top_10_ids_early_fusion = [id_list[idx] for idx in top_10_indices_early_fusion]

    # Combine into a DataFrame and save
    max_len = 10
    data = {
        'Top_10_Overall_ID': top_10_ids_overall + [''] * (max_len - len(top_10_ids_overall)),
        'Top_10_Prot_Node_AA_ID': top_10_ids_prot_node_aa + [''] * (max_len - len(top_10_ids_prot_node_aa)),
        'Top_10_Prot_X_Protein_ID': top_10_ids_prot_x_protein + [''] * (max_len - len(top_10_ids_prot_x_protein)),
        'Top_10_Cross_S2G_ID': top_10_ids_cross_s2g + [''] * (max_len - len(top_10_ids_cross_s2g)),
        'Top_10_Early_Fusion_ID': top_10_ids_early_fusion + [''] * (max_len - len(top_10_ids_early_fusion))
    }
    top_10_df = pd.DataFrame(data)
    os.makedirs(log_dir, exist_ok=True)  
    csv_path = os.path.join(log_dir, 'top_10_shap_ids.csv')
    top_10_df.to_csv(csv_path, index=False)
    logging.info(f"Top 10 IDs for overall, prot_node_aa, prot_x_protein, Cross S2G, and Early Fusion SHAP contributions saved to {csv_path}")

    # Call the visualization function, passing expected_value
    try:
        shap_summary_bar(shap_values, features, feature_names, os.path.join(log_dir, 'shap_summary_multi_task.png'))
        shap_density_scatter(shap_values, features, feature_names, os.path.join(log_dir, 'shap_density_scatter.png'))
        shap_hierarchical_clustering(shap_values, features, feature_names, os.path.join(log_dir, 'shap_hierarchical_clustering.png'))
        shap_dependence_plots(shap_values, features, feature_names, os.path.join(log_dir, 'shap'))
        shap_force_plot(shap_values, features, feature_names, expected_value, os.path.join(log_dir, 'shap_force_plot.png'))
        shap_decision_plot(shap_values, features, feature_names, expected_value, os.path.join(log_dir, 'shap_decision_plot.png'), num_samples=50)
        # New SHAP heatmap
        shap_heatmap_plot(shap_values, features, feature_names, os.path.join(log_dir, 'shap_heatmap_plot.png'))
    except Exception as e:
        logging.error(f"Error during SHAP visualization: {str(e)}")
        raise

def shap_waterfall(model, loader, device, log_dir, sample_idx=0):
    logging.info("Starting SHAP waterfall computation...")
    model.eval()

    # Get the background dataset
    background_batch = next(iter(loader)).to(device)
    background_features = compute_intermediate_features(background_batch, model, device)

    # Define the model wrapper function
    def model_wrapper(features):
        features_tensor = torch.tensor(features, dtype=torch.float32, device=device)
        with torch.no_grad():
            reg_output, _ = model(intermediate_features=features_tensor)
        return reg_output.cpu().numpy()

    # Creating a SHAP Interpreter
    explainer = shap.Explainer(model_wrapper, background_features.cpu().numpy())
    batch = next(iter(loader)).to(device)
    batch_features = compute_intermediate_features(batch, model, device)

    # Calculating SHAP Values
    shap_values = explainer(batch_features.cpu().numpy())

    # Manually calculate the expected value of the background dataset
    background_predictions = model_wrapper(background_features.cpu().numpy())
    expected_value = np.mean(background_predictions)
    logging.info(f"Expected value for waterfall plot: {expected_value}")

    # Make sure feature_names and shap_summary are consistent
    feature_names = [
        'prot_node_evo', 'prot_node_aa', 'prot_one_hot', 'prot_x_protein',
        'mol_x_feat', 'mol_edge_attr', 'clique_x', 'chemBERTa', 'smiles_atomic',
        'prot_edge_weight', 'clique_edge_weight'
    ]

    # Check if the number of feature_names matches
    if len(feature_names) != batch_features.shape[1]:
        logging.error(f"Feature names count ({len(feature_names)}) does not match feature dimension ({batch_features.shape[1]})")
        raise ValueError(f"Feature names count ({len(feature_names)}) does not match feature dimension ({batch_features.shape[1]})")

    # Generate a waterfall chart
    os.makedirs(log_dir, exist_ok=True)
    shap_waterfall_plot(shap_values.values, batch_features.cpu().numpy(), feature_names, expected_value, sample_idx,
                        os.path.join(log_dir, 'shap_waterfall_multi_task.png'))
    logging.info(f"SHAP waterfall plot saved to {os.path.join(log_dir, 'shap_waterfall_multi_task.png')}")

def visualize_model_attentions(model, batch, log_dir, epoch):
    logging.info("Starting attention visualization...")
    epoch_dir = os.path.join(log_dir, f"epoch_{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Move to CPU to avoid GPU memory issues
    model.to('cpu')
    batch = batch.to('cpu')
    model.eval()
    
    logging.info("Running model forward pass for attention weights...")
    with torch.no_grad():
        reg_output, cls_output = model(batch)
        logging.info(f"Forward pass completed. Regression output shape: {reg_output.shape}, "
                     f"Classification output shape: {cls_output.shape}")
    
    logging.info("Calling visualize_all_attentions...")
    visualize_all_attentions(model, batch, os.path.join(epoch_dir, "combined_attention.png"))
    logging.info("Attention visualization completed.")
    
    # Move model back to GPU and clear memory
    model.to(device)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        # Load datasets (unchanged)
        datasets = {
            'train_reg': ProteinLigandDataset(
                pkl_path="pdb2020_train.pkl",
                normalize_affinity=True, training=True, noise_level=0.5, use_augmentation=True,
                aug_prob=0.3, clean_and_balance=True, z_threshold=3.0, n_bins=5
            ),
            'val_reg': ProteinLigandDataset(
                pkl_path="pdb2020_val.pkl",
                normalize_affinity=True, training=False
            ),
            'test_reg': ProteinLigandDataset(
                pkl_path="CASF_core2016.pkl",
                normalize_affinity=True, training=False
            ),
            'train_cls': ProteinLigandDataset(
                pkl_path="human_train.pkl",
                training=True, use_augmentation=True, aug_prob=0.3, clean_and_balance=True,
                is_classification=True
            ),
            'val_cls': ProteinLigandDataset(
                pkl_path="/human_val.pkl",
                training=False, is_classification=True
            ),
            'test_cls': ProteinLigandDataset(
                pkl_path="human_test.pkl",
                training=False, is_classification=True
            )
        }

        batch_size = 4
        accum_steps = 8
        loaders = {
            key: DataLoader(dataset, batch_size=batch_size, shuffle='train' in key, collate_fn=custom_collate_fn)
            for key, dataset in datasets.items()
        }

        model = FusedAffinityPredictor(
            atom_feat_dim=43,
            mol_edge_attr_dim=1,
            hidden_dim=128,
            chemberta_dim=768,
            gat_heads_protein=2,
            gat_heads_ligand=2,
            transformer_nhead=2,
            cross_heads=8,
            input_dropout=0.22129538276104416,
            hidden_dropout=0.2
        ).to(device)
        logging.info("Model initialized with optimal hyperparameters: "
                     f"gat_heads_protein={2}, gat_heads_ligand={2}, transformer_nhead={2}, cross_heads={8}")

        optimizer = torch.optim.Adam(model.parameters(), lr=0.00027813723109127255, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        criterion_reg = WeightedMSELoss(alpha=1.0, gamma=1.5, max_weight=1000.0)
        criterion_cls = nn.BCEWithLogitsLoss()
        multi_task_loss = MultiTaskLossWrapper(task_num=2).to(device)

        best_val_loss = float('inf')
        patience = 10
        counter = 0
        train_losses_reg, train_losses_cls = [], []
        val_losses_reg, val_losses_cls = [], []
        train_metrics_cls = {'accuracy': [], 'roc_auc': [], 'f1': []}
        val_metrics_cls = {'accuracy': [], 'roc_auc': [], 'f1': []}
        best_metrics = {'val_r2': 0.0, 'val_adjusted_r2': 0.0, 'val_mse': float('inf'), 
                        'val_rmse': float('inf'), 'val_mae': float('inf'), 'val_pcc': 0.0,
                        'val_spearman': 0.0, 'val_ci': 0.0, 'val_accuracy': 0.0, 'val_roc_auc': 0.0, 'val_f1': 0.0}

        num_epochs = 100
        for epoch in range(num_epochs):
            model.train()
            total_loss_reg, total_loss_cls = 0, 0
            optimizer.zero_grad()
            
            max_steps = max(len(loaders['train_reg']), len(loaders['train_cls']))
            train_reg_iter = iter(loaders['train_reg'])
            train_cls_iter = iter(loaders['train_cls'])
            
            for i in tqdm(range(max_steps), desc=f"Epoch {epoch + 1}/{num_epochs}"):
                batch_reg = next(train_reg_iter, None)
                batch_cls = next(train_cls_iter, None)

                if batch_reg is not None:
                    batch_reg = batch_reg.to(device)
                    reg_output, cls_output = model(batch_reg)
                    loss_reg = criterion_reg(reg_output, batch_reg.regression_label.view(-1, 1))
                    total_loss_reg += loss_reg.item() * batch_size
                else:
                    reg_output = torch.zeros(batch_size, 1, device=device)
                    cls_output = torch.zeros(batch_size, 1, device=device)
                    loss_reg = torch.tensor(0.0, device=device)

                if batch_cls is not None:
                    batch_cls = batch_cls.to(device)
                    reg_output_cls, cls_output = model(batch_cls)
                    loss_cls = criterion_cls(cls_output, batch_cls.classification_label.view(-1, 1).float())
                    total_loss_cls += loss_cls.item() * batch_size
                else:
                    cls_output = torch.zeros(batch_size, 1, device=device)
                    loss_cls = torch.tensor(0.0, device=device)

                total_loss, loss_reg_weighted, loss_cls_weighted = multi_task_loss(
                    reg_output, cls_output,
                    batch_reg.regression_label.view(-1, 1) if batch_reg else torch.zeros(batch_size, 1, device=device),
                    batch_cls.classification_label.view(-1, 1).float() if batch_cls else torch.zeros(batch_size, 1, device=device, dtype=torch.float),
                    criterion_reg, criterion_cls
                )
                total_loss = total_loss / accum_steps
                total_loss.backward()

                if (i + 1) % accum_steps == 0 or (i + 1) == max_steps:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    optimizer.zero_grad()

            train_losses_reg.append(total_loss_reg / (len(datasets['train_reg']) if len(datasets['train_reg']) > 0 else 1))
            train_losses_cls.append(total_loss_cls / (len(datasets['train_cls']) if len(datasets['train_cls']) > 0 else 1))
            logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Train Reg Loss: {train_losses_reg[-1]:.4f}, "
                         f"Train Cls Loss: {train_losses_cls[-1]:.4f}")

            model.eval()
            val_loss_reg, val_loss_cls = 0, 0
            with torch.no_grad():
                for batch in loaders['val_reg']:
                    batch = batch.to(device)
                    reg_output, cls_output = model(batch)
                    loss = criterion_reg(reg_output, batch.regression_label.view(-1, 1))
                    val_loss_reg += loss.item()
                for batch in loaders['val_cls']:
                    batch = batch.to(device)
                    reg_output, cls_output = model(batch)
                    loss = criterion_cls(cls_output, batch.classification_label.view(-1, 1).float())
                    val_loss_cls += loss.item()

            val_losses_reg.append(val_loss_reg / len(loaders['val_reg']))
            val_losses_cls.append(val_loss_cls / len(loaders['val_cls']))
            total_val_loss = val_losses_reg[-1] + val_losses_cls[-1]
            logging.info(f"Epoch [{epoch + 1}/{num_epochs}], Val Reg Loss: {val_losses_reg[-1]:.4f}, "
                         f"Val Cls Loss: {val_losses_cls[-1]:.4f}, Total Val Loss: {total_val_loss:.4f}")

            logging.info("Evaluating validation metrics...")
            train_metrics_reg = evaluate_model(model, loaders['train_reg'], device)
            val_metrics_reg = evaluate_model(model, loaders['val_reg'], device)
            train_metrics_cls_current = evaluate_model(model, loaders['train_cls'], device, is_classification=True)
            val_metrics_cls_current = evaluate_model(model, loaders['val_cls'], device, is_classification=True)
            logging.info("Validation metrics computed.")

            # Append classification metrics for plotting
            train_metrics_cls['accuracy'].append(train_metrics_cls_current['accuracy'])
            train_metrics_cls['roc_auc'].append(train_metrics_cls_current['roc_auc'])
            train_metrics_cls['f1'].append(train_metrics_cls_current['f1'])
            val_metrics_cls['accuracy'].append(val_metrics_cls_current['accuracy'])
            val_metrics_cls['roc_auc'].append(val_metrics_cls_current['roc_auc'])
            val_metrics_cls['f1'].append(val_metrics_cls_current['f1'])

            # Restore visualization with smaller batch
            if total_val_loss < best_val_loss or epoch == num_epochs - 1:
                logging.info("Starting attention visualization for best or last epoch...")
                val_loader_small = DataLoader(loaders['val_reg'].dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
                val_batch = next(iter(val_loader_small))
                visualize_model_attentions(model, val_batch, log_dir, epoch)
                logging.info("Attention visualization completed.")

            # Update best metrics
            current_val_r2 = val_metrics_reg.get('r2', 0.0)
            current_val_adjusted_r2 = val_metrics_reg.get('adjusted_r2', 0.0)
            current_val_mse = val_metrics_reg.get('mse', float('inf'))
            current_val_rmse = val_metrics_reg.get('rmse', float('inf'))
            current_val_mae = val_metrics_reg.get('mae', float('inf'))
            current_val_pcc = val_metrics_reg.get('pcc', 0.0)
            current_val_spearman = val_metrics_reg.get('spearman', 0.0)
            current_val_ci = val_metrics_reg.get('ci', 0.0)
            current_val_accuracy = val_metrics_cls_current.get('accuracy', 0.0)
            current_val_roc_auc = val_metrics_cls_current.get('roc_auc', 0.0)
            current_val_f1 = val_metrics_cls_current.get('f1', 0.0)

            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                best_metrics.update({
                    'val_r2': current_val_r2, 'val_adjusted_r2': current_val_adjusted_r2, 'val_mse': current_val_mse,
                    'val_rmse': current_val_rmse, 'val_mae': current_val_mae, 'val_pcc': current_val_pcc,
                    'val_spearman': current_val_spearman, 'val_ci': current_val_ci, 'val_accuracy': current_val_accuracy,
                    'val_roc_auc': current_val_roc_auc, 'val_f1': current_val_f1
                })
                torch.save(model.state_dict(), os.path.join(log_dir, 'best_model_multi_task.pth'))
                logging.info(f"New best model saved at epoch {epoch + 1} with Val Loss: {best_val_loss:.4f}")
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    logging.info("Early stopping triggered")
                    break

            scheduler.step()

        logging.info("Plotting losses...")
        plot_losses(train_losses_reg, train_losses_cls, val_losses_reg, val_losses_cls, log_dir)
        logging.info("Loss plotting completed.")

        logging.info("Loading best model for final evaluation...")
        model.load_state_dict(torch.load(os.path.join(log_dir, 'best_model_multi_task.pth')))
        metrics = {
            split: evaluate_model(model, loaders[split], device, 'cls' in split)
            for split in ['train_reg', 'val_reg', 'test_reg', 'train_cls', 'val_cls', 'test_cls']
        }
        logging.info("Final evaluation completed.")

        logging.info("Final Evaluation Metrics:")
        for split in ['train_reg', 'val_reg', 'test_reg']:
            m = metrics[split]
            logging.info(f"{split.replace('_reg', '')} Regression - R2: {m['r2']:.4f}, Adjusted R2: {m['adjusted_r2']:.4f}, "
                         f"MSE: {m['mse']:.4f}, RMSE: {m['rmse']:.4f}, MAE: {m['mae']:.4f}, "
                         f"PCC: {m['pcc']:.4f}, Spearman: {m['spearman']:.4f}, CI: {m['ci']:.4f}")
        for split in ['train_cls', 'val_cls', 'test_cls']:
            m = metrics[split]
            logging.info(f"{split.replace('_cls', '')} Classification - Accuracy: {m['accuracy']:.4f}, "
                         f"ROC-AUC: {m['roc_auc']:.4f}, F1: {m['f1']:.4f}, AP: {m['ap']:.4f}")

        # Plot additional visualizations for regression
        logging.info("Generating regression visualizations...")
        for split in ['train_reg', 'val_reg', 'test_reg']:
            dataset_name = split.replace('_reg', '')
            plot_combined_affinity_error(metrics[split], dataset_name, log_dir)
            plot_residuals(metrics[split], dataset_name, log_dir)

        # Plot classification metrics over epochs
        logging.info("Plotting classification metrics over epochs...")
        plot_classification_metrics(train_metrics_cls, val_metrics_cls, log_dir)

        # Plot classification visualizations
        logging.info("Generating classification visualizations...")
        for split in ['train_cls', 'val_cls', 'test_cls']:
            dataset_name = split.replace('_cls', '')
            plot_roc_curve(metrics[split]['fpr'], metrics[split]['tpr'], metrics[split]['roc_auc'], dataset_name, log_dir)
            plot_precision_recall_curve(metrics[split]['precision'], metrics[split]['recall'], metrics[split]['ap'], dataset_name, log_dir)
            plot_confusion_matrix(metrics[split], dataset_name, log_dir)

        # Perform SHAP analysis
        logging.info("Starting SHAP analysis...")
        shap_summary(model, loaders['val_reg'], device, log_dir)
        shap_waterfall(model, loaders['val_reg'], device, log_dir, sample_idx=0)
        logging.info("SHAP analysis completed.")

        final_model_path = os.path.join(log_dir, "fused_affinity_predictor_multi_task.pth")
        torch.save(model.state_dict(), final_model_path)
        logging.info(f"Final model saved successfully to {final_model_path}")

    except Exception as e:
        logging.error(f"Training process failed: {e}")